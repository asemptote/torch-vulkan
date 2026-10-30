// Matrix products for device "vk": mm, bmm, addmm, baddbmm, addmv, dot.
//
// All of them run shaders/gemm.slang on 3-D views [batch, rows, cols]:
// operands are passed by strides, so transposed (e.g. from autograd) and
// broadcast operands need no copies. float32 and float64; other dtypes and
// shapes the kernel cannot address run on the CPU.

#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cstddef>

namespace vk {
namespace {

struct GemmParams {
  uint64_t a, b, c, cin;
  double alpha, beta;
  uint32_t m, n, k, pad;
  int32_t a_sm, a_sk, a_sb;
  int32_t b_sk, b_sn, b_sb;
  int32_t c_sm, c_sn, c_sb;
  int32_t cin_sm, cin_sn, cin_sb;
};
static_assert(sizeof(GemmParams) == 112 && offsetof(GemmParams, a_sm) == 64, "must match shaders/gemm.slang");

constexpr int64_t kTile = 64;  // BM = BN in shaders/gemm.slang
constexpr int64_t kMaxGroups = 65535;

bool gemm_dtype(at::ScalarType t) {
  return t == at::kFloat || (t == at::kDouble && vkt_has_float64()) || t == at::kInt || t == at::kLong;
}

// Largest element offset reachable through a 3-D view, or -1 if a stride is
// negative.
int64_t max_offset(const at::Tensor& t) {
  if (t.numel() == 0) {
    return 0;  // never read (e.g. K == 0)
  }
  int64_t span = 0;
  for (int64_t d = 0; d < t.dim(); ++d) {
    if (t.stride(d) < 0) {
      return -1;
    }
    span += (t.size(d) - 1) * t.stride(d);
  }
  return span;
}

// c = alpha * a @ b (+ beta * cin), all 3-D views: a [B, M, K], b [B, K, N],
// c and cin [B, M, N] (cin may broadcast through zero strides). Returns false
// if the kernel cannot handle the case.
bool gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin, double alpha,
          double beta) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2), k = a.size(2);
  if (!gemm_dtype(c.scalar_type()) || a.scalar_type() != c.scalar_type() || b.scalar_type() != c.scalar_type() ||
      (cin != nullptr && cin->scalar_type() != c.scalar_type())) {
    return false;
  }
  for (const at::Tensor* t : {&a, &b, &c, cin}) {
    if (t != nullptr && (!is_vk(*t) || max_offset(*t) < 0 || max_offset(*t) >= (int64_t(1) << 31))) {
      return false;
    }
  }
  int64_t gx = (n + kTile - 1) / kTile, gy = (m + kTile - 1) / kTile;
  if (gx > kMaxGroups || gy > kMaxGroups || batch > kMaxGroups) {
    return false;
  }
  if (c.numel() == 0) {
    return true;
  }
  bool use_cin = cin != nullptr && beta != 0.0;
  GemmParams p{};
  p.a = gpu_address(a);
  p.b = gpu_address(b);
  p.c = gpu_address(c);
  p.cin = use_cin ? gpu_address(*cin) : 0;
  p.alpha = alpha;
  p.beta = beta;
  p.m = static_cast<uint32_t>(m);
  p.n = static_cast<uint32_t>(n);
  p.k = static_cast<uint32_t>(k);
  p.a_sb = static_cast<int32_t>(a.stride(0));
  p.a_sm = static_cast<int32_t>(a.stride(1));
  p.a_sk = static_cast<int32_t>(a.stride(2));
  p.b_sb = static_cast<int32_t>(b.stride(0));
  p.b_sk = static_cast<int32_t>(b.stride(1));
  p.b_sn = static_cast<int32_t>(b.stride(2));
  p.c_sb = static_cast<int32_t>(c.stride(0));
  p.c_sm = static_cast<int32_t>(c.stride(1));
  p.c_sn = static_cast<int32_t>(c.stride(2));
  if (use_cin) {
    p.cin_sb = static_cast<int32_t>(cin->stride(0));
    p.cin_sm = static_cast<int32_t>(cin->stride(1));
    p.cin_sn = static_cast<int32_t>(cin->stride(2));
  }
  dispatch(std::string("gemm_") + dtype_label(c.scalar_type()), {use_cin ? 1u : 0u}, p,
           static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), static_cast<uint32_t>(batch));
  return true;
}

// Complex c = alpha a @ b (+ beta cin): four real GEMMs on the real and
// imaginary parts (strided views, no copies), then complex element-wise ops
// for alpha and beta.
bool complex_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin,
                  const at::Scalar& alpha, const at::Scalar& beta) {
  at::ScalarType t = c.scalar_type();
  if (!at::isComplexType(t) || dtype_label(t) == nullptr || a.scalar_type() != t || b.scalar_type() != t ||
      !is_vk(a) || !is_vk(b) || !is_vk(c)) {
    return false;
  }
  at::Tensor ar = at::view_as_real(a.resolve_conj()), br = at::view_as_real(b.resolve_conj());
  at::Tensor a_re = ar.select(-1, 0), a_im = ar.select(-1, 1), b_re = br.select(-1, 0), b_im = br.select(-1, 1);
  at::TensorOptions real = c.options().dtype(c10::toRealValueType(t));
  at::Tensor pr = at::empty(c.sizes(), real), pi = at::empty(c.sizes(), real);
  bool ok = gemm(a_re, b_re, pr, nullptr, 1.0, 0.0) && gemm(a_im, b_im, pr, &pr, -1.0, 1.0) &&
            gemm(a_re, b_im, pi, nullptr, 1.0, 0.0) && gemm(a_im, b_re, pi, &pi, 1.0, 1.0);
  if (!ok) {
    return false;
  }
  at::Tensor result = at::complex(pr, pi);
  if (!alpha.equal(1)) {
    result = result * alpha;
  }
  if (cin != nullptr && !beta.equal(0)) {
    result = result + *cin * beta;
  }
  c.copy_(result);
  return true;
}

// c = alpha a @ b (+ beta cin) for real or complex 3-D views.
bool any_gemm(const at::Tensor& a_in, const at::Tensor& b_in, const at::Tensor& c, const at::Tensor* cin_in,
              const at::Scalar& alpha, const at::Scalar& beta) {
  // Matrix products handle lazy conjugate / negative inputs themselves in
  // PyTorch (they skip the fallbacks that materialize them): resolve them here.
  at::Tensor a = a_in.resolve_conj().resolve_neg(), b = b_in.resolve_conj().resolve_neg();
  at::Tensor cin_resolved = cin_in != nullptr ? cin_in->resolve_conj().resolve_neg() : at::Tensor();
  const at::Tensor* cin = cin_in != nullptr ? &cin_resolved : nullptr;
  if (at::isComplexType(c.scalar_type())) {
    return complex_gemm(a, b, c, cin, alpha, beta);
  }
  if (alpha.isComplex() || beta.isComplex()) {
    return false;
  }
  return gemm(a, b, c, cin, alpha.toDouble(), beta.toDouble());
}

at::Tensor& mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::resize_output(out, {self.size(0), mat2.size(1)});
  if (!any_gemm(self.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), nullptr, 1, 0)) {
    on_cpu("aten::mm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  if (!any_gemm(self, mat2, out, nullptr, 1, 0)) {
    on_cpu("aten::bmm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  int64_t m = mat1.size(0), n = mat2.size(1);
  at::native::resize_output(out, {m, n});
  bool done = false;
  if (self.dim() <= 2 && is_vk(self)) {
    at::Tensor cin = self.expand({m, n}).unsqueeze(0);
    done = any_gemm(mat1.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), &cin, alpha, beta);
  }
  if (!done) {
    on_cpu("aten::addmm", "out", self, mat1, mat2, beta, alpha, out);
  }
  return out;
}

at::Tensor& baddbmm_out(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
                        const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  std::vector<int64_t> shape = {batch1.size(0), batch1.size(1), batch2.size(2)};
  at::native::resize_output(out, shape);
  bool done = false;
  if (self.dim() <= 3 && is_vk(self)) {
    at::Tensor cin = self.expand(shape);
    done = any_gemm(batch1, batch2, out, &cin, alpha, beta);
  }
  if (!done) {
    on_cpu("aten::baddbmm", "out", self, batch1, batch2, beta, alpha, out);
  }
  return out;
}

at::Tensor& addmv_out(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  int64_t m = mat.size(0);
  at::native::resize_output(out, {m});
  bool done = false;
  if (self.dim() <= 1 && is_vk(self)) {
    // A matrix-vector product is a GEMM with one column.
    at::Tensor cin = self.expand({m}).unsqueeze(1).unsqueeze(0);
    done = any_gemm(mat.unsqueeze(0), vec.unsqueeze(1).unsqueeze(0), out.unsqueeze(1).unsqueeze(0), &cin, alpha,
                    beta);
  }
  if (!done) {
    on_cpu("aten::addmv", "out", self, mat, vec, beta, alpha, out);
  }
  return out;
}

// sum_i batch1[i] @ batch2[i] is one GEMM over the concatenated inner dims:
// [n, B k] @ [B k, m].
at::Tensor& addbmm_out(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2,
                       const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out) {
  TORCH_CHECK(batch1.dim() == 3 && batch2.dim() == 3, "batch1 and batch2 must be 3D tensors");
  int64_t B = batch1.size(0), n = batch1.size(1), k = batch1.size(2), m = batch2.size(2);
  at::Tensor a = batch1.permute({1, 0, 2}).reshape({n, B * k});
  at::Tensor b = batch2.reshape({B * k, m});
  return at::addmm_out(out, self, a, b, beta, alpha);
}

at::Tensor addbmm(const at::Tensor& self, const at::Tensor& batch1, const at::Tensor& batch2, const at::Scalar& beta,
                  const at::Scalar& alpha) {
  at::Tensor out = at::empty({0}, self.options().dtype(at::result_type(batch1, batch2)));
  return addbmm_out(self, batch1, batch2, beta, alpha, out);
}

at::Tensor dot(const at::Tensor& self, const at::Tensor& other) {
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1, "1D tensors expected, but got ", self.dim(), "D and ", other.dim(),
              "D tensors");
  TORCH_CHECK(self.size(0) == other.size(0), "inconsistent tensor size, expected tensor [", self.size(0),
              "] and src [", other.size(0), "] to have the same number of elements, but got ", self.size(0), " and ",
              other.size(0), " elements respectively");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(), " and ", other.scalar_type());
  // Element-wise product and a sum reduction (both Vulkan kernels).
  return at::sum(at::mul(self, other)).to(self.scalar_type());
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("mm.out", &mm_out);
  m.impl("bmm.out", &bmm_out);
  m.impl("addmm.out", &addmm_out);
  m.impl("baddbmm.out", &baddbmm_out);
  m.impl("addmv.out", &addmv_out);
  m.impl("addbmm.out", &addbmm_out);
  m.impl("addbmm", &vk::addbmm);
  m.impl("dot", &dot);
}
