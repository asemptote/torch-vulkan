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
  return t == at::kFloat || (t == at::kDouble && vkt_has_float64());
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

at::Tensor& mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::resize_output(out, {self.size(0), mat2.size(1)});
  if (!gemm(self.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), nullptr, 1.0, 0.0)) {
    on_cpu("aten::mm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  if (!gemm(self, mat2, out, nullptr, 1.0, 0.0)) {
    on_cpu("aten::bmm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  int64_t m = mat1.size(0), n = mat2.size(1);
  at::native::resize_output(out, {m, n});
  bool done = false;
  if (self.dim() <= 2 && is_vk(self) && !alpha.isComplex() && !beta.isComplex()) {
    at::Tensor cin = self.expand({m, n}).unsqueeze(0);
    done = gemm(mat1.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), &cin, alpha.toDouble(), beta.toDouble());
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
  if (self.dim() <= 3 && is_vk(self) && !alpha.isComplex() && !beta.isComplex()) {
    at::Tensor cin = self.expand(shape);
    done = gemm(batch1, batch2, out, &cin, alpha.toDouble(), beta.toDouble());
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
  if (self.dim() <= 1 && is_vk(self) && !alpha.isComplex() && !beta.isComplex()) {
    // A matrix-vector product is a GEMM with one column.
    at::Tensor cin = self.expand({m}).unsqueeze(1).unsqueeze(0);
    done = gemm(mat.unsqueeze(0), vec.unsqueeze(1).unsqueeze(0), out.unsqueeze(1).unsqueeze(0), &cin,
                alpha.toDouble(), beta.toDouble());
  }
  if (!done) {
    on_cpu("aten::addmv", "out", self, mat, vec, beta, alpha, out);
  }
  return out;
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
  m.impl("dot", &dot);
}
