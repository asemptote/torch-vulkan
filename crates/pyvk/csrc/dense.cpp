// Dense linear algebra for device "vk": LU (lu_factor, solve, inv, det,
// slogdet, lu_solve), Cholesky, triangular solves, tril/triu.
//
// Blocked right-looking algorithms with panels of kBlock columns: each panel
// (or diagonal block) is factored by LAPACK on the CPU (memory is shared, so
// this is a copy plus a sync), and the O(n^3) trailing updates run on the GPU
// as GEMMs (csrc/linalg.cpp). Triangular solves apply inverses of the small
// diagonal blocks (computed on the CPU) with GEMMs. Row interchanges from
// pivoting are applied with a gather/scatter kernel (shaders/rows.slang) to
// the rows that actually move.
//
// The GPU path handles single float32/float64 matrices of size >= kMinGpu;
// smaller or batched inputs, other dtypes and unsupported options (QR, eigh,
// SVD, right-side solves, ...) run on the CPU.

#include "vk.h"

#include "opcodes.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace vk {
namespace {

constexpr int64_t kBlock = 64;
constexpr int64_t kMinGpu = 128;  // below this, LAPACK on the CPU is faster

struct RowsParams {
  uint64_t in, out, idx;
  uint32_t nrows, ncols;
  int32_t in_rs, in_cs, out_rs, out_cs;
  uint32_t ngroups, pad;
};
static_assert(sizeof(RowsParams) == 56, "must match shaders/rows.slang");

struct TriangleParams {
  uint64_t in, out;
  uint32_t batch, rows, cols;
  int32_t diag;
  int32_t in_sb, in_sr, in_sc, out_sb, out_sr, out_sc;
  uint32_t ngroups, pad;
};
static_assert(sizeof(TriangleParams) == 64, "must match shaders/triangle.slang");

uint32_t groups_for(int64_t threads) {
  return static_cast<uint32_t>(std::clamp<int64_t>((threads + 255) / 256, 1, 65535));
}

// TORCH_VK_LINALG=cpu sends these ops to LAPACK on the CPU (faster when the
// GPU's GEMM is slower than the CPU's, as on integrated GPUs).
bool linalg_on_cpu() {
  static const bool cpu = [] {
    const char* v = std::getenv("TORCH_VK_LINALG");
    return v != nullptr && std::string(v) == "cpu";
  }();
  return cpu;
}

bool gpu_matrix(const at::Tensor& t) {
  at::ScalarType s = t.scalar_type();
  return !linalg_on_cpu() && is_vk(t) && t.dim() == 2 && t.size(0) == t.size(1) && t.size(0) >= kMinGpu &&
         (s == at::kFloat || (s == at::kDouble && vkt_has_float64()));
}

// A CPU tensor moved to `like`'s device and dtype.
at::Tensor to_vk(const at::Tensor& host, const at::Tensor& like) {
  return host.to(like.options());
}

// ROWS_GATHER: out[i, :] = in[idx[i], :];  ROWS_SCATTER: out[idx[i], :] = in[i, :].
void rows_kernel(const at::Tensor& in, const at::Tensor& out, const std::vector<int64_t>& idx, uint32_t mode) {
  if (idx.empty() || in.size(1) == 0) {
    return;
  }
  at::Tensor index = at::tensor(idx, at::kLong).to(in.device());
  RowsParams p{};
  p.in = gpu_address(in);
  p.out = gpu_address(out);
  p.idx = gpu_address(index);
  p.nrows = static_cast<uint32_t>(idx.size());
  p.ncols = static_cast<uint32_t>(in.size(1));
  p.in_rs = static_cast<int32_t>(in.stride(0));
  p.in_cs = static_cast<int32_t>(in.stride(1));
  p.out_rs = static_cast<int32_t>(out.stride(0));
  p.out_cs = static_cast<int32_t>(out.stride(1));
  p.ngroups = groups_for(static_cast<int64_t>(idx.size()) * in.size(1));
  dispatch(std::string("rows_") + dtype_label(in.scalar_type()), {mode}, p, p.ngroups);
}

// Rows of M are replaced in place: new row i = old row perm[i]. Only rows
// that move are touched.
void permute_rows(const at::Tensor& m, const std::vector<int64_t>& perm) {
  std::vector<int64_t> dst, src;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != static_cast<int64_t>(i)) {
      dst.push_back(static_cast<int64_t>(i));
      src.push_back(perm[i]);
    }
  }
  if (dst.empty()) {
    return;
  }
  at::Tensor moved = at::empty({static_cast<int64_t>(dst.size()), m.size(1)}, m.options());
  rows_kernel(m, moved, src, op::ROWS_GATHER);
  rows_kernel(moved, m, dst, op::ROWS_SCATTER);
}

// Inverse of a small triangular block (on the CPU, in float64 for accuracy).
at::Tensor block_inverse(const at::Tensor& block_vk, bool upper, bool unit) {
  at::Tensor host = block_vk.cpu().to(at::kDouble);
  at::Tensor tri = upper ? host.triu() : host.tril();
  at::Tensor eye = at::eye(tri.size(0), tri.options());
  at::Tensor inv = at::linalg_solve_triangular(tri, eye, upper, /*left=*/true, unit);
  return to_vk(inv.to(block_vk.scalar_type()), block_vk);
}

// Y <- T^{-1} Y for square triangular T (n x n, vk) and Y (n x k, vk).
void triangular_solve_inplace(const at::Tensor& t, const at::Tensor& y, bool upper, bool unit) {
  int64_t n = t.size(0);
  int64_t nblocks = (n + kBlock - 1) / kBlock;
  for (int64_t bi = 0; bi < nblocks; ++bi) {
    // Lower: top-down (forward substitution); upper: bottom-up.
    int64_t blk = upper ? nblocks - 1 - bi : bi;
    int64_t k = blk * kBlock, b = std::min(kBlock, n - k);
    at::Tensor inv = block_inverse(t.narrow(0, k, b).narrow(1, k, b), upper, unit);
    at::Tensor yk = y.narrow(0, k, b);
    yk.copy_(at::mm(inv, yk));
    if (upper && k > 0) {
      y.narrow(0, 0, k).addmm_(t.narrow(0, 0, k).narrow(1, k, b), yk, 1.0, -1.0);
    } else if (!upper && k + b < n) {
      y.narrow(0, k + b, n - k - b).addmm_(t.narrow(0, k + b, n - k - b).narrow(1, k, b), yk, 1.0, -1.0);
    }
  }
}

struct LuInfo {
  std::vector<int64_t> pivots;  // 1-based, LAPACK convention
  int64_t info = 0;
};

// In-place LU with partial pivoting of the square vk matrix a.
LuInfo lu_inplace(const at::Tensor& a) {
  int64_t n = a.size(0);
  LuInfo result;
  result.pivots.resize(n);
  for (int64_t k = 0; k < n; k += kBlock) {
    int64_t b = std::min(kBlock, n - k);
    // Factor the panel A[k:, k:k+b] on the CPU.
    at::Tensor panel = a.narrow(0, k, n - k).narrow(1, k, b);
    auto [lu, piv, info] = at::linalg_lu_factor_ex(panel.cpu());
    int64_t local_info = info.item<int64_t>();
    if (local_info > 0 && result.info == 0) {
      result.info = k + local_info;
    }
    // Apply this panel's interchanges to the full rows k.. of A, then store
    // the factored panel.
    std::vector<int64_t> perm(n - k);
    for (int64_t i = 0; i < n - k; ++i) perm[i] = i;
    auto* pv = piv.data_ptr<int32_t>();
    for (int64_t i = 0; i < b; ++i) {
      std::swap(perm[i], perm[pv[i] - 1]);
      result.pivots[k + i] = k + pv[i];
    }
    permute_rows(a.narrow(0, k, n - k), perm);
    panel.copy_(lu);
    if (k + b < n) {
      // U12 = L11^{-1} A12;  A22 -= L21 U12.
      at::Tensor inv = block_inverse(a.narrow(0, k, b).narrow(1, k, b), /*upper=*/false, /*unit=*/true);
      at::Tensor a12 = a.narrow(0, k, b).narrow(1, k + b, n - k - b);
      a12.copy_(at::mm(inv, a12));
      a.narrow(0, k + b, n - k - b).narrow(1, k + b, n - k - b)
          .addmm_(a.narrow(0, k + b, n - k - b).narrow(1, k, b), a12, 1.0, -1.0);
    }
  }
  return result;
}

// Row permutation (new row i = old row perm[i]) equivalent to the pivots.
std::vector<int64_t> pivots_to_perm(const std::vector<int64_t>& pivots) {
  std::vector<int64_t> perm(pivots.size());
  for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int64_t>(i);
  for (size_t i = 0; i < pivots.size(); ++i) std::swap(perm[i], perm[pivots[i] - 1]);
  return perm;
}

// Solves A X = B in place in x (n x k), given A's LU factors and pivots.
void lu_solve_inplace(const at::Tensor& lu, const std::vector<int64_t>& pivots, const at::Tensor& x) {
  permute_rows(x, pivots_to_perm(pivots));
  triangular_solve_inplace(lu, x, /*upper=*/false, /*unit=*/true);
  triangular_solve_inplace(lu, x, /*upper=*/true, /*unit=*/false);
}

int64_t permutation_sign(const std::vector<int64_t>& pivots) {
  int64_t sign = 1;
  for (size_t i = 0; i < pivots.size(); ++i) {
    if (pivots[i] != static_cast<int64_t>(i) + 1) sign = -sign;
  }
  return sign;
}

void write_pivots(const at::Tensor& out, const std::vector<int64_t>& pivots) {
  at::native::resize_output(out, {static_cast<int64_t>(pivots.size())});
  out.copy_(at::tensor(pivots, at::kLong).to(at::kInt));
}

void write_info(const at::Tensor& out, int64_t info) {
  at::native::resize_output(out, {});
  out.fill_(info);
}

// ---------------------------------------------------------------- aten kernels

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> lu_factor_ex_out(const at::Tensor& a, bool pivot, bool check_errors,
                                                                   at::Tensor& lu, at::Tensor& pivots,
                                                                   at::Tensor& info) {
  if (!pivot || !gpu_matrix(a) || !is_vk(lu)) {
    on_cpu("aten::linalg_lu_factor_ex", "out", a, pivot, check_errors, lu, pivots, info);
    return {lu, pivots, info};
  }
  at::native::resize_output(lu, a.sizes());
  lu.copy_(a);
  LuInfo r = lu_inplace(lu);
  write_pivots(pivots, r.pivots);
  write_info(info, r.info);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.lu_factor_ex", /*is_matrix=*/true);
  }
  return {lu, pivots, info};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> solve_ex_out(
    const at::Tensor& a, const at::Tensor& b, bool left, bool check_errors, at::Tensor& result, at::Tensor& lu,
    at::Tensor& pivots, at::Tensor& info) {
  bool vector_rhs = b.dim() == 1;
  if (!left || !gpu_matrix(a) || !is_vk(b) || (b.dim() != 1 && b.dim() != 2) || b.size(0) != a.size(0) ||
      b.scalar_type() != a.scalar_type() || !is_vk(result)) {
    on_cpu("aten::_linalg_solve_ex", "result", a, b, left, check_errors, result, lu, pivots, info);
    return {result, lu, pivots, info};
  }
  at::native::resize_output(lu, a.sizes());
  lu.copy_(a);
  LuInfo r = lu_inplace(lu);
  write_pivots(pivots, r.pivots);
  write_info(info, r.info);
  at::Tensor x = (vector_rhs ? b.unsqueeze(1) : b).clone(at::MemoryFormat::Contiguous);
  lu_solve_inplace(lu, r.pivots, x);
  at::native::resize_output(result, b.sizes());
  result.copy_(vector_rhs ? x.squeeze(1) : x);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.solve_ex", /*is_matrix=*/true);
  }
  return {result, lu, pivots, info};
}

std::tuple<at::Tensor&, at::Tensor&> inv_ex_out(const at::Tensor& a, bool check_errors, at::Tensor& inverse,
                                                at::Tensor& info) {
  if (!gpu_matrix(a) || !is_vk(inverse)) {
    on_cpu("aten::linalg_inv_ex", "inverse", a, check_errors, inverse, info);
    return {inverse, info};
  }
  at::Tensor lu = a.clone(at::MemoryFormat::Contiguous);
  LuInfo r = lu_inplace(lu);
  at::Tensor x = at::zeros(a.sizes(), a.options());
  x.diagonal().fill_(1);
  lu_solve_inplace(lu, r.pivots, x);
  at::native::resize_output(inverse, a.sizes());
  inverse.copy_(x);
  write_info(info, r.info);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.inv_ex", /*is_matrix=*/true);
  }
  return {inverse, info};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> det_out(const at::Tensor& a, at::Tensor& result, at::Tensor& lu,
                                                          at::Tensor& pivots) {
  if (!gpu_matrix(a) || !is_vk(result)) {
    on_cpu("aten::_linalg_det", "result", a, result, lu, pivots);
    return {result, lu, pivots};
  }
  at::native::resize_output(lu, a.sizes());
  lu.copy_(a);
  LuInfo r = lu_inplace(lu);
  write_pivots(pivots, r.pivots);
  at::native::resize_output(result, {});
  result.copy_(lu.diagonal().prod().mul_(static_cast<double>(permutation_sign(r.pivots))));
  return {result, lu, pivots};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> slogdet_out(const at::Tensor& a, at::Tensor& sign,
                                                                           at::Tensor& logabsdet, at::Tensor& lu,
                                                                           at::Tensor& pivots) {
  if (!gpu_matrix(a) || !is_vk(sign) || !is_vk(logabsdet)) {
    on_cpu("aten::_linalg_slogdet", "sign", a, sign, logabsdet, lu, pivots);
    return {sign, logabsdet, lu, pivots};
  }
  at::native::resize_output(lu, a.sizes());
  lu.copy_(a);
  LuInfo r = lu_inplace(lu);
  write_pivots(pivots, r.pivots);
  at::Tensor diag = lu.diagonal();
  at::native::resize_output(logabsdet, {});
  at::native::resize_output(sign, {});
  logabsdet.copy_(diag.abs().log().sum());
  sign.copy_(diag.sign().prod().mul_(static_cast<double>(permutation_sign(r.pivots))));
  return {sign, logabsdet, lu, pivots};
}

at::Tensor& lu_solve_out(const at::Tensor& lu, const at::Tensor& pivots, const at::Tensor& b, bool left, bool adjoint,
                         at::Tensor& out) {
  if (!left || adjoint || !gpu_matrix(lu) || !is_vk(b) || b.dim() != 2 || b.size(0) != lu.size(0) ||
      b.scalar_type() != lu.scalar_type() || pivots.dim() != 1 || !is_vk(out)) {
    on_cpu("aten::linalg_lu_solve", "out", lu, pivots, b, left, adjoint, out);
    return out;
  }
  at::Tensor host = pivots.cpu().to(at::kLong);
  std::vector<int64_t> piv(host.data_ptr<int64_t>(), host.data_ptr<int64_t>() + host.numel());
  at::Tensor x = b.clone(at::MemoryFormat::Contiguous);
  lu_solve_inplace(lu, piv, x);
  at::native::resize_output(out, b.sizes());
  out.copy_(x);
  return out;
}

at::Tensor& solve_triangular_out(const at::Tensor& a, const at::Tensor& b, bool upper, bool left, bool unitriangular,
                                 at::Tensor& out) {
  if (!left || !gpu_matrix(a) || !is_vk(b) || b.dim() != 2 || b.size(0) != a.size(0) ||
      b.scalar_type() != a.scalar_type() || !is_vk(out)) {
    on_cpu("aten::linalg_solve_triangular", "out", a, b, upper, left, unitriangular, out);
    return out;
  }
  at::Tensor x = b.clone(at::MemoryFormat::Contiguous);
  triangular_solve_inplace(a, x, upper, unitriangular);
  at::native::resize_output(out, b.sizes());
  out.copy_(x);
  return out;
}

at::Tensor solve_triangular(const at::Tensor& a, const at::Tensor& b, bool upper, bool left, bool unitriangular) {
  at::Tensor out = at::empty({0}, b.options());
  return solve_triangular_out(a, b, upper, left, unitriangular, out);
}

std::tuple<at::Tensor&, at::Tensor&> cholesky_ex_out(const at::Tensor& self, bool upper, bool check_errors,
                                                     at::Tensor& l, at::Tensor& info) {
  if (!gpu_matrix(self) || !is_vk(l)) {
    on_cpu("aten::linalg_cholesky_ex", "L", self, upper, check_errors, l, info);
    return {l, info};
  }
  int64_t n = self.size(0);
  at::Tensor a = self.clone(at::MemoryFormat::Contiguous);
  int64_t status = 0;
  for (int64_t k = 0; k < n && status == 0; k += kBlock) {
    int64_t b = std::min(kBlock, n - k);
    at::Tensor a11 = a.narrow(0, k, b).narrow(1, k, b);
    auto [l11, block_info] = at::linalg_cholesky_ex(a11.cpu());
    int64_t local = block_info.item<int64_t>();
    if (local > 0) {
      status = k + local;  // leading minor of this order is not positive-definite
      break;
    }
    a11.copy_(l11);
    if (k + b < n) {
      // L21 = A21 L11^{-T};  A22 -= L21 L21^T.
      at::Tensor inv = block_inverse(a11, /*upper=*/false, /*unit=*/false);
      at::Tensor a21 = a.narrow(0, k + b, n - k - b).narrow(1, k, b);
      a21.copy_(at::mm(a21, inv.t()));
      a.narrow(0, k + b, n - k - b).narrow(1, k + b, n - k - b).addmm_(a21, a21.t(), 1.0, -1.0);
    }
  }
  at::native::resize_output(l, self.sizes());
  at::Tensor lower = a.tril();
  l.copy_(upper ? lower.t() : lower);
  write_info(info, status);
  if (check_errors) {
    at::_linalg_check_errors(info, "torch.linalg.cholesky_ex", /*is_matrix=*/true);
  }
  return {l, info};
}

// ---------------------------------------------------------------- tril / triu

template <bool Upper>
at::Tensor& triangle_out(const at::Tensor& self, int64_t diagonal, at::Tensor& out) {
  const char* label = dtype_label(self.scalar_type());
  at::native::resize_output(out, self.sizes());
  if (label == nullptr || !is_vk(self) || !is_vk(out) || self.dim() < 2 || out.scalar_type() != self.scalar_type()) {
    on_cpu(Upper ? "aten::triu" : "aten::tril", "out", self, diagonal, out);
    return out;
  }
  if (self.numel() == 0) {
    return out;
  }
  // [batch, rows, cols] views (a copy of the input if it cannot be viewed so;
  // a temporary output if `out` cannot be).
  at::Tensor in3 = self.reshape({-1, self.size(-2), self.size(-1)});
  bool direct = out.is_contiguous();
  at::Tensor target = direct ? out : at::empty(out.sizes(), out.options());
  at::Tensor out3 = target.view({-1, out.size(-2), out.size(-1)});
  TriangleParams p{};
  p.in = gpu_address(in3);
  p.out = gpu_address(out3);
  p.batch = static_cast<uint32_t>(in3.size(0));
  p.rows = static_cast<uint32_t>(in3.size(1));
  p.cols = static_cast<uint32_t>(in3.size(2));
  p.diag = static_cast<int32_t>(diagonal);
  p.in_sb = static_cast<int32_t>(in3.stride(0));
  p.in_sr = static_cast<int32_t>(in3.stride(1));
  p.in_sc = static_cast<int32_t>(in3.stride(2));
  p.out_sb = static_cast<int32_t>(out3.stride(0));
  p.out_sr = static_cast<int32_t>(out3.stride(1));
  p.out_sc = static_cast<int32_t>(out3.stride(2));
  p.ngroups = groups_for(in3.numel());
  dispatch(std::string("triangle_") + label, {Upper ? 1u : 0u}, p, p.ngroups);
  if (!direct) {
    out.copy_(target);
  }
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("linalg_lu_factor_ex.out", &lu_factor_ex_out);
  m.impl("_linalg_solve_ex.result", &solve_ex_out);
  m.impl("linalg_inv_ex.inverse", &inv_ex_out);
  m.impl("_linalg_det.result", &det_out);
  m.impl("_linalg_slogdet.sign", &slogdet_out);
  m.impl("linalg_lu_solve.out", &lu_solve_out);
  m.impl("linalg_solve_triangular.out", &solve_triangular_out);
  m.impl("linalg_solve_triangular", &solve_triangular);
  m.impl("linalg_cholesky_ex.L", &cholesky_ex_out);
  m.impl("tril.out", &triangle_out<false>);
  m.impl("triu.out", &triangle_out<true>);
}
