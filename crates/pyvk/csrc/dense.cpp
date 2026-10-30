// Dense linear algebra for device "vk", entirely on the GPU (M13): LU
// (lu_factor, lu, solve, inv, det, slogdet, lu_solve, lu_unpack), Cholesky
// (cholesky, cholesky_solve, cholesky_inverse), triangular solves, QR
// (geqrf, householder_product, ormqr, qr), symmetric eigendecomposition
// (eigh), SVD, least squares, tril/triu.
//
// Blocked algorithms over batches of matrices ([B, rows, cols], row-major):
// the steps LAPACK would do column by column (LU and QR panels, Cholesky
// diagonal blocks, inverses of small triangular blocks, row interchanges)
// run in shaders/factor.slang with one workgroup per matrix; the O(n^3)
// trailing updates are batched GEMMs (csrc/linalg.cpp). Eigenvalues and the
// SVD use Jacobi methods (rotations of disjoint index pairs in parallel,
// round-robin ordering), which are accurate to working precision and suit
// the GPU. TORCH_VK_LINALG=cpu sends these ops to LAPACK on the CPU instead.
//
// Complex matrices (M14) run through the same code: factor.slang has complex
// variants following LAPACK's z* routines, and transposes that stand for
// adjoints here are conjugate transposes (adj()).

#include "vk.h"

#include "opcodes.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace vk {
namespace {

constexpr int64_t kBlock = 64;
constexpr int64_t kMaxBatch = 65535;  // workgroups per dispatch dimension

struct FactorParams {
  uint64_t a, piv, aux, aux2, info, pairs;
  uint32_t rows, cols, ld, k, b, sa, saux, saux2, npairs, spiv;
};
static_assert(sizeof(FactorParams) == 88, "must match shaders/factor.slang");

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

// TORCH_VK_LINALG=cpu sends these ops to LAPACK on the CPU.
bool linalg_on_cpu() {
  static const bool cpu = [] {
    const char* v = std::getenv("TORCH_VK_LINALG");
    return v != nullptr && std::string(v) == "cpu";
  }();
  return cpu;
}

bool float_dtype(at::ScalarType s) {
  bool f64 = has_feature(kFeatureFloat64);
  return s == at::kFloat || s == at::kComplexFloat || (f64 && (s == at::kDouble || s == at::kComplexDouble));
}

// Conjugate transpose of the last two dims (a plain transpose for real t),
// materialized for complex t: the kernels here read raw memory, which a lazy
// conjugate view would not change.
at::Tensor adj(const at::Tensor& t) {
  return t.is_complex() ? t.transpose(-2, -1).conj().resolve_conj() : t.transpose(-2, -1);
}

// |t|^2 element-wise, real.
at::Tensor abs2(const at::Tensor& t) {
  return t.is_complex() ? at::view_as_real(t.resolve_conj()).square().sum(-1) : t.square();
}

// A batch of matrices this file handles: vk, float32/64 (or complex), at least 2-D, and
// at most kMaxBatch matrices.
bool gpu_matrices(const at::Tensor& t) {
  if (linalg_on_cpu() || !is_vk(t) || t.dim() < 2 || !float_dtype(t.scalar_type())) {
    return false;
  }
  int64_t batch = 1;
  for (int64_t d = 0; d + 2 < t.dim(); ++d) batch *= t.size(d);
  return batch <= kMaxBatch;
}

// [B, rows, cols] view (or copy) of t's matrices; B explicit so empty
// batches work. Lazy conjugate / negative views are materialized: the
// kernels read raw memory (and .contiguous() keeps a contiguous view lazy).
at::Tensor flat3(const at::Tensor& t) {
  int64_t batch = 1;
  for (int64_t d = 0; d + 2 < t.dim(); ++d) batch *= t.size(d);
  return t.resolve_conj().resolve_neg().reshape({batch, t.size(-2), t.size(-1)});
}

// [B, k] of a batch of vectors t [..., k].
at::Tensor flat2(const at::Tensor& t) {
  int64_t batch = 1;
  for (int64_t d = 0; d + 1 < t.dim(); ++d) batch *= t.size(d);
  return t.resolve_conj().resolve_neg().reshape({batch, t.size(-1)});
}

// [B, rows, cols] contiguous copy of t's matrices.
at::Tensor batched(const at::Tensor& t) {
  return flat3(t).clone(at::MemoryFormat::Contiguous);
}

std::vector<int64_t> batch_shape(const at::Tensor& t) {
  return std::vector<int64_t>(t.sizes().begin(), t.sizes().end() - 2);
}

std::vector<int64_t> with_tail(std::vector<int64_t> shape, std::initializer_list<int64_t> tail) {
  shape.insert(shape.end(), tail.begin(), tail.end());
  return shape;
}

// Dispatches factor.slang's `op` with one workgroup per matrix (or per pair
// and matrix when pairs != 0).
void factor(uint32_t op, FactorParams p, int64_t batch, at::ScalarType dtype, bool upper = false, bool unit = false,
            uint32_t pairs = 0) {
  if (batch == 0) {
    return;
  }
  std::string kernel = std::string("factor_") + dtype_label(dtype);
  if (pairs == 0) {
    dispatch(kernel, {op, upper ? 1u : 0u, unit ? 1u : 0u}, p, static_cast<uint32_t>(batch));
  } else {
    dispatch(kernel, {op, upper ? 1u : 0u, unit ? 1u : 0u}, p, pairs, static_cast<uint32_t>(batch));
  }
}

FactorParams params_for(const at::Tensor& a3) {
  FactorParams p{};
  p.a = gpu_address(a3);
  p.rows = static_cast<uint32_t>(a3.size(1));
  p.cols = static_cast<uint32_t>(a3.size(2));
  p.ld = static_cast<uint32_t>(a3.stride(1));
  p.sa = static_cast<uint32_t>(a3.stride(0));
  return p;
}

// An empty tensor of `sizes` whose matrices (last two dims) are column-major.
at::Tensor empty_column_major(at::IntArrayRef sizes, const at::TensorOptions& opts) {
  std::vector<int64_t> strides(sizes.size());
  int64_t step = 1;
  int64_t n = static_cast<int64_t>(sizes.size());
  for (int64_t d : {n - 2, n - 1}) {
    if (d >= 0) {
      strides[d] = step;
      step *= std::max<int64_t>(sizes[d], 1);
    }
  }
  for (int64_t d = n - 3; d >= 0; --d) {
    strides[d] = step;
    step *= std::max<int64_t>(sizes[d], 1);
  }
  return at::empty_strided(sizes, strides, opts);
}

// ---------------------------------------------------------------- triangular solves

// Inverses [B, b, b] of the b x b triangular blocks at (k, k) of a3.
at::Tensor block_inverse(const at::Tensor& a3, int64_t k, int64_t b, bool upper, bool unit) {
  at::Tensor inv = at::empty({a3.size(0), b, b}, a3.options());
  FactorParams p = params_for(a3);
  p.aux = gpu_address(inv);
  p.saux = static_cast<uint32_t>(b * b);
  p.k = static_cast<uint32_t>(k);
  p.b = static_cast<uint32_t>(b);
  factor(op::FACTOR_TRI_INV, p, a3.size(0), a3.scalar_type(), upper, unit);
  return inv;
}

// Y <- T^{-1} Y for square triangular T [B, n, n] (contiguous) and Y [B, n, k].
void triangular_solve_inplace(const at::Tensor& t3, const at::Tensor& y, bool upper, bool unit) {
  int64_t n = t3.size(1);
  int64_t nblocks = (n + kBlock - 1) / kBlock;
  for (int64_t bi = 0; bi < nblocks; ++bi) {
    // Lower: top-down (forward substitution); upper: bottom-up.
    int64_t blk = upper ? nblocks - 1 - bi : bi;
    int64_t k = blk * kBlock, b = std::min(kBlock, n - k);
    at::Tensor inv = block_inverse(t3, k, b, upper, unit);
    at::Tensor yk = y.narrow(1, k, b);
    yk.copy_(at::bmm(inv, yk));
    if (upper && k > 0) {
      y.narrow(1, 0, k).baddbmm_(t3.narrow(1, 0, k).narrow(2, k, b), yk, 1.0, -1.0);
    } else if (!upper && k + b < n) {
      y.narrow(1, k + b, n - k - b).baddbmm_(t3.narrow(1, k + b, n - k - b).narrow(2, k, b), yk, 1.0, -1.0);
    }
  }
}

// Solves op(A) X = B (left) or X op(A) = B (right) for triangular A [B, n, n]
// and B [B, n, k] / [B, k, n]; returns X [B, ...] (new tensor).
at::Tensor tri_solve(const at::Tensor& a3, const at::Tensor& b3, bool upper, bool left, bool unit) {
  if (left) {
    at::Tensor y = b3.clone(at::MemoryFormat::Contiguous);
    triangular_solve_inplace(a3, y, upper, unit);
    return y;
  }
  // X A = B  <=>  A^T X^T = B^T, with A^T triangular the other way.
  at::Tensor at3 = a3.transpose(1, 2).contiguous();
  at::Tensor y = b3.transpose(1, 2).contiguous();
  triangular_solve_inplace(at3, y, !upper, unit);
  return y.transpose(1, 2);
}

// ---------------------------------------------------------------- LU

struct Lu {
  at::Tensor lu;     // [B, m, n]
  at::Tensor piv;    // [B, min(m, n)] int32, 1-based
  at::Tensor info;   // [B] int32
};

Lu lu_factor(const at::Tensor& a) {
  Lu r;
  r.lu = batched(a);
  int64_t B = r.lu.size(0), m = r.lu.size(1), n = r.lu.size(2), kmax = std::min(m, n);
  r.piv = at::empty({B, kmax}, a.options().dtype(at::kInt));
  r.info = at::zeros({B}, a.options().dtype(at::kInt));
  if (B == 0 || kmax == 0) {
    return r;
  }
  FactorParams p = params_for(r.lu);
  p.piv = gpu_address(r.piv);
  p.spiv = static_cast<uint32_t>(kmax);
  p.info = gpu_address(r.info);
  for (int64_t k = 0; k < kmax; k += kBlock) {
    int64_t b = std::min(kBlock, kmax - k);
    p.k = static_cast<uint32_t>(k);
    p.b = static_cast<uint32_t>(b);
    factor(op::FACTOR_LU_PANEL, p, B, a.scalar_type());
    if (k + b < n) {
      // U12 = L11^{-1} A12;  A22 -= L21 U12.
      at::Tensor inv = block_inverse(r.lu, k, b, /*upper=*/false, /*unit=*/true);
      at::Tensor a12 = r.lu.narrow(1, k, b).narrow(2, k + b, n - k - b);
      a12.copy_(at::bmm(inv, a12));
      if (k + b < m) {
        r.lu.narrow(1, k + b, m - k - b).narrow(2, k + b, n - k - b)
            .baddbmm_(r.lu.narrow(1, k + b, m - k - b).narrow(2, k, b), a12, 1.0, -1.0);
      }
    }
  }
  return r;
}

// Applies LAPACK row interchanges piv [B, np] to x3 [B, rows, cols] (in place;
// reverse order undoes them).
void apply_pivots(const at::Tensor& x3, const at::Tensor& piv, bool reverse) {
  if (x3.numel() == 0 || piv.size(1) == 0) {
    return;
  }
  FactorParams p = params_for(x3);
  p.piv = gpu_address(piv);
  p.spiv = static_cast<uint32_t>(piv.stride(0));
  p.b = static_cast<uint32_t>(piv.size(1));
  factor(op::FACTOR_LASWP, p, x3.size(0), x3.scalar_type(), /*upper=*/reverse);
}

// X = A^{-1} B (or A^{-T} B with adjoint) from LU factors; b3 [B, n, k].
at::Tensor lu_solve(const at::Tensor& lu3, const at::Tensor& piv, const at::Tensor& b3, bool adjoint) {
  at::Tensor x = b3.clone(at::MemoryFormat::Contiguous);
  if (!adjoint) {
    apply_pivots(x, piv, false);  // P^T B
    triangular_solve_inplace(lu3, x, /*upper=*/false, /*unit=*/true);
    triangular_solve_inplace(lu3, x, /*upper=*/true, /*unit=*/false);
  } else {
    // A^H = U^H L^H P^T:  X = P L^{-H} U^{-H} B.
    at::Tensor lut = adj(lu3).contiguous();
    triangular_solve_inplace(lut, x, /*upper=*/false, /*unit=*/false);
    triangular_solve_inplace(lut, x, /*upper=*/true, /*unit=*/true);
    apply_pivots(x, piv, true);
  }
  return x;
}

// Sign of the row permutation of each matrix's pivots: [B] in a's dtype.
at::Tensor pivot_sign(const at::Tensor& piv, const at::TensorOptions& opts) {
  at::Tensor ids = at::arange(1, piv.size(1) + 1, piv.options());
  at::Tensor swaps = at::ne(piv, ids).sum(1);
  return (1 - 2 * at::remainder(swaps, 2)).to(opts.dtype());
}

void check_errors(const at::Tensor& info, const char* api, bool is_matrix) {
  at::_linalg_check_errors(info, api, is_matrix);
}

template <typename T>
void assign(at::Tensor& out, const T& value) {
  if (out.scalar_type() != at::kBool) {  // logical results (any, all, ...) accumulate in other dtypes
    check_out_cast(value.scalar_type(), out);
  }
  at::native::resize_output(out, value.sizes());
  out.copy_(value);
}

// ---------------------------------------------------------------- QR (Householder)

// Unit lower-trapezoidal V of the panel at column k (width b), rows k.., of
// reflectors stored below the diagonal of a3.
at::Tensor reflectors(const at::Tensor& a3, int64_t k, int64_t b) {
  int64_t m = a3.size(1);
  at::Tensor v = a3.narrow(1, k, m - k).narrow(2, k, b).tril(-1);
  v.diagonal(0, 1, 2).fill_(1);
  return v;
}

// T [B, b, b] of the panel's block reflector (from the reflectors in a3 and tau).
at::Tensor block_t(const at::Tensor& a3, const at::Tensor& tau, int64_t k, int64_t b) {
  at::Tensor t = at::empty({a3.size(0), b, b}, a3.options());
  FactorParams p = params_for(a3);
  p.aux = gpu_address(tau);
  p.saux = static_cast<uint32_t>(tau.stride(0));
  p.aux2 = gpu_address(t);
  p.saux2 = static_cast<uint32_t>(b * b);
  p.k = static_cast<uint32_t>(k);
  p.b = static_cast<uint32_t>(b);
  factor(op::FACTOR_LARFT, p, a3.size(0), a3.scalar_type());
  return t;
}

// In-place geqrf of a3 [B, m, n]; returns tau [B, min(m, n)].
at::Tensor geqrf_inplace(const at::Tensor& a3) {
  int64_t B = a3.size(0), m = a3.size(1), n = a3.size(2), kmax = std::min(m, n);
  at::Tensor tau = at::zeros({B, kmax}, a3.options());
  if (B == 0 || kmax == 0) {
    return tau;
  }
  FactorParams p = params_for(a3);
  p.aux = gpu_address(tau);
  p.saux = static_cast<uint32_t>(kmax);
  for (int64_t k = 0; k < kmax; k += kBlock) {
    int64_t b = std::min(kBlock, kmax - k);
    p.k = static_cast<uint32_t>(k);
    p.b = static_cast<uint32_t>(b);
    factor(op::FACTOR_QR_PANEL, p, B, a3.scalar_type());
    if (k + b < n) {
      // A2 <- H^H A2 = A2 - V T^H (V^H A2)
      at::Tensor v = reflectors(a3, k, b);
      at::Tensor t = block_t(a3, tau, k, b);
      at::Tensor a2 = a3.narrow(1, k, m - k).narrow(2, k + b, n - k - b);
      at::Tensor w = at::bmm(adj(t), at::bmm(adj(v), a2));
      a2.baddbmm_(v, w, 1.0, -1.0);
    }
  }
  return tau;
}

// C <- op(Q) C (left) or C op(Q) (right) for Q = H_0 ... H_{k-1} from
// reflectors in a3 [B, m, k'] and tau [B, k]; c3 [B, ...] contiguous.
void apply_q(const at::Tensor& a3, const at::Tensor& tau, const at::Tensor& c3, bool left, bool transpose) {
  int64_t kmax = tau.size(1);
  std::vector<int64_t> starts;
  for (int64_t k = 0; k < kmax; k += kBlock) starts.push_back(k);
  // Q C applies the last panel first; Q^H C the first; on the right the reverse.
  bool forward = left == transpose;
  if (!forward) std::reverse(starts.begin(), starts.end());
  int64_t m = a3.size(1);
  for (int64_t k : starts) {
    int64_t b = std::min(kBlock, kmax - k);
    at::Tensor v = reflectors(a3, k, b);
    at::Tensor t = block_t(a3, tau, k, b);
    at::Tensor tt = transpose ? adj(t) : t;
    if (left) {
      at::Tensor c = c3.narrow(1, k, m - k);
      c.baddbmm_(v, at::bmm(tt, at::bmm(adj(v), c)), 1.0, -1.0);
    } else {
      at::Tensor c = c3.narrow(2, k, m - k);
      c.baddbmm_(at::bmm(at::bmm(c, v), tt), adj(v), 1.0, -1.0);
    }
  }
}

// The first `cols` columns of Q (m x m) from reflectors a3 [B, m, ...], tau.
at::Tensor form_q(const at::Tensor& a3, const at::Tensor& tau, int64_t cols) {
  int64_t B = a3.size(0), m = a3.size(1);
  at::Tensor q = at::zeros({B, m, cols}, a3.options());
  q.diagonal(0, 1, 2).fill_(1);
  apply_q(a3, tau, q, /*left=*/true, /*transpose=*/false);
  return q;
}

// ---------------------------------------------------------------- Jacobi methods

// Round-robin pairings of `count` indices (padded to even with a dummy
// `count`): rounds x pairs, as int32 pairs on the device.
at::Tensor jacobi_pairs(int64_t count, const at::TensorOptions& opts, int64_t& rounds, int64_t& per_round) {
  int64_t n = count + (count % 2);
  rounds = std::max<int64_t>(n - 1, 1);
  per_round = n / 2;
  std::vector<int32_t> data;
  std::vector<int64_t> ring(n);
  for (int64_t i = 0; i < n; ++i) ring[i] = i;
  for (int64_t r = 0; r < rounds; ++r) {
    for (int64_t i = 0; i < per_round; ++i) {
      int64_t a = ring[i], b = ring[n - 1 - i];
      data.push_back(static_cast<int32_t>(std::min(a, b)));
      data.push_back(static_cast<int32_t>(std::max(a, b)));
    }
    // Rotate all but the first.
    int64_t last = ring[n - 1];
    for (int64_t i = n - 1; i > 1; --i) ring[i] = ring[i - 1];
    if (n > 1) ring[1] = last;
  }
  return at::tensor(data, at::kInt).to(opts.device()).view({rounds, per_round, 2});
}

double epsilon(at::ScalarType t) {
  return t == at::kDouble ? std::numeric_limits<double>::epsilon() : std::numeric_limits<float>::epsilon();
}

// Symmetric (Hermitian) eigendecomposition of a3 [B, n, n] (overwritten):
// (eigenvalues ascending [B, n], real; eigenvectors [B, n, n]).
std::tuple<at::Tensor, at::Tensor> jacobi_eigh(const at::Tensor& a3, bool want_vectors) {
  int64_t B = a3.size(0), n = a3.size(2);
  at::Tensor v = at::zeros({B, n, n}, a3.options());
  v.diagonal(0, 1, 2).fill_(1);
  if (n > 1 && B > 0) {
    int64_t rounds = 0, per_round = 0;
    at::Tensor pairs = jacobi_pairs(n, a3.options(), rounds, per_round);
    at::Tensor rot = at::empty({B, per_round, 2}, a3.options());
    FactorParams p = params_for(a3);
    p.aux = gpu_address(v);
    p.saux = static_cast<uint32_t>(n * n);
    p.aux2 = gpu_address(rot);
    p.saux2 = static_cast<uint32_t>(per_round * 2);
    p.npairs = static_cast<uint32_t>(per_round);
    double tol = epsilon(a3.scalar_type());
    at::Tensor norm2 = abs2(a3).sum({1, 2});
    for (int sweep = 0; sweep < 40; ++sweep) {
      for (int64_t r = 0; r < rounds; ++r) {
        p.pairs = gpu_address(pairs) + static_cast<uint64_t>(r * per_round * 2 * sizeof(int32_t));
        factor(op::FACTOR_EIG_COLS, p, B, a3.scalar_type(), false, false, static_cast<uint32_t>(per_round));
        factor(op::FACTOR_EIG_ROWS, p, B, a3.scalar_type(), false, false, static_cast<uint32_t>(per_round));
      }
      // (Summed directly: "total minus diagonal" cancels and can stop early.)
      at::Tensor off2 = abs2(a3 - at::diag_embed(a3.diagonal(0, 1, 2))).sum({1, 2});
      // Converged when the off-diagonal part is negligible in every matrix.
      if (at::le(off2, norm2 * (tol * tol)).all().item<bool>()) {
        break;
      }
    }
  }
  at::Tensor diag = at::real(a3.diagonal(0, 1, 2)).contiguous();
  auto [values, order] = at::sort(diag, /*stable=*/true, 1, false);
  at::Tensor vectors = want_vectors ? at::gather(v, 2, order.unsqueeze(1).expand({B, n, n})) : at::Tensor();
  return {values, vectors};
}

// One-sided Jacobi SVD of a3 [B, m, n], m >= n (overwritten):
// (singular values descending [B, n], W = U S unnormalized [B, m, n], V [B, n, n]).
std::tuple<at::Tensor, at::Tensor, at::Tensor> jacobi_svd(const at::Tensor& a3) {
  int64_t B = a3.size(0), m = a3.size(1), n = a3.size(2);
  at::Tensor v = at::zeros({B, n, n}, a3.options());
  v.diagonal(0, 1, 2).fill_(1);
  if (n > 1 && B > 0 && m > 0) {
    int64_t rounds = 0, per_round = 0;
    at::Tensor pairs = jacobi_pairs(n, a3.options(), rounds, per_round);
    at::Tensor off = at::empty({B, per_round}, a3.options());  // real parts used
    FactorParams p = params_for(a3);
    p.aux = gpu_address(v);
    p.saux = static_cast<uint32_t>(n * n);
    p.aux2 = gpu_address(off);
    p.saux2 = static_cast<uint32_t>(per_round);
    p.npairs = static_cast<uint32_t>(per_round);
    double tol = epsilon(a3.scalar_type()) * std::sqrt(static_cast<double>(m));
    for (int sweep = 0; sweep < 40; ++sweep) {
      at::Tensor worst = at::zeros({}, a3.options().dtype(c10::toRealValueType(a3.scalar_type())));
      for (int64_t r = 0; r < rounds; ++r) {
        p.pairs = gpu_address(pairs) + static_cast<uint64_t>(r * per_round * 2 * sizeof(int32_t));
        factor(op::FACTOR_SVD_ROUND, p, B, a3.scalar_type(), false, false, static_cast<uint32_t>(per_round));
        worst = at::maximum(worst, at::real(off).amax());
      }
      if (worst.item<double>() <= tol) {
        break;
      }
    }
  }
  at::Tensor sigma = abs2(a3).sum(1).sqrt();  // [B, n], real
  auto [values, order] = at::sort(sigma, /*stable=*/true, 1, /*descending=*/true);
  at::Tensor w = at::gather(a3, 2, order.unsqueeze(1).expand({B, m, n}));
  at::Tensor vs = at::gather(v, 2, order.unsqueeze(1).expand({B, n, n}));
  return {values, w, vs};
}

// Left singular vectors from W = U S: normalized columns where S is not
// negligible, completed to an orthonormal basis of `cols` columns with QR.
at::Tensor left_vectors(const at::Tensor& w, const at::Tensor& sigma, int64_t cols) {
  int64_t B = w.size(0), m = w.size(1), n = w.size(2);
  double tol = epsilon(w.scalar_type()) * static_cast<double>(std::max(m, n));
  at::Tensor big = at::gt(sigma, sigma.narrow(1, 0, std::min<int64_t>(n, 1)) * tol).unsqueeze(1);  // [B, 1, n]
  at::Tensor safe = at::where(big, sigma.unsqueeze(1), at::ones_like(sigma.unsqueeze(1))).to(w.scalar_type());
  at::Tensor u = at::where(big, w / safe, at::zeros_like(w));
  bool full_rank = big.all().item<bool>();
  if (full_rank && cols == n) {
    return u;
  }
  // Householder QR of [u | 0]: its Q's first columns are +-u, the rest an
  // orthonormal complement.
  at::Tensor basis = at::zeros({B, m, m}, w.options());
  basis.narrow(2, 0, n).copy_(u);
  at::Tensor tau = geqrf_inplace(basis);
  at::Tensor q = form_q(basis, tau, cols);
  at::Tensor keep = at::zeros({B, 1, cols}, big.options());
  keep.narrow(2, 0, n).copy_(big);
  at::Tensor ufull = at::zeros({B, m, cols}, w.options());
  ufull.narrow(2, 0, n).copy_(u);
  return at::where(keep, ufull, q);
}

// ---------------------------------------------------------------- aten kernels: LU family

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> lu_factor_ex_out(const at::Tensor& a, bool pivot, bool check,
                                                                   at::Tensor& lu, at::Tensor& pivots,
                                                                   at::Tensor& info) {
  if (!pivot || !gpu_matrices(a) || !is_vk(lu)) {
    on_cpu("aten::linalg_lu_factor_ex", "out", a, pivot, check, lu, pivots, info);
    return {lu, pivots, info};
  }
  Lu r = lu_factor(a);
  auto bs = batch_shape(a);
  assign(lu, r.lu.view(a.sizes()));
  assign(pivots, r.piv.view(with_tail(bs, {r.piv.size(1)})));
  assign(info, r.info.view(bs));
  if (check) {
    check_errors(info, "torch.linalg.lu_factor_ex", /*is_matrix=*/a.dim() == 2);
  }
  return {lu, pivots, info};
}

// P, L, U from LU factors and pivots (lu_unpack).
std::tuple<at::Tensor, at::Tensor, at::Tensor> unpack_lu(const at::Tensor& lu3, const at::Tensor& piv, bool data,
                                                         bool pivots) {
  int64_t B = lu3.size(0), m = lu3.size(1), n = lu3.size(2), k = std::min(m, n);
  at::Tensor P, L, U;
  if (data) {
    L = lu3.narrow(2, 0, k).tril(-1);
    L.diagonal(0, 1, 2).fill_(1);
    U = lu3.narrow(1, 0, k).triu();
  }
  if (pivots) {
    // A = P L U with P^T = the interchanges applied to I's rows.
    at::Tensor pt = at::zeros({B, m, m}, lu3.options());
    pt.diagonal(0, 1, 2).fill_(1);
    apply_pivots(pt, piv, false);
    P = pt.transpose(1, 2).contiguous();
  }
  return {P, L, U};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> lu_unpack_out(const at::Tensor& lu, const at::Tensor& pivots,
                                                                bool unpack_data, bool unpack_pivots, at::Tensor& P,
                                                                at::Tensor& L, at::Tensor& U) {
  if (!gpu_matrices(lu) || !is_vk(pivots) || !is_vk(P)) {
    on_cpu("aten::lu_unpack", "out", lu, pivots, unpack_data, unpack_pivots, P, L, U);
    return {P, L, U};
  }
  at::Tensor lu3 = batched(lu);
  at::Tensor piv = flat2(pivots).to(at::kInt).contiguous();
  auto [p, l, u] = unpack_lu(lu3, piv, unpack_data, unpack_pivots);
  auto bs = batch_shape(lu);
  int64_t m = lu.size(-2), n = lu.size(-1), k = std::min(m, n);
  if (unpack_pivots) assign(P, p.view(with_tail(bs, {m, m})));
  else at::native::resize_output(P, {0});
  if (unpack_data) {
    assign(L, l.view(with_tail(bs, {m, k})));
    assign(U, u.view(with_tail(bs, {k, n})));
  } else {
    at::native::resize_output(L, {0});
    at::native::resize_output(U, {0});
  }
  return {P, L, U};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> linalg_lu_out(const at::Tensor& a, bool pivot, at::Tensor& P,
                                                                at::Tensor& L, at::Tensor& U) {
  if (!pivot || !gpu_matrices(a) || !is_vk(P)) {
    on_cpu("aten::linalg_lu", "out", a, pivot, P, L, U);
    return {P, L, U};
  }
  Lu r = lu_factor(a);
  auto [p, l, u] = unpack_lu(r.lu, r.piv, true, true);
  auto bs = batch_shape(a);
  int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
  assign(P, p.view(with_tail(bs, {m, m})));
  assign(L, l.view(with_tail(bs, {m, k})));
  assign(U, u.view(with_tail(bs, {k, n})));
  return {P, L, U};
}

bool is_square(const at::Tensor& a) {
  return a.size(-1) == a.size(-2);
}

// B as [B, n, k] matching A's batch (broadcast), and whether it was a vector.
at::Tensor rhs_batched(const at::Tensor& a, const at::Tensor& b, bool& vector) {
  vector = b.dim() == 1 || (b.dim() == a.dim() - 1 && b.sizes().equals(a.sizes().slice(0, a.dim() - 1)));
  at::Tensor bm = vector ? b.unsqueeze(-1) : b;
  auto bs = at::infer_size(batch_shape(a), std::vector<int64_t>(bm.sizes().begin(), bm.sizes().end() - 2));
  return flat3(bm.expand(with_tail(bs, {bm.size(-2), bm.size(-1)}))).contiguous();
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> solve_ex_out(
    const at::Tensor& a, const at::Tensor& b, bool left, bool check, at::Tensor& result, at::Tensor& lu,
    at::Tensor& pivots, at::Tensor& info) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(b) || b.scalar_type() != a.scalar_type() || !is_vk(result)) {
    on_cpu("aten::_linalg_solve_ex", "result", a, b, left, check, result, lu, pivots, info);
    return {result, lu, pivots, info};
  }
  // X A = B  <=>  A^T X^T = B^T.
  at::Tensor aa = left ? a : a.transpose(-2, -1);
  at::Tensor bb = left ? b : b.transpose(-2, -1);
  bool vector = false;
  at::Tensor b3 = rhs_batched(aa, bb, vector);
  Lu r = lu_factor(aa);
  int64_t B = b3.size(0);
  at::Tensor lu3 = r.lu.size(0) == B ? r.lu : r.lu.expand({B, r.lu.size(1), r.lu.size(2)}).contiguous();
  at::Tensor piv = r.piv.size(0) == B ? r.piv : r.piv.expand({B, r.piv.size(1)}).contiguous();
  at::Tensor x = lu_solve(lu3, piv, b3, false);
  auto bs = at::infer_size(batch_shape(aa), vector ? std::vector<int64_t>(bb.sizes().begin(), bb.sizes().end() - 1)
                                                   : batch_shape(bb));
  at::Tensor xr = vector ? x.view(with_tail(bs, {x.size(1)})) : x.view(with_tail(bs, {x.size(1), x.size(2)}));
  if (!left) xr = xr.transpose(-2, -1);
  assign(result, xr);
  assign(lu, r.lu.view(aa.sizes()));
  assign(pivots, r.piv.view(with_tail(batch_shape(aa), {r.piv.size(1)})));
  assign(info, r.info.view(batch_shape(aa)));
  if (check) {
    check_errors(info, "torch.linalg.solve_ex", a.dim() == 2);
  }
  return {result, lu, pivots, info};
}

std::tuple<at::Tensor&, at::Tensor&> inv_ex_out(const at::Tensor& a, bool check, at::Tensor& inverse,
                                                at::Tensor& info) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(inverse)) {
    on_cpu("aten::linalg_inv_ex", "inverse", a, check, inverse, info);
    return {inverse, info};
  }
  Lu r = lu_factor(a);
  int64_t B = r.lu.size(0), n = r.lu.size(1);
  at::Tensor eye = at::zeros({B, n, n}, a.options());
  eye.diagonal(0, 1, 2).fill_(1);
  at::Tensor x = lu_solve(r.lu, r.piv, eye, false);
  assign(inverse, x.view(a.sizes()));
  assign(info, r.info.view(batch_shape(a)));
  if (check) {
    check_errors(info, "torch.linalg.inv_ex", a.dim() == 2);
  }
  return {inverse, info};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> det_out(const at::Tensor& a, at::Tensor& result, at::Tensor& lu,
                                                          at::Tensor& pivots) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(result)) {
    on_cpu("aten::_linalg_det", "result", a, result, lu, pivots);
    return {result, lu, pivots};
  }
  Lu r = lu_factor(a);
  auto bs = batch_shape(a);
  at::Tensor det = r.lu.diagonal(0, 1, 2).prod(1) * pivot_sign(r.piv, a.options());
  assign(result, det.view(bs));
  assign(lu, r.lu.view(a.sizes()));
  assign(pivots, r.piv.view(with_tail(bs, {r.piv.size(1)})));
  return {result, lu, pivots};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> slogdet_out(const at::Tensor& a, at::Tensor& sign,
                                                                           at::Tensor& logabsdet, at::Tensor& lu,
                                                                           at::Tensor& pivots) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(sign) || !is_vk(logabsdet)) {
    on_cpu("aten::_linalg_slogdet", "sign", a, sign, logabsdet, lu, pivots);
    return {sign, logabsdet, lu, pivots};
  }
  Lu r = lu_factor(a);
  auto bs = batch_shape(a);
  at::Tensor diag = r.lu.diagonal(0, 1, 2);
  assign(logabsdet, diag.abs().log().sum(1).view(bs));
  assign(sign, (diag.sgn().prod(1) * pivot_sign(r.piv, a.options())).view(bs));
  assign(lu, r.lu.view(a.sizes()));
  assign(pivots, r.piv.view(with_tail(bs, {r.piv.size(1)})));
  return {sign, logabsdet, lu, pivots};
}

at::Tensor& lu_solve_out(const at::Tensor& lu, const at::Tensor& pivots, const at::Tensor& b, bool left, bool adjoint,
                         at::Tensor& out) {
  if (!gpu_matrices(lu) || !is_square(lu) || !is_vk(b) || !is_vk(pivots) || b.scalar_type() != lu.scalar_type() ||
      !is_vk(out)) {
    on_cpu("aten::linalg_lu_solve", "out", lu, pivots, b, left, adjoint, out);
    return out;
  }
  // X op(A) = B  <=>  op(A)^H X^H = B^H: solve with the other adjoint flag,
  // on conjugated transposes.
  at::Tensor bb = left ? b : adj(b);
  bool tr = left ? adjoint : !adjoint;
  auto bs = at::infer_size(batch_shape(lu), batch_shape(bb));
  int64_t n = lu.size(-1);
  at::Tensor lu3 = flat3(lu.expand(with_tail(bs, {n, n}))).contiguous();
  at::Tensor piv = flat2(pivots.expand(with_tail(bs, {pivots.size(-1)}))).to(at::kInt)
                       .contiguous();
  at::Tensor b3 = flat3(bb.expand(with_tail(bs, {bb.size(-2), bb.size(-1)})))
                      .contiguous();
  at::Tensor x = lu_solve(lu3, piv, b3, tr).view(with_tail(bs, {bb.size(-2), bb.size(-1)}));
  assign(out, left ? x : adj(x));
  return out;
}

at::Tensor& solve_triangular_out(const at::Tensor& a, const at::Tensor& b, bool upper, bool left, bool unit,
                                 at::Tensor& out) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(b) || b.scalar_type() != a.scalar_type() || b.dim() < 2 ||
      !is_vk(out)) {
    on_cpu("aten::linalg_solve_triangular", "out", a, b, upper, left, unit, out);
    return out;
  }
  auto bs = at::infer_size(batch_shape(a), batch_shape(b));
  int64_t n = a.size(-1);
  at::Tensor a3 = flat3(a.expand(with_tail(bs, {n, n}))).contiguous();
  at::Tensor b3 = flat3(b.expand(with_tail(bs, {b.size(-2), b.size(-1)})))
                      .contiguous();
  at::Tensor x = tri_solve(a3, b3, upper, left, unit);
  assign(out, x.reshape(with_tail(bs, {b.size(-2), b.size(-1)})));
  return out;
}

at::Tensor solve_triangular_vk(const at::Tensor& a, const at::Tensor& b, bool upper, bool left, bool unit) {
  // Laid out as the meta kernel says (column-major matrices, as LAPACK's).
  at::Tensor out = empty_like_meta(at::linalg_solve_triangular(meta_of(a), meta_of(b), upper, left, unit),
                                   b.options());
  return solve_triangular_out(a, b, upper, left, unit, out);
}

// Legacy torch.triangular_solve: (solution, copy of A).
std::tuple<at::Tensor&, at::Tensor&> triangular_solve_legacy_out(const at::Tensor& b, const at::Tensor& a, bool upper,
                                                                  bool transpose, bool unit, at::Tensor& x,
                                                                  at::Tensor& m) {
  if (!gpu_matrices(a) || !is_vk(b)) {
    on_cpu("aten::triangular_solve", "X", b, a, upper, transpose, unit, x, m);
    return {x, m};
  }
  at::Tensor aa = transpose ? a.transpose(-2, -1) : a;
  at::Tensor sol = solve_triangular_vk(aa, b, transpose ? !upper : upper, true, unit);
  assign(x, sol);
  assign(m, a.clone());
  return {x, m};
}

// ---------------------------------------------------------------- Cholesky family

// Lower Cholesky factor of a3 [B, n, n] in place; returns info [B].
at::Tensor cholesky_inplace(const at::Tensor& a3) {
  int64_t B = a3.size(0), n = a3.size(1);
  at::Tensor info = at::zeros({B}, a3.options().dtype(at::kInt));
  FactorParams p = params_for(a3);
  p.info = gpu_address(info);
  for (int64_t k = 0; k < n; k += kBlock) {
    int64_t b = std::min(kBlock, n - k);
    p.k = static_cast<uint32_t>(k);
    p.b = static_cast<uint32_t>(b);
    factor(op::FACTOR_CHOL_BLOCK, p, B, a3.scalar_type());
    if (k + b < n) {
      // L21 = A21 L11^{-H};  A22 -= L21 L21^H.
      at::Tensor inv = block_inverse(a3, k, b, /*upper=*/false, /*unit=*/false);
      at::Tensor a21 = a3.narrow(1, k + b, n - k - b).narrow(2, k, b);
      a21.copy_(at::bmm(a21, adj(inv)));
      a3.narrow(1, k + b, n - k - b).narrow(2, k + b, n - k - b).baddbmm_(a21, adj(a21), 1.0, -1.0);
    }
  }
  return info;
}

std::tuple<at::Tensor&, at::Tensor&> cholesky_ex_out(const at::Tensor& self, bool upper, bool check, at::Tensor& l,
                                                     at::Tensor& info) {
  if (!gpu_matrices(self) || !is_square(self) || !is_vk(l)) {
    on_cpu("aten::linalg_cholesky_ex", "L", self, upper, check, l, info);
    return {l, info};
  }
  // The lower factor of A (upper=True: from A^H's lower triangle, as LAPACK reads it).
  at::Tensor a3 = batched(upper ? adj(self) : self);
  at::Tensor status = cholesky_inplace(a3);
  at::Tensor lower = a3.tril();
  assign(l, (upper ? adj(lower) : lower).reshape(self.sizes()));
  assign(info, status.view(batch_shape(self)));
  if (check) {
    check_errors(info, "torch.linalg.cholesky_ex", self.dim() == 2);
  }
  return {l, info};
}

// cholesky_solve: A X = B with A = L L^H (or U^H U), self = B.
at::Tensor cholesky_solve_helper(const at::Tensor& self, const at::Tensor& a, bool upper) {
  if (!gpu_matrices(a) || !is_vk(self)) {
    return on_cpu("aten::_cholesky_solve_helper", "", self, a, upper)[0].toTensor();
  }
  at::Tensor l = upper ? adj(a) : a;  // lower factor
  at::Tensor y = solve_triangular_vk(l, self, /*upper=*/false, true, false);
  return solve_triangular_vk(adj(l), y, /*upper=*/true, true, false);
}

at::Tensor& cholesky_inverse_out(const at::Tensor& a, bool upper, at::Tensor& out) {
  if (!gpu_matrices(a) || !is_vk(out)) {
    on_cpu("aten::cholesky_inverse", "out", a, upper, out);
    return out;
  }
  int64_t n = a.size(-1);
  at::Tensor eye = at::zeros(a.sizes(), a.options());
  eye.diagonal(0, -2, -1).fill_(1);
  assign(out, cholesky_solve_helper(eye, a, upper).view(a.sizes()));
  (void)n;
  return out;
}

at::Tensor cholesky_inverse_vk(const at::Tensor& a, bool upper) {
  at::Tensor out = at::empty({0}, a.options());
  return cholesky_inverse_out(a, upper, out);
}

// ---------------------------------------------------------------- QR family

std::tuple<at::Tensor&, at::Tensor&> geqrf_out(const at::Tensor& a, at::Tensor& qr, at::Tensor& tau) {
  if (!gpu_matrices(a) || !is_vk(qr)) {
    on_cpu("aten::geqrf", "a", a, qr, tau);
    return {qr, tau};
  }
  at::Tensor a3 = batched(a);
  at::Tensor t = geqrf_inplace(a3);
  assign(qr, a3.view(a.sizes()));
  assign(tau, t.view(with_tail(batch_shape(a), {t.size(1)})));
  return {qr, tau};
}

std::tuple<at::Tensor, at::Tensor> geqrf_vk(const at::Tensor& a) {
  at::Tensor qr = at::empty({0}, a.options()), tau = at::empty({0}, a.options());
  geqrf_out(a, qr, tau);
  return {qr, tau};
}

at::Tensor& householder_product_out(const at::Tensor& a, const at::Tensor& tau, at::Tensor& out) {
  if (!gpu_matrices(a) || !is_vk(tau) || !is_vk(out)) {
    on_cpu("aten::linalg_householder_product", "out", a, tau, out);
    return out;
  }
  at::Tensor a3 = batched(a);
  at::Tensor t = flat2(tau).contiguous();
  at::Tensor q = form_q(a3, t, a.size(-1));
  assign(out, q.view(a.sizes()));
  return out;
}

at::Tensor householder_product_vk(const at::Tensor& a, const at::Tensor& tau) {
  at::Tensor out = at::empty({0}, a.options());
  return householder_product_out(a, tau, out);
}

at::Tensor& ormqr_out(const at::Tensor& a, const at::Tensor& tau, const at::Tensor& c, bool left, bool transpose,
                      at::Tensor& out) {
  if (!gpu_matrices(a) || !is_vk(tau) || !is_vk(c) || !is_vk(out) || c.scalar_type() != a.scalar_type()) {
    on_cpu("aten::ormqr", "out", a, tau, c, left, transpose, out);
    return out;
  }
  // Reflectors as an m x m-wide block (Q is m x m for left, n x n for right).
  at::Tensor a3 = batched(a);
  at::Tensor t = flat2(tau).contiguous();
  at::Tensor c3 = batched(c);
  apply_q(a3, t, c3, left, transpose);
  assign(out, c3.view(c.sizes()));
  return out;
}

at::Tensor ormqr_vk(const at::Tensor& a, const at::Tensor& tau, const at::Tensor& c, bool left, bool transpose) {
  at::Tensor out = at::empty({0}, c.options());
  return ormqr_out(a, tau, c, left, transpose, out);
}

std::tuple<at::Tensor&, at::Tensor&> linalg_qr_out(const at::Tensor& a, c10::string_view mode, at::Tensor& Q,
                                                   at::Tensor& R) {
  if (!gpu_matrices(a) || !is_vk(Q)) {
    on_cpu("aten::linalg_qr", "out", a, mode, Q, R);
    return {Q, R};
  }
  bool complete = mode == "complete", want_q = mode != "r";
  int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
  int64_t qcols = complete ? m : k, rrows = complete ? m : k;
  at::Tensor a3 = batched(a);
  at::Tensor tau = geqrf_inplace(a3);
  auto bs = batch_shape(a);
  assign(R, a3.narrow(1, 0, rrows).triu().reshape(with_tail(bs, {rrows, n})));
  if (want_q) {
    assign(Q, form_q(a3, tau, qcols).reshape(with_tail(bs, {m, qcols})));
  } else {
    at::native::resize_output(Q, {0});
  }
  return {Q, R};
}

// ---------------------------------------------------------------- eigh, SVD, least squares

std::tuple<at::Tensor&, at::Tensor&> eigh_out(const at::Tensor& a, c10::string_view uplo, bool compute_v,
                                              at::Tensor& values, at::Tensor& vectors) {
  if (!gpu_matrices(a) || !is_square(a) || !is_vk(values)) {
    on_cpu("aten::_linalg_eigh", "eigenvalues", a, uplo, compute_v, values, vectors);
    return {values, vectors};
  }
  // Symmetric (Hermitian) matrix from the chosen triangle and the real part
  // of the diagonal.
  at::Tensor strict = uplo == "U" ? a.triu(1) : a.tril(-1);
  at::Tensor diag = at::real(a.diagonal(0, -2, -1)).to(a.scalar_type());
  at::Tensor sym = strict + adj(strict) + at::diag_embed(diag);
  at::Tensor a3 = batched(sym);
  auto [w, v] = jacobi_eigh(a3, compute_v);
  auto bs = batch_shape(a);
  int64_t n = a.size(-1);
  assign(values, w.view(with_tail(bs, {n})));
  if (compute_v) {
    assign(vectors, v.view(a.sizes()));
  } else {
    at::native::resize_output(vectors, {0});
  }
  return {values, vectors};
}

std::tuple<at::Tensor, at::Tensor> eigh_vk(const at::Tensor& a, c10::string_view uplo, bool compute_v) {
  auto [mw, mv] = at::_linalg_eigh(meta_of(a), uplo, compute_v);
  at::Tensor w = empty_like_meta(mw, a.options());
  at::Tensor v = empty_like_meta(mv, a.options());
  eigh_out(a, uplo, compute_v, w, v);
  return {w, v};
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> svd_out(const at::Tensor& a, bool full_matrices, bool compute_uv,
                                                          std::optional<c10::string_view> driver, at::Tensor& U,
                                                          at::Tensor& S, at::Tensor& Vh) {
  if (!gpu_matrices(a) || !is_vk(S)) {
    on_cpu("aten::_linalg_svd", "U", a, full_matrices, compute_uv, driver, U, S, Vh);
    return {U, S, Vh};
  }
  int64_t m = a.size(-2), n = a.size(-1), k = std::min(m, n);
  bool wide = m < n;  // work with A^H (tall): A = U S V^H  <=>  A^H = V S U^H
  at::Tensor a3 = batched(wide ? adj(a) : a);
  int64_t tm = a3.size(1), tn = a3.size(2);  // tm >= tn = k
  auto [sigma, w, v] = jacobi_svd(a3);
  auto bs = batch_shape(a);
  assign(S, sigma.view(with_tail(bs, {k})));
  if (!compute_uv) {
    at::native::resize_output(U, {0});
    at::native::resize_output(Vh, {0});
    return {U, S, Vh};
  }
  at::Tensor left = left_vectors(w, sigma, full_matrices ? tm : tn);  // [B, tm, tm or k]
  // For the tall problem: U_t = left, V_t = v (tn x tn).
  at::Tensor u_out = wide ? v : left;
  at::Tensor v_out = wide ? left : v;
  int64_t ucols = full_matrices ? m : k, vrows = full_matrices ? n : k;
  assign(U, u_out.reshape(with_tail(bs, {m, ucols})));
  assign(Vh, adj(v_out).reshape(with_tail(bs, {vrows, n})));
  return {U, S, Vh};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_vk(const at::Tensor& a, bool full_matrices, bool compute_uv,
                                                   std::optional<c10::string_view> driver) {
  // (PyTorch's meta kernel treats meta tensors as CUDA ones, whose Vh is
  // row-major; for other devices, as on the CPU, Vh is column-major.)
  auto [mU, mS, mVh] = at::_linalg_svd(meta_of(a), full_matrices, compute_uv, driver);
  at::Tensor U = empty_like_meta(mU, a.options());
  at::Tensor Vh = compute_uv ? empty_column_major(mVh.sizes(), a.options()) : empty_like_meta(mVh, a.options());
  at::Tensor S = empty_like_meta(mS, a.options());
  svd_out(a, full_matrices, compute_uv, driver, U, S, Vh);
  return {U, S, Vh};
}

// Least squares through the SVD: the minimum-norm solution, which is what
// LAPACK's default gelsy driver also returns (also for rank-deficient A).
std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> lstsq_out(
    const at::Tensor& a, const at::Tensor& b, std::optional<double> rcond, std::optional<c10::string_view> driver,
    at::Tensor& solution, at::Tensor& residuals, at::Tensor& rank, at::Tensor& singular_values) {
  if (!gpu_matrices(a) || !is_vk(b) || b.scalar_type() != a.scalar_type() || !is_vk(solution)) {
    on_cpu("aten::linalg_lstsq", "out", a, b, rcond, driver, solution, residuals, rank, singular_values);
    return {solution, residuals, rank, singular_values};
  }
  std::string drv = driver.has_value() ? std::string(*driver) : "gelsy";
  int64_t m = a.size(-2), n = a.size(-1);
  bool vector = b.dim() == a.dim() - 1;
  at::Tensor bm = vector ? b.unsqueeze(-1) : b;
  auto bs = at::infer_size(batch_shape(a), batch_shape(bm));
  at::Tensor ae = a.expand(with_tail(bs, {m, n}));
  at::Tensor be = bm.expand(with_tail(bs, {m, bm.size(-1)}));
  auto [U, S, Vh] = at::linalg_svd(ae, /*full_matrices=*/false);
  double rc = rcond.has_value() && *rcond > 0 ? *rcond : epsilon(a.scalar_type()) * std::max(m, n);
  at::Tensor smax = S.narrow(-1, 0, std::min<int64_t>(S.size(-1), 1));
  at::Tensor keep = at::gt(S, smax * rc);
  at::Tensor inv = at::where(keep, 1.0 / at::where(keep, S, at::ones_like(S)), at::zeros_like(S));
  at::Tensor x = at::matmul(adj(Vh), inv.unsqueeze(-1).to(a.scalar_type()) * at::matmul(adj(U), be));
  at::Tensor r = keep.sum(-1);
  assign(solution, vector ? x.squeeze(-1) : x);
  assign(rank, drv == "gels" ? at::empty({0}, r.options()) : r);
  if (drv == "gelsd" || drv == "gelss") {
    assign(singular_values, S);
  } else {
    at::native::resize_output(singular_values, {0});
  }
  // Residuals for tall, full-rank systems (as LAPACK reports them).
  bool full = m > n && at::eq(r, n).all().item<bool>();
  if (full && drv != "gelsy") {
    at::Tensor res = abs2(at::matmul(ae, x) - be).sum(-2);
    assign(residuals, vector ? res.squeeze(-1) : res);
  } else {
    at::native::resize_output(residuals, {0});
  }
  return {solution, residuals, rank, singular_values};
}

// ---------------------------------------------------------------- LDL (Bunch-Kaufman)

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> ldl_factor_ex_out(const at::Tensor& self, bool hermitian,
                                                                    bool check, at::Tensor& LD, at::Tensor& pivots,
                                                                    at::Tensor& info) {
  if (!gpu_matrices(self) || !is_square(self) || !is_vk(LD)) {
    on_cpu("aten::linalg_ldl_factor_ex", "out", self, hermitian, check, LD, pivots, info);
    return {LD, pivots, info};
  }
  at::Tensor a3 = batched(self);
  int64_t B = a3.size(0), n = a3.size(1);
  at::Tensor piv = at::zeros({B, n}, a3.options().dtype(at::kInt));
  at::Tensor status = at::zeros({B}, a3.options().dtype(at::kInt));
  at::Tensor scratch = at::empty({B, 2 * std::max<int64_t>(n, 1)}, a3.options());
  FactorParams p = params_for(a3);
  p.piv = gpu_address(piv);
  p.spiv = static_cast<uint32_t>(n);
  p.info = gpu_address(status);
  p.aux = gpu_address(scratch);
  p.saux = static_cast<uint32_t>(scratch.size(1));
  if (n > 0) {
    factor(op::FACTOR_LDL, p, B, a3.scalar_type(), /*upper=hermitian*/ hermitian && a3.is_complex());
  }
  auto bs = batch_shape(self);
  assign(LD, a3.tril().view(self.sizes()));
  assign(pivots, piv.view(with_tail(bs, {n})));
  assign(info, status.view(bs));
  if (check) {
    check_errors(info, "torch.linalg.ldl_factor_ex", self.dim() == 2);
  }
  return {LD, pivots, info};
}

at::Tensor& ldl_solve_out(const at::Tensor& LD, const at::Tensor& pivots, const at::Tensor& b, bool hermitian,
                          at::Tensor& out) {
  if (!gpu_matrices(LD) || !is_square(LD) || !is_vk(b) || !is_vk(pivots) || b.scalar_type() != LD.scalar_type() ||
      b.dim() < 2 || !is_vk(out)) {
    on_cpu("aten::linalg_ldl_solve", "out", LD, pivots, b, hermitian, out);
    return out;
  }
  auto bs = at::infer_size(batch_shape(LD), batch_shape(b));
  int64_t n = LD.size(-1), k = b.size(-1);
  at::Tensor a3 = flat3(LD.expand(with_tail(bs, {n, n}))).contiguous();
  at::Tensor piv = flat2(pivots.expand(with_tail(bs, {n}))).to(at::kInt).contiguous();
  at::Tensor x = flat3(b.expand(with_tail(bs, {n, k}))).clone(at::MemoryFormat::Contiguous);
  FactorParams p = params_for(a3);
  p.cols = static_cast<uint32_t>(k);
  p.piv = gpu_address(piv);
  p.spiv = static_cast<uint32_t>(n);
  p.aux = gpu_address(x);
  p.saux = static_cast<uint32_t>(n * k);
  if (n > 0 && k > 0) {
    factor(op::FACTOR_LDL_SOLVE, p, a3.size(0), a3.scalar_type(), hermitian && a3.is_complex());
  }
  assign(out, x.view(with_tail(bs, {n, k})));
  return out;
}

// ---------------------------------------------------------------- general eigendecomposition

struct EigParams {
  uint64_t a, z, aux, aux2, state;
  uint32_t n, budget;
};
static_assert(sizeof(EigParams) == 48, "must match shaders/eig.slang");

at::ScalarType complex_of(at::ScalarType t) {
  return t == at::kFloat || t == at::kComplexFloat ? at::kComplexFloat : at::kComplexDouble;
}

// Eigenvalues [B, n] and (want_vectors) unit eigenvectors [B, n, n] of a's
// matrices, complex: Hessenberg reduction, shifted QR to the complex Schur
// form T = Z^H A Z, eigenvectors of T by back substitution, times Z. Each
// vector has unit norm and its largest component real, as LAPACK's geev.
std::tuple<at::Tensor, at::Tensor> general_eig(const at::Tensor& a, bool want_vectors) {
  at::ScalarType ct = complex_of(a.scalar_type());
  at::Tensor h = flat3(a).to(ct).clone(at::MemoryFormat::Contiguous);
  int64_t B = h.size(0), n = h.size(1);
  at::Tensor z = at::empty({B, n, n}, h.options());
  at::Tensor aux = at::empty({B, std::max<int64_t>(n * n, 1)}, h.options());
  at::Tensor state = at::zeros({B, 2}, h.options().dtype(at::kInt));
  state.select(1, 0).fill_(n - 1);
  std::string kernel = std::string("eig_") + dtype_label(ct);
  EigParams p{};
  p.a = gpu_address(h);
  p.z = gpu_address(z);
  p.aux = gpu_address(aux);
  p.state = gpu_address(state);
  p.n = static_cast<uint32_t>(n);
  if (B > 0 && n > 0) {
    dispatch(kernel, {op::EIG_HESSENBERG}, p, static_cast<uint32_t>(B));
    // A few sweeps per dispatch keeps each dispatch short; QR needs ~2-3
    // sweeps per eigenvalue, 30 per eigenvalue at most.
    p.budget = 8;
    for (int64_t round = 0; round < (30 * n) / 8 + 2; ++round) {
      dispatch(kernel, {op::EIG_QR}, p, static_cast<uint32_t>(B));
      if (at::lt(state.select(1, 0), 0).all().item<bool>()) {
        break;
      }
    }
  }
  at::Tensor values = h.diagonal(0, 1, 2).clone();
  if (!want_vectors) {
    return {values, at::Tensor()};
  }
  if (B == 0 || n == 0) {
    return {values, z};
  }
  // Pivots below smin = eps |T| are replaced by smin (ztrevc).
  at::Tensor tabs = at::view_as_real(h).abs().sum(-1).amax({1, 2});
  at::Tensor smin = at::clamp_min(tabs * epsilon(c10::toRealValueType(ct)),
                                  std::numeric_limits<float>::min()).to(ct);
  p.aux2 = gpu_address(smin);
  dispatch(kernel, {op::EIG_VECTORS}, p, static_cast<uint32_t>(B));
  at::Tensor v = at::bmm(z, aux.narrow(1, 0, n * n).view({B, n, n}));
  at::Tensor norms = abs2(v).sum(1, true).sqrt();
  v = v / norms.to(ct);
  at::Tensor big = at::gather(v, 1, abs2(v).argmax(1, true));  // [B, 1, n]
  at::Tensor phase = at::conj_physical(big) / at::abs(big).to(ct);
  return {values, v * phase};
}

bool eig_ok(const at::Tensor& a) {
  return gpu_matrices(a) && is_square(a);
}

std::tuple<at::Tensor&, at::Tensor&> eig_out(const at::Tensor& a, at::Tensor& values, at::Tensor& vectors) {
  if (!eig_ok(a) || !is_vk(values) || !is_vk(vectors)) {
    on_cpu("aten::linalg_eig", "out", a, values, vectors);
    return {values, vectors};
  }
  auto [w, v] = general_eig(a, true);
  auto bs = batch_shape(a);
  int64_t n = a.size(-1);
  assign(values, w.view(with_tail(bs, {n})));
  assign(vectors, v.view(with_tail(bs, {n, n})));
  return {values, vectors};
}

std::tuple<at::Tensor, at::Tensor> eig_vk(const at::Tensor& a) {
  at::ScalarType ct = complex_of(a.scalar_type());
  // Column-major eigenvectors, as LAPACK's geev returns them on the CPU.
  at::Tensor w = at::empty({0}, a.options().dtype(ct));
  at::Tensor v = empty_column_major(a.sizes(), a.options().dtype(ct));
  eig_out(a, w, v);
  return {w, v};
}

at::Tensor& eigvals_out(const at::Tensor& a, at::Tensor& out) {
  if (!eig_ok(a) || !is_vk(out)) {
    on_cpu("aten::linalg_eigvals", "out", a, out);
    return out;
  }
  auto [w, v] = general_eig(a, false);
  assign(out, w.view(with_tail(batch_shape(a), {a.size(-1)})));
  return out;
}

at::Tensor eigvals_vk(const at::Tensor& a) {
  if (!eig_ok(a)) {
    return on_cpu("aten::_linalg_eigvals", "", a)[0].toTensor();
  }
  at::Tensor out = at::empty({0}, a.options().dtype(complex_of(a.scalar_type())));
  return eigvals_out(a, out);
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
  at::Tensor in3 = flat3(self);
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
  m.impl("linalg_lu.out", &linalg_lu_out);
  m.impl("lu_unpack.out", &lu_unpack_out);
  m.impl("_linalg_solve_ex.result", &solve_ex_out);
  m.impl("linalg_inv_ex.inverse", &inv_ex_out);
  m.impl("_linalg_det.result", &det_out);
  m.impl("_linalg_slogdet.sign", &slogdet_out);
  m.impl("linalg_lu_solve.out", &lu_solve_out);
  m.impl("linalg_solve_triangular.out", &solve_triangular_out);
  m.impl("linalg_solve_triangular", &solve_triangular_vk);
  m.impl("triangular_solve.X", &triangular_solve_legacy_out);
  m.impl("linalg_cholesky_ex.L", &cholesky_ex_out);
  m.impl("_cholesky_solve_helper", &cholesky_solve_helper);
  m.impl("cholesky_inverse", &vk::cholesky_inverse_vk);
  m.impl("cholesky_inverse.out", &cholesky_inverse_out);
  m.impl("geqrf", &vk::geqrf_vk);
  m.impl("geqrf.a", &geqrf_out);
  m.impl("linalg_householder_product", &householder_product_vk);
  m.impl("linalg_householder_product.out", &householder_product_out);
  m.impl("ormqr", &vk::ormqr_vk);
  m.impl("ormqr.out", &ormqr_out);
  m.impl("linalg_qr.out", &linalg_qr_out);
  m.impl("_linalg_eigh", &vk::eigh_vk);
  m.impl("_linalg_eigh.eigenvalues", &eigh_out);
  m.impl("_linalg_svd", &vk::svd_vk);
  m.impl("_linalg_svd.U", &svd_out);
  m.impl("linalg_lstsq.out", &lstsq_out);
  m.impl("linalg_ldl_factor_ex.out", &ldl_factor_ex_out);
  m.impl("linalg_ldl_solve.out", &ldl_solve_out);
  m.impl("linalg_eig", &vk::eig_vk);
  m.impl("linalg_eig.out", &eig_out);
  m.impl("_linalg_eigvals", &vk::eigvals_vk);
  m.impl("linalg_eigvals.out", &eigvals_out);
  m.impl("tril.out", &triangle_out<false>);
  m.impl("triu.out", &triangle_out<true>);
}
