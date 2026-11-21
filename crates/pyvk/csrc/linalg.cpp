// Matrix products for device "vk": mm, bmm, addmm, baddbmm, addmv, dot.
//
// All of them run a GEMM kernel on 3-D views [batch, rows, cols]: operands
// are passed by strides, so transposed (e.g. from autograd) and broadcast
// operands need no copies. float32 and float64 run shaders/gemmsg.slang
// (register tiles and subgroup broadcasts) where the device can, gemm4.slang
// on Arm GPUs, gemm.slang (shared-memory tiles, also the integer products)
// otherwise; dtypes and shapes no kernel can address run on the CPU.
// Products with one output column or one output row are matrix-vector
// products and run shaders/gemv.slang.

#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace vk {
namespace {

struct GemmParams {
  uint64_t a, b, c, cin;
  uint64_t alpha, beta;  // pack_param slots
  uint32_t m, n, k, kc;  // kc: k per split (gemmcm.slang and gemmsg.slang with SPLIT)
  int32_t a_sm, a_sk, a_sb;
  int32_t b_sk, b_sn, b_sb;
  int32_t c_sm, c_sn, c_sb;
  int32_t cin_sm, cin_sn, cin_sb;
};
static_assert(sizeof(GemmParams) == 112 && offsetof(GemmParams, a_sm) == 64, "must match shaders/gemm.slang");

struct GemvParams {
  uint64_t mat, vec, out, cin;
  uint64_t alpha, beta;  // pack_param slots
  uint32_t rows, k;
  int32_t m_sr, m_sk, m_sb;
  int32_t v_sk, v_sb;
  int32_t o_sr, o_sb;
  int32_t c_sr, c_sb;
};
static_assert(sizeof(GemvParams) == 96 && offsetof(GemvParams, m_sr) == 56 && offsetof(GemvParams, c_sb) == 88,
              "must match shaders/gemv.slang");

constexpr int64_t kTile = 64;  // BM = BN in shaders/gemm.slang
constexpr int64_t kGemvRows0 = 8, kGemvRows1 = 64;  // rows per workgroup in shaders/gemv.slang's modes
constexpr int64_t kSgThreads = 128, kSgRows = 8;  // WG and MR in shaders/gemmsg.slang
constexpr int64_t kSgOccupancy = 8;  // gemmsg workgroups an SM runs at once (k is split below that)
constexpr int64_t kTile4M = 8, kTile4N = 128;  // BM, BN in shaders/gemm4.slang
constexpr int64_t kMaxGroups = 65535;

struct SplitKParams {
  uint64_t ws, c, cin;
  uint64_t alpha, beta;  // pack_param slots
  uint32_t m, n, splits, stride;
  int32_t c_sm, c_sn, c_sb;
  int32_t cin_sm, cin_sn, cin_sb;
};
static_assert(sizeof(SplitKParams) == 80, "must match shaders/splitk.slang");

// shaders/gemmcm.slang's tiles (its axis TILE): label, BM, BN.
struct CoopTile {
  const char* label;
  int64_t bm, bn;
};
constexpr CoopTile kCoopTiles[] = {{"l", 128, 256}, {"m", 128, 128}, {"s", 64, 128}};

int64_t env_int(const char* name, int64_t fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != 0 ? std::atoll(v) : fallback;
}

// The device's SMs or compute units (0: unknown).
int64_t compute_units() {
  static const int64_t units = vkt_compute_units();
  return units;
}

// Testing and tuning switches: TORCH_VK_GEMM_TILE (l, m or s) forces
// gemmcm.slang's tile, TORCH_VK_SPLIT_K=n splits every product's k n ways
// where it can (1: never), TORCH_VK_GEMM_GROUP=g the launch order's bands
// of tile rows (0: row by row).
std::string forced_tile() {
  static const std::string tile = [] {
    const char* env = std::getenv("TORCH_VK_GEMM_TILE");
    return std::string(env != nullptr ? env : "");
  }();
  return tile;
}
int64_t forced_splits() {
  static const int64_t s = env_int("TORCH_VK_SPLIT_K", 0);
  return s;
}
int64_t gemm_group() {
  static const int64_t g = env_int("TORCH_VK_GEMM_GROUP", -1);
  return g;
}

// TORCH_VK_GEMM: "tiled" (gemm.slang), "float4" (gemm4.slang where it can),
// "subgroup" (gemmsg.slang where the device can run it), or unset. Any but
// unset turns off gemmcm.slang (float16 on cooperative matrices), so that
// float16 products then run in float32 like other 16-bit ones.
std::string gemm_choice() {
  static const std::string choice = [] {
    const char* env = std::getenv("TORCH_VK_GEMM");
    return std::string(env != nullptr ? env : "");
  }();
  return choice;
}

// Whether float32 and int32 products use shaders/gemm4.slang where they
// can: on Arm GPUs (Mali) by default.
bool prefer_gemm4() {
  static const bool prefer = gemm_choice() == "float4" || (gemm_choice().empty() && vkt_vendor_id() == 0x13B5);
  return prefer;
}

// Lanes of shaders/gemmsg.slang for float32 or float64 products, 0 where it
// is not used: it is the default wherever the device can run it (subgroup
// size control, shuffles), except on Arm GPUs, which keep gemm4.
uint32_t subgroup_lanes(at::ScalarType t) {
  static const bool wanted = gemm_choice() == "subgroup" || (gemm_choice().empty() && vkt_vendor_id() != 0x13B5);
  static const uint32_t f32 = wanted ? vkt_kernel_subgroup_size("gemmsg_f32") : 0;
  static const uint32_t f64 = wanted && vkt_has_float64() ? vkt_kernel_subgroup_size("gemmsg_f64") : 0;
  return t == at::kFloat ? f32 : t == at::kDouble ? f64 : 0;
}

// Whether float16 (bfloat16) products use shaders/gemmcm.slang: where the
// device has float16 (bfloat16) cooperative matrices in subgroups of 32 (and
// TORCH_VK_GEMM is unset).
bool use_coopmat(at::ScalarType t) {
  static const bool f16 = gemm_choice().empty() && has_feature(kFeatureCoopMatF16) &&
                          vkt_kernel_subgroup_size("gemmcm_f16") == 32;
  static const bool bf16 = gemm_choice().empty() && has_feature(kFeatureCoopMatBf16) &&
                           vkt_kernel_subgroup_size("gemmcm_bf16") == 32;
  return t == at::kHalf ? f16 : t == at::kBFloat16 ? bf16 : false;
}

// gemm4's layout: row-major A and B (unit k and n strides) whose rows and
// batches start 16-byte aligned, K a multiple of 4, and B's rows N wide
// rounded up to 4 (a padded copy's, when N is not a multiple of 4).
bool float4_rows(const at::Tensor& t) {
  return t.stride(2) == 1 && t.stride(1) % 4 == 0 && t.stride(0) % 4 == 0 && gpu_address(t) % 16 == 0;
}
int64_t round4(int64_t x) { return (x + 3) / 4 * 4; }
bool gemm4_layout(const at::Tensor& a, const at::Tensor& b, int64_t n, int64_t k) {
  return k % 4 == 0 && b.size(2) >= round4(n) && float4_rows(a) && float4_rows(b);
}

// t [B, R, C] in a zeroed [B, rows, cols] (rows >= R, cols >= C), row-major.
at::Tensor padded(const at::Tensor& t, int64_t rows, int64_t cols) {
  if (t.size(1) == rows && t.size(2) == cols) return t.contiguous();
  at::Tensor out = at::zeros({t.size(0), rows, cols}, t.options());
  out.narrow(1, 0, t.size(1)).narrow(2, 0, t.size(2)).copy_(t);
  return out;
}

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

// gemm()'s product when C has one column (N == 1: c[r] = sum_k a[r, k] b[k])
// or one row (M == 1: c[r] = sum_k b[k, r] a[k]), as shaders/gemv.slang's
// matrix-vector product. The operands are already checked by gemm(). Returns
// false if the grid is too large (gemm.slang then takes it).
bool gemv(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin, double alpha,
          double beta) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2), k = a.size(2);
  bool column = n == 1;  // the product's rows are C's rows (else C's columns)
  const at::Tensor& mat = column ? a : b;
  const at::Tensor& vec = column ? b : a;
  int64_t rows = column ? m : n;
  auto m_sr = static_cast<int32_t>(column ? mat.stride(1) : mat.stride(2));
  auto m_sk = static_cast<int32_t>(column ? mat.stride(2) : mat.stride(1));
  // Threads along k when k is contiguous (or neither stride is), else along rows.
  int64_t mode = (m_sk == 1 || m_sr != 1) ? 0 : 1;
  int64_t per_group = mode == 0 ? kGemvRows0 : kGemvRows1;
  int64_t groups = (rows + per_group - 1) / per_group;
  if (groups > kMaxGroups) {
    return false;
  }
  int64_t rdim = column ? 1 : 2;  // C's (and cin's) dim along the product's rows
  bool use_cin = cin != nullptr && beta != 0.0;
  GemvParams p{};
  p.mat = gpu_address(mat);
  p.vec = gpu_address(vec);
  p.out = gpu_address_out(c);
  p.cin = use_cin ? gpu_address(*cin) : 0;
  p.alpha = pack_param(alpha, c.scalar_type());
  p.beta = pack_param(beta, c.scalar_type());
  p.rows = static_cast<uint32_t>(rows);
  p.k = static_cast<uint32_t>(k);
  p.m_sr = m_sr;
  p.m_sk = m_sk;
  p.m_sb = static_cast<int32_t>(mat.stride(0));
  p.v_sk = static_cast<int32_t>(column ? vec.stride(1) : vec.stride(2));
  p.v_sb = static_cast<int32_t>(vec.stride(0));
  p.o_sr = static_cast<int32_t>(c.stride(rdim));
  p.o_sb = static_cast<int32_t>(c.stride(0));
  if (use_cin) {
    p.c_sr = static_cast<int32_t>(cin->stride(rdim));
    p.c_sb = static_cast<int32_t>(cin->stride(0));
  }
  dispatch(std::string("gemv_") + dtype_label(c.scalar_type()), {use_cin ? 1u : 0u, static_cast<uint32_t>(mode)}, p,
           static_cast<uint32_t>(groups), static_cast<uint32_t>(batch));
  return true;
}

int64_t ceil_div(int64_t x, int64_t y) { return (x + y - 1) / y; }

}  // namespace

// ------------------------------------------------------------ gemv4.slang

namespace {

struct Gemv4Params {
  uint64_t w, x, out, cin, scales;
  uint64_t alpha, beta;  // float bits
  uint32_t rows, k, w_sr;
  int32_t x_sc, o_sr, o_sc, c_sr, c_sc;
  uint32_t group, ncol;
};
static_assert(sizeof(Gemv4Params) == 96 && offsetof(Gemv4Params, rows) == 56 && offsetof(Gemv4Params, group) == 88,
              "must match shaders/gemv4.slang");

// TORCH_VK_GEMV: "vector" (gemv4.slang for products with one to eight
// vectors), "scalar" (gemv.slang or the GEMM kernels for float products,
// qmm8.slang for int8 weights), or unset (see use_gemv4).
std::string gemv_choice() {
  static const std::string choice = [] {
    const char* env = std::getenv("TORCH_VK_GEMV");
    return std::string(env != nullptr ? env : "");
  }();
  return choice;
}

}  // namespace

bool use_gemv4(at::ScalarType t) {
  // By default everywhere but for float32 on NVIDIA and AMD GPUs, where
  // gemv.slang reads near the memory's bandwidth already (elsewhere, and for
  // 16-bit and quantised weights, the old paths widen to float32 first, read
  // one element per load, or take GEMM tiles for two to eight vectors).
  static const uint32_t vendor = vkt_vendor_id();
  static const bool all = gemv_choice() == "vector" || (gemv_choice().empty() && vendor != 0x10DE && vendor != 0x1002);
  static const bool narrow = gemv_choice() != "scalar";
  return all || (narrow && t != at::kFloat);
}

bool gemv4(const Gemv4& g) {
  int64_t ncol = g.x.size(0), rows = g.rows, k = g.k;
  at::ScalarType t = g.x.scalar_type();
  const char* xl = t == at::kFloat ? "f32" : t == at::kHalf ? "f16" : t == at::kBFloat16 ? "bf16" : nullptr;
  std::string w = g.weights;
  if (xl == nullptr || storage_label(t) == nullptr || ncol < 1 || ncol > 8 || rows < 1 || k < 1 ||
      g.out.scalar_type() != t || (g.cin != nullptr && g.cin->scalar_type() != t) ||
      (w == "q8" && storage_label(at::kChar) == nullptr) || (w == "q4" && storage_label(at::kShort) == nullptr)) {
    return false;
  }
  // (elements of W per 16-byte chunk, of x per 16-byte load: the shader's CHUNK)
  int64_t esize = c10::elementSize(t);
  int64_t chunk = w == "q4" ? 32 : w == "q8" ? 16 : 16 / esize;
  int64_t wsize = w == "q8" ? 1 : esize;  // (q4: never addressed by element)
  bool aligned = g.w % 16 == 0 && g.w_row % 16 == 0;
  if ((w == "q4" && (!aligned || k % chunk != 0)) || g.w_row % wsize != 0 ||
      (rows - 1) * g.w_row + ceil_div(k * (w == "q4" ? 1 : 2 * wsize), 2) >= (int64_t(1) << 32)) {
    return false;
  }
  // The vectors contiguous along k, 16-byte aligned, padded with zeros to a
  // whole chunk (else a copy: they are small).
  at::Tensor x = g.x;
  int64_t kp = ceil_div(k, chunk) * chunk, step = 16 / esize;
  if (x.stride(1) != 1 || kp != k || (ncol > 1 && x.stride(0) % step != 0) || gpu_address(x) % 16 != 0) {
    at::Tensor padded = at::zeros({ncol, kp}, x.options());
    padded.narrow(1, 0, k).copy_(x);
    x = padded;
  }
  for (const at::Tensor* o : std::initializer_list<const at::Tensor*>{&x, &g.out, g.cin}) {
    if (o != nullptr && (max_offset(*o) < 0 || max_offset(*o) >= (int64_t(1) << 31))) {
      return false;
    }
  }
  // The kernel's vector count (1, 2, 4 or 8; its rows per thread follow:
  // 4 for one or two vectors, else 2), and the lanes per row: on the
  // Mali-G57, 32 for one or two vectors of float or int8 weights (13 GB/s
  // for float32 weights, the bus's speed), 8 for 4-bit weights and for four
  // or eight vectors, whose ALU work limits them.
  int64_t nc = ncol <= 2 ? ncol : ncol <= 4 ? 4 : 8;
  int64_t per_thread = nc <= 2 ? 4 : 2;
  int64_t lanes = w == "q4" || nc >= 4 ? 8 : 32;
  int64_t groups = ceil_div(rows, 64 / lanes * per_thread);
  if (groups > kMaxGroups) {
    return false;
  }
  bool use_cin = g.cin != nullptr && g.beta != 0.0;
  Gemv4Params p{};
  p.w = g.w;
  p.x = gpu_address(x);
  p.out = gpu_address_out(g.out);
  p.cin = use_cin ? gpu_address(*g.cin) : 0;
  p.scales = g.scales != nullptr ? gpu_address(*g.scales) : 0;
  p.alpha = pack_param(g.alpha, at::kFloat);
  p.beta = pack_param(g.beta, at::kFloat);
  p.rows = static_cast<uint32_t>(rows);
  p.k = static_cast<uint32_t>(k);
  p.w_sr = static_cast<uint32_t>(g.w_row);
  p.x_sc = static_cast<int32_t>(ncol > 1 ? x.stride(0) : 0);
  p.o_sr = static_cast<int32_t>(g.out.stride(1));
  p.o_sc = static_cast<int32_t>(g.out.stride(0));
  if (use_cin) {
    p.c_sr = static_cast<int32_t>(g.cin->stride(1));
    p.c_sc = static_cast<int32_t>(g.cin->stride(0));
  }
  p.group = static_cast<uint32_t>(g.group);
  p.ncol = static_cast<uint32_t>(ncol);
  std::vector<uint32_t> spec = {use_cin ? 1u : 0u, static_cast<uint32_t>(lanes), aligned ? 1u : 0u};
  dispatch(std::string("gemv4_") + w + xl + "_c" + std::to_string(nc), spec, p, static_cast<uint32_t>(groups));
  return true;
}

namespace {

// gemm()'s product as gemv4.slang's when one side has one to eight vectors
// and the matrix side's rows are contiguous along k: C's columns (N <= 8,
// W = A) or C's rows (M <= 8, W = B^T, as x @ W.t()). Batches of one only.
bool gemm_vectors(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin,
                  double alpha, double beta) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2), k = a.size(2);
  at::ScalarType t = c.scalar_type();
  if (!use_gemv4(t) || batch != 1 || k == 0 || a.scalar_type() != t || b.scalar_type() != t ||
      (t != at::kFloat && t != at::kHalf && t != at::kBFloat16)) {
    return false;
  }
  if (t != at::kFloat && use_coopmat(t) && std::min(m, n) > 1 && !use_gemv4(at::kFloat)) {
    return false;  // (two to eight vectors stay on the tensor cores, where float32 keeps gemv.slang)
  }
  for (const at::Tensor* x : {&a, &b, &c, cin}) {
    if (x != nullptr && (!is_vk(*x) || max_offset(*x) < 0 || max_offset(*x) >= (int64_t(1) << 31))) {
      return false;
    }
  }
  bool columns = n <= 8 && a.stride(2) == 1;  // W = A [m, k]: c[r, j] = sum_k a[r, k] b[k, j]
  bool rowwise = m <= 8 && b.stride(1) == 1;  // W = B^T [n, k]: c[j, r] = sum_k b[k, r] a[j, k]
  if (columns && rowwise) {
    columns = m >= n;  // (the longer side as W's rows)
  }
  if (!columns && !rowwise) {
    return false;
  }
  int64_t esize = c10::elementSize(t);
  Gemv4 g;
  g.k = k;
  g.alpha = alpha;
  g.beta = beta;
  at::Tensor cin2;
  if (columns) {
    g.w = gpu_address(a);
    g.w_row = a.stride(1) * esize;
    g.rows = m;
    g.x = b.select(0, 0).t();    // [n, k]
    g.out = c.select(0, 0).t();  // [n, m]
    if (cin != nullptr) cin2 = cin->select(0, 0).t();
  } else {
    g.w = gpu_address(b);
    g.w_row = b.stride(2) * esize;
    g.rows = n;
    g.x = a.select(0, 0);    // [m, k]
    g.out = c.select(0, 0);  // [m, n]
    if (cin != nullptr) cin2 = cin->select(0, 0);
  }
  if (g.rows > 1 && g.w_row < k * esize) {
    return false;  // (rows overlapping: a broadcast matrix)
  }
  if (cin != nullptr) g.cin = &cin2;
  return gemv4(g);
}

// gemm.slang's Params (gemm4.slang's and gemmsg.slang's too) for c = alpha
// a @ b (+ beta cin): cin is read only when beta != 0 (p.cin = 0 otherwise).
GemmParams gemm_params(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin,
                       double alpha, double beta) {
  bool use_cin = cin != nullptr && beta != 0.0;
  GemmParams p{};
  p.a = gpu_address(a);
  p.b = gpu_address(b);
  p.c = gpu_address_out(c);
  p.cin = use_cin ? gpu_address(*cin) : 0;
  p.alpha = pack_param(alpha, c.scalar_type());
  p.beta = pack_param(beta, c.scalar_type());
  p.m = static_cast<uint32_t>(c.size(1));
  p.n = static_cast<uint32_t>(c.size(2));
  p.k = static_cast<uint32_t>(a.size(2));
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
  return p;
}

// How gemmcm.slang stages an operand [B, R, C] (A: R = m, C = k; B: R = k,
// C = n): 1 in 16-byte pieces along C (unit C stride), 2 along R (unit R
// stride), where the pieces are 16-byte aligned; else 0, value by value.
uint32_t coop_mode(const at::Tensor& t) {
  bool aligned = gpu_address(t) % 16 == 0 && t.stride(0) % 8 == 0;
  if (aligned && t.stride(2) == 1 && t.stride(1) % 8 == 0) return 1;
  if (aligned && t.stride(1) == 1 && t.stride(2) % 8 == 0) return 2;
  return 0;
}


// How many pieces of k would let `groups` workgroups fill `slots` (the
// workgroups the GPU runs at once, about), as llama.cpp's Vulkan backend
// chooses: none while k is under `min_k` or the workgroups fill more than
// two thirds of the slots, 3 pieces between a half and two thirds, else
// enough to fill them, at most `cap`.
int64_t fill_splits(int64_t groups, int64_t slots, int64_t k, int64_t min_k, int64_t cap) {
  if (slots <= 0 || k < min_k) return 1;
  if (2 * groups <= slots) return std::min(slots / groups, cap);
  if (3 * groups <= 2 * slots) return 3;
  return 1;
}

// The pieces k is split into (TORCH_VK_SPLIT_K, else `wanted`), at most one
// per 256 values of k, and the k of each (kc, a multiple of 256; the last
// piece may be shorter): 1 for no split.
int64_t split_pieces(int64_t wanted, int64_t k, int64_t& kc) {
  int64_t s = forced_splits() > 0 ? forced_splits() : wanted;
  s = std::min(s, ceil_div(k, 256));
  if (s <= 1) {
    return 1;
  }
  kc = ceil_div(ceil_div(k, s), 256) * 256;
  return ceil_div(k, kc);
}

// c = alpha a @ b (+ beta cin) split along k: GEMM kernel `kernel` (with
// spec constants `spec`, whose [0] is USE_CIN and [split_spec] SPLIT; p from
// gemm_params) writes the `splits` partial sums of each product (k in
// pieces of kc) to a workspace of `partial` values, and shaders/splitk.slang
// adds them up in a fixed order, scales them, adds beta cin and rounds:
// results are the same from run to run. Returns false where the workspace
// would be too large to address (the caller then runs the product whole).
bool split_k_gemm(const std::string& kernel, std::vector<uint32_t> spec, size_t split_spec, GemmParams p, int64_t gx,
                  int64_t gy, int64_t splits, int64_t kc, const at::Tensor& c, const at::Tensor* cin, double alpha,
                  double beta, at::ScalarType partial) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2);
  if (batch * splits > kMaxGroups || batch * splits * m * n > (int64_t(1) << 28)) {
    return false;
  }
  at::Tensor ws = at::empty({batch * splits, m, n}, c.options().dtype(partial));
  bool use_cin = p.cin != 0;
  GemmParams q = p;
  q.c = gpu_address_out(ws);
  q.cin = 0;
  q.kc = static_cast<uint32_t>(kc);
  q.c_sb = static_cast<int32_t>(m * n);
  q.c_sm = static_cast<int32_t>(n);
  q.c_sn = 1;
  spec[0] = 0;
  spec[split_spec] = 1;
  dispatch(kernel, spec, q, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), static_cast<uint32_t>(batch * splits));
  SplitKParams r{};
  r.ws = gpu_address(ws);
  r.c = gpu_address_out(c);
  r.cin = use_cin ? gpu_address(*cin) : 0;
  r.alpha = pack_param(alpha, partial);
  r.beta = pack_param(beta, partial);
  r.m = static_cast<uint32_t>(m);
  r.n = static_cast<uint32_t>(n);
  r.splits = static_cast<uint32_t>(splits);
  r.c_sm = static_cast<int32_t>(c.stride(1));
  r.c_sn = static_cast<int32_t>(c.stride(2));
  r.c_sb = static_cast<int32_t>(c.stride(0));
  if (use_cin) {
    r.cin_sm = static_cast<int32_t>(cin->stride(1));
    r.cin_sn = static_cast<int32_t>(cin->stride(2));
    r.cin_sb = static_cast<int32_t>(cin->stride(0));
  }
  uint32_t groups = 0;
  check(vkt_grid_1d(static_cast<uint64_t>(m * n), &groups, &r.stride));
  at::ScalarType t = c.scalar_type();
  const char* label = t == at::kHalf ? "f16" : t == at::kBFloat16 ? "bf16" : t == at::kFloat ? "f32" : "f64";
  dispatch(std::string("splitk_") + label, {use_cin ? 1u : 0u}, r, groups, static_cast<uint32_t>(batch));
  return true;
}

// gemmcm.slang's tile for a batch of m x n products and the pieces k is
// split into, by the GPU's SM count U (docs/log/M35: measured on the
// L40S; TORCH_VK_GEMM_TILE forces the tile). An SM runs one
// large tile (8 subgroups) at a time, about as fast as two smaller ones.
// - U unknown: the large tile.
// - At least U large tiles: whole rounds of them, unless the medium tile's
//   rounds would be fuller by a tenth (at 3072^3 the large tiles' third
//   round runs 4 of them).
// - Fewer: split k to fill the SMs (where k >= 2048: weight gradients'
//   x.t() @ y); else the large tile when they fill two thirds of the SMs,
//   else the small tile (more workgroups for a small product).
struct CoopPlan {
  const CoopTile* tile;
  int64_t splits;
};
CoopPlan coop_plan(int64_t m, int64_t n, int64_t k, int64_t batch) {
  int64_t units = compute_units();
  auto tiles = [&](const CoopTile& t) { return ceil_div(m, t.bm) * ceil_div(n, t.bn) * batch; };
  const CoopTile &large = kCoopTiles[0], &medium = kCoopTiles[1], &small = kCoopTiles[2];
  for (const CoopTile& t : kCoopTiles) {
    if (forced_tile() == t.label) return {&t, fill_splits(tiles(t), units, k, 2048, 8)};
  }
  if (units == 0) {
    return {&large, 1};
  }
  int64_t tl = tiles(large), tm = tiles(medium);
  if (tl >= units) {
    auto full = [&](int64_t t) { return double(t) / double(ceil_div(t, units) * units); };
    return {full(tm) >= full(tl) + 0.1 ? &medium : &large, 1};
  }
  int64_t s = fill_splits(tl, units, k, 2048, 8);
  if (s > 1 || 3 * tl > 2 * units) {
    return {&large, s};
  }
  return {&small, 1};
}

// gemm() for float16 and bfloat16: shaders/gemmcm.slang on cooperative
// matrices, products accumulated in float32 and rounded once. Returns false
// where it does not run (no cooperative matrices of the type, matrix-vector
// products: those take the float32 route of other 16-bit products).
bool coop_gemm_16(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin,
                  double alpha, double beta) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2), k = a.size(2);
  at::ScalarType t = c.scalar_type();
  if (!use_coopmat(t) || a.scalar_type() != t || b.scalar_type() != t || (cin != nullptr && cin->scalar_type() != t) ||
      m == 1 || n == 1 || k == 0) {
    return false;  // (k = 0 too: gemmcm.slang's pipelined k loop needs one tile; it hung on NVIDIA)
  }
  for (const at::Tensor* t : {&a, &b, &c, cin}) {
    if (t != nullptr && (!is_vk(*t) || max_offset(*t) < 0 || max_offset(*t) >= (int64_t(1) << 31))) {
      return false;
    }
  }
  CoopPlan plan = coop_plan(m, n, k, batch);
  const CoopTile& tile = *plan.tile;
  int64_t gx = ceil_div(n, tile.bn), gy = ceil_div(m, tile.bm);
  if (gx > kMaxGroups || gy > kMaxGroups || batch > kMaxGroups) {
    return false;
  }
  if (c.numel() == 0) {
    return true;
  }
  GemmParams p = gemm_params(a, b, c, cin, alpha, beta);
  p.alpha = pack_param(alpha, at::kFloat);  // (the kernel scales in float32)
  p.beta = pack_param(beta, at::kFloat);
  std::string kernel = std::string(t == at::kHalf ? "gemmcm_f16_" : "gemmcm_bf16_") + tile.label;
  int64_t group = gemm_group() >= 0 ? gemm_group() : 0;
  std::vector<uint32_t> spec = {p.cin != 0 ? 1u : 0u, coop_mode(a), coop_mode(b), 0u, static_cast<uint32_t>(group)};
  int64_t kc = 0, splits = split_pieces(plan.splits, k, kc);
  if (splits > 1 && split_k_gemm(kernel, spec, 3, p, gx, gy, splits, kc, c, cin, alpha, beta, at::kFloat)) {
    return true;
  }
  dispatch(kernel, spec, p, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), static_cast<uint32_t>(batch));
  return true;
}

// c = alpha * a @ b (+ beta * cin), all 3-D views: a [B, M, K], b [B, K, N],
// c and cin [B, M, N] (cin may broadcast through zero strides). Returns false
// if the kernel cannot handle the case.
bool gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin, double alpha,
          double beta) {
  int64_t batch = c.size(0), m = c.size(1), n = c.size(2), k = a.size(2);
  if ((n <= 8 || m <= 8) && c.numel() > 0 && gemm_vectors(a, b, c, cin, alpha, beta)) {
    return true;
  }
  if (c.scalar_type() == at::kHalf || c.scalar_type() == at::kBFloat16) {
    return coop_gemm_16(a, b, c, cin, alpha, beta);
  }
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
  if ((n == 1 || m == 1) && gemv(a, b, c, cin, alpha, beta)) {
    return true;
  }
  if (uint32_t lanes = subgroup_lanes(c.scalar_type()); lanes != 0) {
    // (WG / S) MR x S NR tiles; NR = 4, or 2 for float64 with 32 lanes
    int64_t nr = c.scalar_type() == at::kDouble && lanes == 32 ? 2 : 4;
    int64_t tile_m = kSgThreads / lanes * kSgRows, tile_n = lanes * nr;
    int64_t sx = (n + tile_n - 1) / tile_n, sy = (m + tile_m - 1) / tile_m;
    if (sx <= kMaxGroups && sy <= kMaxGroups) {
      GemmParams p = gemm_params(a, b, c, cin, alpha, beta);
      std::string kernel = std::string("gemmsg_") + dtype_label(c.scalar_type());
      std::vector<uint32_t> spec = {p.cin != 0 ? 1u : 0u, 0u};
      int64_t kc = 0;
      int64_t splits = split_pieces(fill_splits(sx * sy * batch, kSgOccupancy * compute_units(), k, 1024, 16), k, kc);
      if (splits > 1 && split_k_gemm(kernel, spec, 1, p, sx, sy, splits, kc, c, cin, alpha, beta, c.scalar_type())) {
        return true;
      }
      dispatch(kernel, spec, p, static_cast<uint32_t>(sx), static_cast<uint32_t>(sy), static_cast<uint32_t>(batch));
      return true;
    }
  }
  bool gemm4_type = c.scalar_type() == at::kFloat || c.scalar_type() == at::kInt;
  if (gemm4_type && prefer_gemm4() && !gemm4_layout(a, b, n, k)) {
    // Another layout (as x @ W.t()'s column-major W.t()), or K or N not
    // multiples of 4: row-major copies (zero-padded to multiples of 4: the
    // padding adds zero products), then gemm4, when the product's work is
    // 16x the copies' or more. On the A13: x @ W.t() at 2048^3 0.57 s against
    // 1.2 s in the tiles.
    int64_t k4 = round4(k), n4 = round4(n), work = batch * m * n * k;
    bool copy_a = !float4_rows(a) || k4 != k, copy_b = !float4_rows(b) || k4 != k || n4 != n;
    int64_t copied = (copy_a ? a.size(0) * m * k4 : 0) + (copy_b ? b.size(0) * k4 * n4 : 0);
    if (copied * 16 <= work) {
      at::Tensor a2 = copy_a ? padded(a, m, k4) : a, b2 = copy_b ? padded(b, k4, n4) : b;
      if (gemm4_layout(a2, b2, n, k4)) return gemm(a2, b2, c, cin, alpha, beta);
    }
  }
  GemmParams p = gemm_params(a, b, c, cin, alpha, beta);
  bool use_cin = p.cin != 0;
  if (gemm4_type && prefer_gemm4() && gemm4_layout(a, b, n, k) && (m + kTile4M - 1) / kTile4M <= kMaxGroups) {
    dispatch(std::string("gemm4_") + dtype_label(c.scalar_type()), {use_cin ? 1u : 0u}, p, static_cast<uint32_t>((n + kTile4N - 1) / kTile4N),
             static_cast<uint32_t>((m + kTile4M - 1) / kTile4M), static_cast<uint32_t>(batch));
    return true;
  }
  dispatch(std::string("gemm_") + dtype_label(c.scalar_type()), {use_cin ? 1u : 0u}, p,
           static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), static_cast<uint32_t>(batch));
  return true;
}

// Index ranges [lo, lo + size) per dimension of view t within a row-major
// parent whose strides are t's, or false if t is not such a view.
bool parent_box(const at::Tensor& t, std::vector<int64_t>& lo) {
  int64_t off = t.storage_offset();
  lo.assign(t.dim(), 0);
  for (int64_t d = 0; d < t.dim(); ++d) {
    int64_t s = t.stride(d);
    if (s <= 0 || (d > 0 && (t.stride(d - 1) % s != 0 || t.stride(d - 1) <= s))) {
      return false;
    }
    lo[d] = off / s;
    off %= s;
    if (d > 0 && lo[d] + t.size(d) > t.stride(d - 1) / s) {
      return false;
    }
  }
  return off == 0;
}

// Whether views x and y certainly share no element: different storages,
// disjoint byte ranges, or disjoint boxes in one row-major parent (e.g. a
// blocked LU's L21 and A22).
bool disjoint(const at::Tensor& x, const at::Tensor& y) {
  if (x.numel() == 0 || y.numel() == 0 || !x.has_storage() || !y.has_storage() ||
      !x.storage().is_alias_of(y.storage())) {
    return true;
  }
  int64_t xs = max_offset(x), ys = max_offset(y);
  if (xs < 0 || ys < 0) {
    return false;
  }
  int64_t x0 = x.storage_offset() * x.element_size(), y0 = y.storage_offset() * y.element_size();
  if (x0 + (xs + 1) * x.element_size() <= y0 || y0 + (ys + 1) * y.element_size() <= x0) {
    return true;
  }
  std::vector<int64_t> xl, yl;
  if (x.scalar_type() != y.scalar_type() || x.strides() != y.strides() || !parent_box(x, xl) || !parent_box(y, yl)) {
    return false;
  }
  for (int64_t d = 0; d < x.dim(); ++d) {
    if (xl[d] + x.size(d) <= yl[d] || yl[d] + y.size(d) <= xl[d]) {
      return true;
    }
  }
  return false;
}

// Complex c = alpha a @ b (+ beta cin): four real GEMMs on the real and
// imaginary parts (strided views, no copies). With real alpha and beta they
// accumulate straight into c's parts (no temporaries: a blocked LU's
// trailing updates are most of the matrix); otherwise, or when c overlaps
// a or b, into new tensors combined with complex element-wise ops.
bool complex_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, const at::Tensor* cin,
                  const at::Scalar& alpha, const at::Scalar& beta) {
  at::ScalarType t = c.scalar_type();
  if (!at::isComplexType(t) || dtype_label(t) == nullptr || a.scalar_type() != t || b.scalar_type() != t ||
      !is_vk(a) || !is_vk(b) || !is_vk(c)) {
    return false;
  }
  at::Tensor ar = at::view_as_real(a.resolve_conj()), br = at::view_as_real(b.resolve_conj());
  at::Tensor a_re = ar.select(-1, 0), a_im = ar.select(-1, 1), b_re = br.select(-1, 0), b_im = br.select(-1, 1);
  bool use_cin = cin != nullptr && !beta.equal(0);
  if (!alpha.isComplex() && !beta.isComplex() && !c.is_conj() && disjoint(c, a) && disjoint(c, b)) {
    at::Tensor cr = at::view_as_real(c);
    at::Tensor c_re = cr.select(-1, 0), c_im = cr.select(-1, 1);
    at::Tensor cinr = use_cin ? at::view_as_real(cin->resolve_conj()) : at::Tensor();
    // (All four GEMMs must run once the first has written c.)
    for (const at::Tensor* v : {&ar, &br, &cr, use_cin ? &cinr : &cr}) {
      if (max_offset(*v) < 0 || max_offset(*v) >= (int64_t(1) << 31)) {
        return false;
      }
    }
    at::Tensor cin_re = use_cin ? cinr.select(-1, 0) : at::Tensor(), cin_im = use_cin ? cinr.select(-1, 1) : at::Tensor();
    double al = alpha.toDouble(), be = use_cin ? beta.toDouble() : 0.0;
    // (cin may be c itself: each part of cin is read by the first GEMM
    // writing that part, element by element.)
    return gemm(a_re, b_re, c_re, use_cin ? &cin_re : nullptr, al, be) && gemm(a_im, b_im, c_re, &c_re, -al, 1.0) &&
           gemm(a_re, b_im, c_im, use_cin ? &cin_im : nullptr, al, be) && gemm(a_im, b_re, c_im, &c_im, al, 1.0);
  }
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
  if (use_cin) {
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
  TORCH_CHECK(out.dtype() == self.dtype(), "Expected out tensor to have dtype ", self.dtype(), ", but got ", out.dtype(),
              " instead");
  TORCH_CHECK(self.dtype() == mat2.dtype(), "expected m1 and m2 to have the same dtype, but got: ", self.dtype(),
              " != ", mat2.dtype());  // (the CPU's check; its type names, too)
  if (is_vk(self) && is_vk(mat2) && is_vk(out) && (out.numel() == 0 || self.size(1) == 0)) {
    if (out.numel() != 0) out.zero_();  // (k = 0: sums of nothing)
    return out;
  }
  if (!any_gemm(self.unsqueeze(0), mat2.unsqueeze(0), out.unsqueeze(0), nullptr, 1, 0)) {
    on_cpu("aten::mm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& bmm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  at::native::resize_output(out, {self.size(0), self.size(1), mat2.size(2)});
  if (is_vk(self) && is_vk(mat2) && is_vk(out) && self.dim() == 3 && mat2.dim() == 3 &&
      self.scalar_type() == mat2.scalar_type() && (out.numel() == 0 || self.size(2) == 0)) {
    if (out.numel() != 0) out.zero_();  // (k = 0: sums of nothing)
    return out;
  }
  if (is_vk(self) && is_vk(mat2) && is_vk(out) && self.dim() == 3 && mat2.dim() == 3 && self.size(0) > 1 &&
      (self.stride(0) == 0 || mat2.stride(0) == 0) && self.scalar_type() == mat2.scalar_type() &&
      out.scalar_type() == self.scalar_type() && out.numel() > 0) {
    // (one matrix for every batch, as matmul's broadcasting gives: one product
    // of the batches side by side, not many tiny ones)
    int64_t B = self.size(0), m = self.size(1), k = self.size(2), n = mat2.size(2);
    at::Tensor r;
    if (self.stride(0) == 0) {
      r = at::mm(self.select(0, 0), mat2.permute({1, 0, 2}).reshape({k, B * n})).view({m, B, n}).permute({1, 0, 2});
    } else {
      r = at::mm(self.reshape({B * m, k}), mat2.select(0, 0)).view({B, m, n});
    }
    out.copy_(r);
    return out;
  }
  constexpr int64_t kMaxBatch = 65535;  // (workgroups per dispatch dimension)
  if (is_vk(self) && is_vk(mat2) && is_vk(out) && self.dim() == 3 && mat2.dim() == 3 && self.size(0) > kMaxBatch &&
      out.is_contiguous()) {
    // (more matrices than one dispatch takes: in chunks)
    bool ok = true;
    for (int64_t s = 0; s < self.size(0) && ok; s += kMaxBatch) {
      int64_t len = std::min(kMaxBatch, self.size(0) - s);
      ok = any_gemm(self.narrow(0, s, len), mat2.narrow(0, s, len), out.narrow(0, s, len), nullptr, 1, 0);
    }
    if (ok) return out;
  }
  if (!any_gemm(self, mat2, out, nullptr, 1, 0)) {
    on_cpu("aten::bmm", "out", self, mat2, out);
  }
  return out;
}

at::Tensor& addmm_out(const at::Tensor& self, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  int64_t m = mat1.size(0), n = mat2.size(1);
  at::native::resize_output(out, {m, n});
  TORCH_CHECK(out.dtype() == self.dtype(), "Expected out tensor to have dtype ", self.dtype(), ", but got ", out.dtype(),
              " instead");  // (the CPU's check; its type names, too)
  if (is_vk(self) && is_vk(mat1) && is_vk(mat2) && is_vk(out) && mat1.dim() == 2 && mat2.dim() == 2 &&
      (out.numel() == 0 || mat1.size(1) == 0) && self.scalar_type() == mat1.scalar_type() &&
      mat1.scalar_type() == mat2.scalar_type() && out.scalar_type() == self.scalar_type()) {
    // (nothing to multiply: beta self, or zeros for beta 0, as the CPU)
    if (out.numel() == 0) return out;
    at::Tensor s = self.expand({m, n});
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else if (out.scalar_type() == at::kBool) {
      // (the CPU's beta * self: an integer or float beta gives a result bools cannot hold)
      at::ScalarType rt = at::result_type(s, beta);
      TORCH_CHECK(rt == at::kBool, "result type ", rt, " can't be cast to the desired output type Bool");
      out.copy_(s);
    } else {
      out.copy_(at::mul(s, beta));
    }
    return out;
  }
  bool done = false;
  if (is_vk(self)) {
    at::Tensor cin = self.expand({m, n}).unsqueeze(0);  // (more than 2 dims: expand's error, as the CPU's)
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
  if (is_vk(self) && is_vk(batch1) && is_vk(batch2) && is_vk(out) && batch1.dim() == 3 && batch2.dim() == 3 &&
      self.dim() <= 3 && (out.numel() == 0 || batch1.size(2) == 0) && self.scalar_type() == out.scalar_type()) {
    // (nothing to multiply: beta self, or zeros for beta 0, as the CPU)
    if (out.numel() == 0) return out;
    at::Tensor s = self.expand(shape);
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else if (out.scalar_type() == at::kBool) {
      at::ScalarType rt = at::result_type(s, beta);
      TORCH_CHECK(rt == at::kBool, "result type ", rt, " can't be cast to the desired output type Bool");
      out.copy_(s);
    } else {
      out.copy_(at::mul(s, beta));
    }
    return out;
  }
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

// beta self + alpha outer(vec1, vec2), with the CPU kernel's checks (the
// composite kernel "vk" would use otherwise has none).
at::Tensor& addr_out(const at::Tensor& self, const at::Tensor& vec1, const at::Tensor& vec2, const at::Scalar& beta,
                     const at::Scalar& alpha, at::Tensor& out) {
  TORCH_CHECK(vec1.dim() == 1, "addr: Expected 1-D argument vec1, but got ", vec1.dim(), "-D");
  TORCH_CHECK(vec2.dim() == 1, "addr: Expected 1-D argument vec2, but got ", vec2.dim(), "-D");
  check_out_device(vec1, out);
  bool logical = out.scalar_type() == at::kBool;
  TORCH_CHECK(logical || !beta.isBoolean(), "Boolean beta only supported for Boolean results.");
  TORCH_CHECK(logical || !alpha.isBoolean(), "Boolean alpha only supported for Boolean results.");
  bool integral = !logical && !at::isFloatingType(out.scalar_type()) && !at::isComplexType(out.scalar_type());
  TORCH_CHECK(!integral || !beta.isFloatingPoint(),
              "For integral input tensors, argument beta must not be a floating point number.");
  TORCH_CHECK(!integral || !alpha.isFloatingPoint(),
              "For integral input tensors, argument alpha must not be a floating point number.");
  at::Tensor outer = at::outer(vec1, vec2);
  std::vector<int64_t> shape = {vec1.size(0), vec2.size(0)};
  at::Tensor base = self.expand(shape);
  at::Tensor r;
  if (logical) {
    at::Tensor kept = beta.toBool() ? base.to(at::kBool) : at::zeros(shape, out.options());
    r = at::logical_or(kept, alpha.toBool() ? outer.to(at::kBool) : at::zeros(shape, out.options()));
  } else {
    r = alpha.equal(1) ? outer : outer * alpha;
    if (!beta.equal(0)) {
      // (beta = 0 ignores self, NaNs included, as the CPU; a complex beta
      // multiplies first: add's alpha is real on "vk")
      r = beta.isComplex() ? at::add(r, base * beta) : at::add(r, base, beta);
    }
  }
  check_out_cast(r.scalar_type(), out);
  at::native::resize_output(out, shape);
  out.copy_(r);
  return out;
}

at::Tensor addr(const at::Tensor& self, const at::Tensor& vec1, const at::Tensor& vec2, const at::Scalar& beta,
                const at::Scalar& alpha) {
  at::Tensor out = at::empty({0}, vec1.options().dtype(at::promote_types(self.scalar_type(), at::result_type(vec1, vec2))));
  return addr_out(self, vec1, vec2, beta, alpha, out);
}

at::Tensor& addmv_out(const at::Tensor& self, const at::Tensor& mat, const at::Tensor& vec, const at::Scalar& beta,
                      const at::Scalar& alpha, at::Tensor& out) {
  int64_t m = mat.size(0);
  at::native::resize_output(out, {m});
  if (is_vk(out) && is_vk(self) && mat.numel() == 0 && self.dim() <= 1 && mat.dim() == 2 && vec.dim() == 1 &&
      mat.size(1) == vec.size(0) && (self.numel() == m || self.numel() == 1)) {
    // (an empty matrix, any dtype, as the CPU: beta * self, or zeros where beta
    // is 0, so self's NaNs do not propagate)
    if (beta.toComplexDouble() == 0.0) {
      out.zero_();
    } else if (m > 0) {
      at::mul_out(out, self.expand({m}), at::scalar_tensor(beta, self.options()));
    }
    return out;
  }
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
  // (the CPU's checks, in its order)
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1, "1D tensors expected, but got ", self.dim(), "D and ", other.dim(),
              "D tensors");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(), " and ", other.scalar_type());
  TORCH_CHECK(self.numel() == other.numel(), "inconsistent tensor size, expected tensor [", self.numel(),
              "] and src [", other.numel(), "] to have the same number of elements, but got ", self.numel(), " and ",
              other.numel(), " elements respectively");
  reject_dtype("dot", self.scalar_type(), self.scalar_type() == at::kBool);
  TORCH_CHECK(self.dim() == 1 && other.dim() == 1, "1D tensors expected, but got ", self.dim(), "D and ", other.dim(),
              "D tensors");
  TORCH_CHECK(self.size(0) == other.size(0), "inconsistent tensor size, expected tensor [", self.size(0),
              "] and src [", other.size(0), "] to have the same number of elements, but got ", self.size(0), " and ",
              other.size(0), " elements respectively");
  TORCH_CHECK(self.scalar_type() == other.scalar_type(), "dot : expected both vectors to have same dtype, but found ",
              self.scalar_type(), " and ", other.scalar_type());
  // Element-wise product and a sum reduction (both Vulkan kernels). dot
  // falls through the Conjugate key, so self and other may be conjugate
  // views: resolved here, as mul would not inside a TorchDispatchMode (which
  // excludes that key).
  return at::sum(at::mul(self.resolve_conj(), other.resolve_conj())).to(self.scalar_type());
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
  m.impl("addr", &vk::addr);
  m.impl("addr.out", &addr_out);
  m.impl("addbmm.out", &addbmm_out);
  m.impl("addbmm", &vk::addbmm);
  m.impl("dot", &dot);
}
