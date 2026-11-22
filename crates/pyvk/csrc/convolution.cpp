// Convolution for device "vk" (M12): convolution_overrideable and
// convolution_backward_overrideable (what aten::convolution dispatches to on
// a PrivateUse1 backend), for 1-3 spatial dims, groups, stride, padding,
// dilation and transposed convolution.
//
// Built from existing kernels:
//  - forward: im2col as a strided view of the padded input (one copy kernel)
//    and a batched GEMM against the weight (for one sample or one group,
//    the weight times the columns laid out as rows, the bias added by it);
//  - depthwise (one input channel per group): one multiply-add per tap;
//  - transposed, with a stride: one convolution of the input for all the
//    stride's phases (sub-pixel convolution), interleaved into the output;
//    without (or where the phases would not save work): an ordinary
//    convolution of the (zero-inserted) input with the flipped, transposed
//    weight (no accumulation into overlapping windows);
//  - backward: grad_input is a transposed convolution, grad_weight a GEMM of
//    grad_output against the same columns, grad_bias a sum.

#include "vk.h"

#include <torch/library.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace vk {

namespace {
std::atomic<bool>& fp16_arithmetic_flag() {
  static std::atomic<bool> flag{[] {
    const char* env = std::getenv("TORCH_VK_FP16_ARITHMETIC");
    return env != nullptr && *env != 0 && std::strcmp(env, "0") != 0;
  }()};
  return flag;
}
}  // namespace

bool fp16_arithmetic() { return fp16_arithmetic_flag().load(std::memory_order_relaxed); }

namespace {

int64_t ceil_div(int64_t x, int64_t y) { return (x + y - 1) / y; }

// Output length of one spatial dim of a (non-transposed) convolution.
int64_t conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad_lo, int64_t pad_hi, int64_t dilation) {
  return (in + pad_lo + pad_hi - dilation * (k - 1) - 1) / stride + 1;
}

// im2col: [N, prod(O), C * prod(K)] columns of x (already padded) for kernel
// sizes k, stride, dilation, as one copy of a strided view.
at::Tensor im2col_view(const at::Tensor& xp, at::IntArrayRef k, at::IntArrayRef stride, at::IntArrayRef dilation,
                       at::IntArrayRef out) {
  int64_t s = static_cast<int64_t>(k.size());
  std::vector<int64_t> sizes = {xp.size(0)}, strides = {xp.stride(0)};
  for (int64_t d = 0; d < s; ++d) {
    sizes.push_back(out[d]);
    strides.push_back(xp.stride(2 + d) * stride[d]);
  }
  sizes.push_back(xp.size(1));
  strides.push_back(xp.stride(1));
  for (int64_t d = 0; d < s; ++d) {
    sizes.push_back(k[d]);
    strides.push_back(xp.stride(2 + d) * dilation[d]);
  }
  at::Tensor windows = xp.as_strided(sizes, strides, xp.storage_offset());  // [N, *O, C, *K]
  int64_t L = 1, CK = xp.size(1);
  for (int64_t d = 0; d < s; ++d) {
    L *= out[d];
    CK *= k[d];
  }
  return windows.reshape({xp.size(0), L, CK});
}

// Zero-pads the spatial dims of x: pad[d] = (lo, hi), negative values crop.
at::Tensor pad_spatial(const at::Tensor& x, const std::vector<std::pair<int64_t, int64_t>>& pad) {
  std::vector<int64_t> p;  // constant_pad_nd order: last dim first
  bool any = false;
  for (int64_t d = static_cast<int64_t>(pad.size()) - 1; d >= 0; --d) {
    p.push_back(pad[d].first);
    p.push_back(pad[d].second);
    any = any || pad[d].first != 0 || pad[d].second != 0;
  }
  return any ? at::constant_pad_nd(x, p, 0) : x;
}

// t reversed along `dims`, by one copy per index (convolution weights: dims
// of a few taps; aten::flip gathers through a table of offsets, much slower
// for these shapes).
at::Tensor reversed(const at::Tensor& t, const std::vector<int64_t>& dims) {
  at::Tensor r = t;
  for (int64_t d : dims) {
    int64_t n = r.size(d);
    if (n <= 1 || r.numel() == 0) continue;
    at::Tensor o = at::empty(r.sizes(), r.options());
    for (int64_t i = 0; i < n; ++i) o.select(d, n - 1 - i).copy_(r.select(d, i));
    r = o;
  }
  return r;
}

// shaders/conv4s.slang's Params.
struct Conv4sParams {
  uint64_t x, w, y, bias;
  uint32_t cog, cg, ko, kh;
  int32_t off_d, off_h;
  uint32_t nv, plane;
  int32_t x_sn, w_sm, y_sn, y_sc;
  uint32_t wp, wo, hp, ho;
  uint32_t groups, kg;
};
static_assert(sizeof(Conv4sParams) == 104, "must match shaders/conv4s.slang");

// shaders/conv4s.slang's tiles (its axis TILE): label, rows per workgroup
// (WY R), columns per workgroup (16 lanes x 4 NC).
struct Conv4sTile {
  const char* label;
  int64_t rows, cols;
};
constexpr Conv4sTile kConv4sR8N1 = {"r8n1", 16, 64}, kConv4sR4N2 = {"r4n2", 8, 128}, kConv4sR8N2 = {"r8n2", 16, 128};

bool arm_gpu() {
  static const bool arm = vkt_vendor_id() == 0x13B5;
  return arm;
}

// The tile, from docs/log/M47's sweeps: on Mali 4 rows x 8 columns per lane
// (more rows spill float32 accumulators there), elsewhere (Intel) 8 x 4,
// and 8 x 8 for float16 arithmetic.
const Conv4sTile& conv4s_tile(bool half_math) {
  if (arm_gpu()) return kConv4sR4N2;
  return half_math ? kConv4sR8N2 : kConv4sR8N1;
}

// Whether a stride-1 convolution of type t with kw inner taps runs as
// shaders/conv4s.slang: where the device runs it (16-lane subgroups with
// shuffles: Intel's and Mali's Valhall GPUs) and it was measured faster than
// im2col columns and a matrix product (docs/log/M47): on Intel always; on
// Mali for float16 and bfloat16 (float32 stays with gemm4s's product, as
// fast or faster there). TORCH_VK_CONV=im2col never, =direct wherever it
// runs.
bool use_conv4s(at::ScalarType t) {
  static const std::string choice = [] {
    const char* env = std::getenv("TORCH_VK_CONV");
    return std::string(env != nullptr ? env : "");
  }();
  static const bool runs = choice != "im2col" && vkt_kernel_subgroup_size("conv4s_f32_d1_r8n1") == 16;
  static const bool f16 = runs && has_feature(kFeatureFloat16) && has_feature(kFeatureInt16);
  static const bool bf16 = runs && has_feature(kFeatureInt16);
  bool can = t == at::kFloat ? runs : t == at::kHalf ? f16 : t == at::kBFloat16 ? bf16 : false;
  if (!can || choice == "direct" || !arm_gpu()) return can;
  return t != at::kFloat;
}

// A stride-1 convolution by shaders/conv4s.slang (where use_conv4s), reading
// the zero-padded input in place: no im2col columns. The input is copied
// once into a buffer padded as the kernel needs (rows of Wp, a multiple of
// 4 values, for 2-D and 3-D; each channel's plane a multiple of 4; slack at
// the end for the columns and taps read past the last valid one), the
// weight into rows of KW taps padded to a multiple of 4. Returns an
// undefined tensor where it does not apply.
at::Tensor conv_direct(const at::Tensor& x_in, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                       at::IntArrayRef stride, const std::vector<std::pair<int64_t, int64_t>>& pad_in,
                       at::IntArrayRef dilation, int64_t groups, at::IntArrayRef out) {
  at::ScalarType t = x_in.scalar_type();
  int64_t s = x_in.dim() - 2, N = x_in.size(0), C = x_in.size(1), Co = w.size(0);
  bool half_math = t == at::kHalf && fp16_arithmetic();
  if (!use_conv4s(t) || w.scalar_type() != t ||
      (bias.has_value() && bias->defined() && bias->scalar_type() != t)) {
    return {};
  }
  for (int64_t d = 0; d < s; ++d) {
    if (stride[d] != 1) return {};
  }
  int64_t KW = w.size(-1), DW = dilation[s - 1];
  if (DW != 1 && DW != 2 && DW != 3 && DW != 5) return {};  // (the kernel's variants: axis D)
  // (fewer than 3 inner taps, padded to 4, waste half the work or more:
  // M46's transposed-convolution phases, with 2 taps, ran 1.5x slower)
  if (KW < 3) return {};
  int64_t Cg = C / groups, Cog = Co / groups, taps = 1;
  for (int64_t d = 0; d < s; ++d) taps *= w.size(2 + d);
  if (Cg == 1 && taps <= 32) return {};  // (depthwise: multiply-adds per tap, in conv_forward)
  // Crop for negative padding; then the padded sizes.
  at::Tensor x = x_in;
  std::vector<std::pair<int64_t, int64_t>> pad = pad_in;
  for (int64_t d = 0; d < s; ++d) {
    if (pad[d].first < 0) {
      x = x.narrow(2 + d, -pad[d].first, x.size(2 + d) + pad[d].first);
      pad[d].first = 0;
    }
    if (pad[d].second < 0) {
      x = x.narrow(2 + d, 0, x.size(2 + d) + pad[d].second);
      pad[d].second = 0;
    }
  }
  std::vector<int64_t> padded(s);
  for (int64_t d = 0; d < s; ++d) padded[d] = x.size(2 + d) + pad[d].first + pad[d].second;
  int64_t Wo = out[s - 1], Wp = s == 1 ? padded[0] : ceil_div(padded[s - 1], 4) * 4;
  int64_t Hp = s >= 2 ? padded[s - 2] : 1, Ho = s >= 2 ? out[s - 2] : 1, Dp = s == 3 ? padded[0] : 1,
          Do = s == 3 ? out[0] : 1;
  int64_t plane = ceil_div(Dp * Hp * Wp, 4) * 4;
  // (virtual columns: up to the last output's, rounded up to whole workgroups of columns)
  int64_t nv = ((Do - 1) * Hp + (Ho - 1)) * Wp + Wo;
  if (s >= 2 && Wo < 8) return {};  // (mostly wrapped columns: the columns' product does better)
  // (few output positions, many channels: the product of columns did better
  // on both GPUs measured, e.g. 1024 channels over 242 frames)
  if (N * nv < 2048) return {};
  int64_t KH = s >= 2 ? w.size(-2) : 1, KD = s == 3 ? w.size(2) : 1, KO = KD * KH;
  int64_t off_h = s >= 2 ? dilation[s - 2] * Wp : 0, off_d = s == 3 ? dilation[0] * Hp * Wp : 0;
  const Conv4sTile& tile = conv4s_tile(half_math);
  int64_t cols = tile.cols;
  int64_t reach = ceil_div(nv, cols) * cols + (KD - 1) * off_d + (KH - 1) * off_h + (KW + 8) * DW + 32;
  int64_t slack = std::max<int64_t>(0, reach - plane);
  int64_t total = N * C * plane + slack;
  if (total >= (int64_t(1) << 31) || N * Co * (Do * Ho * Wo) >= (int64_t(1) << 31) || N * groups > 65535 ||
      ceil_div(nv, cols) > 65535 || ceil_div(Cog, tile.rows) > 65535) {
    return {};
  }
  at::Tensor xbuf = at::zeros({total}, x.options());
  {
    at::Tensor view = xbuf.narrow(0, 0, N * C * plane).view({N, C, plane}).narrow(2, 0, Dp * Hp * Wp);
    std::vector<int64_t> grid = {N, C};
    if (s == 3) grid.push_back(Dp);
    if (s >= 2) grid.push_back(Hp);
    grid.push_back(Wp);
    view = view.reshape(grid);
    for (int64_t d = 0; d < s; ++d) view = view.narrow(2 + d, pad[d].first, x.size(2 + d));
    view.copy_(x);
  }
  int64_t KW4 = ceil_div(KW, 4) * 4;
  at::Tensor a = at::zeros({Co, Cg * KO, KW4}, w.options());
  a.narrow(2, 0, KW).copy_(w.reshape({Co, Cg * KO, KW}));
  std::vector<int64_t> out_shape = {N, Co};
  out_shape.insert(out_shape.end(), out.begin(), out.end());
  at::Tensor y = at::empty(out_shape, x.options());
  bool use_bias = bias.has_value() && bias->defined();
  at::Tensor b = use_bias ? bias->contiguous() : at::Tensor();
  Conv4sParams p{};
  p.x = gpu_address(xbuf);
  p.w = gpu_address(a);
  p.y = gpu_address_out(y);
  p.bias = use_bias ? gpu_address(b) : 0;
  p.cog = static_cast<uint32_t>(Cog);
  p.cg = static_cast<uint32_t>(Cg);
  p.ko = static_cast<uint32_t>(KO);
  p.kh = static_cast<uint32_t>(KH);
  p.off_d = static_cast<int32_t>(off_d);
  p.off_h = static_cast<int32_t>(off_h);
  p.nv = static_cast<uint32_t>(nv);
  p.plane = static_cast<uint32_t>(plane);
  p.x_sn = static_cast<int32_t>(C * plane);
  p.w_sm = static_cast<int32_t>(Cg * KO * KW4);
  p.y_sc = static_cast<int32_t>(Do * Ho * Wo);
  p.y_sn = static_cast<int32_t>(Co * Do * Ho * Wo);
  p.wp = static_cast<uint32_t>(s == 1 ? nv : Wp);
  p.wo = static_cast<uint32_t>(s == 1 ? nv : Wo);
  p.hp = static_cast<uint32_t>(s == 3 ? Hp : nv);
  p.ho = static_cast<uint32_t>(s == 3 ? Ho : nv);
  p.groups = static_cast<uint32_t>(groups);
  p.kg = static_cast<uint32_t>(KW4 / 4);
  dispatch(std::string("conv4s_") + (half_math ? "f16h" : storage_label(t)) + "_d" + std::to_string(DW) + "_" +
               tile.label,
           {use_bias ? 1u : 0u}, p,
           static_cast<uint32_t>(ceil_div(nv, cols)), static_cast<uint32_t>(ceil_div(Cog, tile.rows)),
           static_cast<uint32_t>(N * groups));
  return y;
}

// Plain convolution: x [N, C, *S], w [Co, C/G, *K], symmetric or asymmetric
// zero padding pad[d] = (lo, hi).
at::Tensor conv_forward(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                        at::IntArrayRef stride, const std::vector<std::pair<int64_t, int64_t>>& pad,
                        at::IntArrayRef dilation, int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), C = x.size(1), Co = w.size(0);
  std::vector<int64_t> k(w.sizes().begin() + 2, w.sizes().end()), out;
  for (int64_t d = 0; d < s; ++d) {
    out.push_back(conv_out_size(x.size(2 + d), k[d], stride[d], pad[d].first, pad[d].second, dilation[d]));
  }
  std::vector<int64_t> out_shape = {N, Co};
  out_shape.insert(out_shape.end(), out.begin(), out.end());
  int64_t L = 1;
  for (int64_t o : out) L *= o;
  if (N == 0 || Co == 0 || L == 0) {
    return at::zeros(out_shape, x.options());
  }
  if (at::Tensor y = conv_direct(x, w, bias, stride, pad, dilation, groups, out); y.defined()) {
    return y;
  }
  if (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) {
    // (16-bit storage the direct kernel does not take: in float32, rounded once)
    std::optional<at::Tensor> bf;
    if (bias.has_value() && bias->defined()) bf = bias->to(at::kFloat);
    return conv_forward(x.to(at::kFloat), w.to(at::kFloat), bf, stride, pad, dilation, groups).to(x.scalar_type());
  }
  at::Tensor xp = pad_spatial(x, pad);
  int64_t Kp = 1;
  for (int64_t v : k) Kp *= v;
  int64_t Cg = C / groups, Cog = Co / groups;
  if (Cg == 1 && Kp <= 32) {
    // Depthwise (one input channel per group): per tap, one multiply-add of
    // the input's strided view into the result, [N, G, Cog, *O] (a product
    // per group would have one row of k = Kp values: tiny matrices).
    std::vector<int64_t> acc_shape = {N, groups, Cog}, xv_shape = {N, groups, 1}, w_shape = {1, groups, Cog};
    std::vector<int64_t> xv_strides = {xp.stride(0), xp.stride(1), 0};
    for (int64_t d = 0; d < s; ++d) {
      acc_shape.push_back(out[d]);
      xv_shape.push_back(out[d]);
      xv_strides.push_back(xp.stride(2 + d) * stride[d]);
      w_shape.push_back(1);
    }
    at::Tensor wt = w.reshape({Co, Kp});
    at::Tensor acc = at::empty(acc_shape, x.options());
    bool first = true;
    if (bias.has_value() && bias->defined()) {
      acc.copy_(bias->view(w_shape).expand(acc_shape));
      first = false;
    }
    for (int64_t t = 0; t < Kp; ++t) {
      int64_t offset = xp.storage_offset(), rest = t;
      for (int64_t d = s - 1; d >= 0; --d) {
        offset += (rest % k[d]) * dilation[d] * xp.stride(2 + d);
        rest /= k[d];
      }
      at::Tensor xv = xp.as_strided(xv_shape, xv_strides, offset);
      at::Tensor wv = wt.select(1, t).view(w_shape);
      if (first) {
        at::mul_out(acc, xv, wv);
        first = false;
      } else {
        acc.addcmul_(xv, wv);
      }
    }
    return acc.view(out_shape);
  }
  if (groups == 1 || N == 1) {
    // out[n, g] = W_g [Cog, Cg*K] @ cols[n, g] [Cg*K, L]: the columns are
    // rows along the output positions (for stride 1, runs of the input),
    // padded to a multiple of 4 positions (gemm4s.slang reads B's rows in
    // groups of 4 and stores only the first L columns), and the bias is the
    // product's beta * Cin (one rounding, as y + bias): the result comes out
    // as [N, Co, L], without the transposing copy and the separate pass that
    // the [N*L, Co] product needs.
    int64_t L4 = (L + 3) / 4 * 4;
    std::vector<int64_t> sizes = {N, C}, strides = {xp.stride(0), xp.stride(1)};
    for (int64_t d = 0; d < s; ++d) {
      sizes.push_back(k[d]);
      strides.push_back(xp.stride(2 + d) * dilation[d]);
    }
    for (int64_t d = 0; d < s; ++d) {
      sizes.push_back(out[d]);
      strides.push_back(xp.stride(2 + d) * stride[d]);
    }
    at::Tensor rows = at::empty({N, C * Kp, L4}, x.options());
    rows.narrow(2, 0, L).view(sizes).copy_(xp.as_strided(sizes, strides, xp.storage_offset()));
    int64_t batch = groups == 1 ? N : groups;
    at::Tensor wg = w.reshape({groups, Cog, Cg * Kp}).expand({batch, Cog, Cg * Kp});
    at::Tensor cg = rows.view({batch, Cg * Kp, L4}).narrow(2, 0, L);
    at::Tensor y;
    if (bias.has_value() && bias->defined()) {
      at::Tensor b = bias->view({groups == 1 ? 1 : groups, Cog, 1}).expand({batch, Cog, L});
      y = at::baddbmm(b, wg, cg);
    } else {
      y = at::bmm(wg, cg);
    }
    return y.view(out_shape);
  }
  at::Tensor cols = im2col_view(xp, k, stride, dilation, out);       // [N, L, C*K] (c-major, then k)
  cols = cols.view({N, L, groups, Cg * Kp}).permute({2, 0, 1, 3});   // [G, N, L, Cg*K]
  cols = cols.reshape({groups, N * L, Cg * Kp});
  at::Tensor wg = w.reshape({groups, Cog, Cg * Kp}).transpose(1, 2);  // [G, Cg*K, Cog]
  at::Tensor y = at::bmm(cols, wg);                                   // [G, N*L, Cog]
  y = y.view({groups, N, L, Cog}).permute({1, 0, 3, 2}).reshape({N, Co, L});
  if (bias.has_value() && bias->defined()) {
    y = y + bias->view({1, Co, 1});
  }
  return y.reshape(out_shape);
}

// Transposed convolution with a stride, by phases (sub-pixel convolution):
// x [N, Ci, *S], w [Ci, Co/G, *K]. Along a dim with stride s > 1, input i
// and tap k reach the uncropped output position o' = o + p = i s + k d.
// Splitting o' = q s + r, the outputs of phase r (0 <= r < s) take the taps
// with k d = r + t s, from input i = q - t: an ordinary stride-1
// convolution of x with T = floor(d (K - 1) / s) + 1 taps (the phase's taps
// at t, zeros where k d = r + t s has no k), the same input positions for
// every phase. So the s phases are one convolution with s Co output
// channels, whose result is interleaved into the output: about K / T times
// less work than convolving the zero-inserted input (s times for d = 1 and
// K a multiple of s; docs/log/M46). Dims with stride 1 stay ordinary
// (flipped kernel, dilation d). Returns an undefined tensor where this does
// not save work (no stride, or a dilation large enough that T >= K) or the
// input is empty.
at::Tensor conv_transpose_phases(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                                 at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding,
                                 at::IntArrayRef dilation, int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), Ci = x.size(1), Cog = w.size(1), Co = Cog * groups, Cig = Ci / groups;
  std::vector<int64_t> T(s), outlen(s), q0(s), qn(s);
  int64_t taps = 1, phase_taps = 1, P = 1;
  for (int64_t d = 0; d < s; ++d) {
    int64_t K = w.size(2 + d), S = x.size(2 + d), full = dilation[d] * (K - 1);
    outlen[d] = (S - 1) * stride[d] - 2 * padding[d] + full + output_padding[d] + 1;
    if (S == 0 || outlen[d] <= 0) return {};
    taps *= K;
    if (stride[d] == 1) {
      T[d] = K;
      phase_taps *= K;
      continue;
    }
    T[d] = full / stride[d] + 1;
    phase_taps *= T[d];
    P *= stride[d];
    // The phases' outputs q0 <= q < q0 + qn cover o' in [p, p + outlen).
    q0[d] = padding[d] / stride[d];
    qn[d] = ceil_div(padding[d] + outlen[d], stride[d]) - q0[d];
  }
  if (P == 1 || phase_taps >= taps || N == 0 || Co == 0) return {};

  // The phases' weight [G Cog P, Cig, *T]: per strided dim, the kernel
  // dilated (zeros between taps), zero-padded to T s, as [T, s] (tap d k = r
  // + t s), t reversed (the convolution's tap T - 1 - t reads input q - t);
  // other dims' kernels reversed.
  std::vector<int64_t> wshape = {Ci, Cog};
  for (int64_t d = 0; d < s; ++d) wshape.push_back(stride[d] == 1 ? w.size(2 + d) : T[d] * stride[d]);
  at::Tensor wd = w;
  bool dilate = false;
  for (int64_t d = 0; d < s; ++d) dilate = dilate || (stride[d] > 1 && wshape[2 + d] != w.size(2 + d));
  if (dilate) {
    wd = at::zeros(wshape, w.options());
    at::Tensor view = wd;
    for (int64_t d = 0; d < s; ++d) {
      if (stride[d] > 1) view = view.slice(2 + d, 0, dilation[d] * (w.size(2 + d) - 1) + 1, dilation[d]);
    }
    view.copy_(w);
  }
  // [G, Cig, Cog, (T, s) or K per dim] -> [G, Cog, s per strided dim, Cig, T or K per dim]
  std::vector<int64_t> split = {groups, Cig, Cog};
  std::vector<int64_t> phase_dims, tap_dims;
  for (int64_t d = 0; d < s; ++d) {
    if (stride[d] == 1) {
      tap_dims.push_back(static_cast<int64_t>(split.size()));
      split.push_back(w.size(2 + d));
    } else {
      tap_dims.push_back(static_cast<int64_t>(split.size()));
      split.push_back(T[d]);
      phase_dims.push_back(static_cast<int64_t>(split.size()));
      split.push_back(stride[d]);
    }
  }
  std::vector<int64_t> order = {0, 2};
  order.insert(order.end(), phase_dims.begin(), phase_dims.end());
  order.push_back(1);
  order.insert(order.end(), tap_dims.begin(), tap_dims.end());
  std::vector<int64_t> wpshape = {Co * P, Cig};
  for (int64_t d = 0; d < s; ++d) wpshape.push_back(T[d]);
  at::Tensor wp = reversed(wd.reshape(split), tap_dims).permute(order).reshape(wpshape);
  std::optional<at::Tensor> bp;
  if (bias.has_value() && bias->defined()) {
    bp = bias->view({Co, 1}).expand({Co, P}).reshape({Co * P});
  }

  // The phases as one convolution: output q0 + j (j < qn) along a strided
  // dim reads inputs q - t, t < T, so the input is padded by T - 1 - q0
  // before (negative: cropped) and q0 + qn - S after; other dims as for a
  // stride-1 transposed convolution.
  std::vector<std::pair<int64_t, int64_t>> pad;
  std::vector<int64_t> ones(s, 1), dil(s, 1);
  for (int64_t d = 0; d < s; ++d) {
    if (stride[d] == 1) {
      int64_t full = dilation[d] * (w.size(2 + d) - 1);
      pad.push_back({full - padding[d], full - padding[d] + output_padding[d]});
      dil[d] = dilation[d];
    } else {
      pad.push_back({T[d] - 1 - q0[d], q0[d] + qn[d] - x.size(2 + d)});
    }
  }
  at::Tensor y = conv_forward(x, wp, bp, ones, pad, dil, groups);  // [N, Co P, *Q]

  // Interleave: out[n, c, j s + r - e] = y[n, c, r, j] per strided dim, e = p - q0 s.
  std::vector<int64_t> yshape = {N, Co}, zshape = {N, Co}, zorder = {0, 1}, zlen = {N, Co};
  int64_t ph = 0;
  for (int64_t d = 0; d < s; ++d) {
    if (stride[d] > 1) yshape.push_back(stride[d]);
  }
  for (int64_t d = 0; d < s; ++d) {
    yshape.push_back(stride[d] == 1 ? outlen[d] : qn[d]);
  }
  int64_t nphase = static_cast<int64_t>(phase_dims.size());
  for (int64_t d = 0; d < s; ++d) {
    zorder.push_back(2 + nphase + d);
    zshape.push_back(stride[d] == 1 ? outlen[d] : qn[d]);
    if (stride[d] > 1) {
      zorder.push_back(2 + ph++);
      zshape.push_back(stride[d]);
    }
    zlen.push_back(stride[d] == 1 ? outlen[d] : qn[d] * stride[d]);
  }
  at::Tensor z = y.view(yshape).permute(zorder).contiguous().view(zlen);
  for (int64_t d = 0; d < s; ++d) {
    if (stride[d] > 1) z = z.narrow(2 + d, padding[d] - q0[d] * stride[d], outlen[d]);
  }
  return z.contiguous();
}

// Transposed convolution: x [N, Ci, *S], w [Ci, Co/G, *K].
at::Tensor conv_transpose_forward(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                                  at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding,
                                  at::IntArrayRef dilation, int64_t groups) {
  if (at::Tensor y = conv_transpose_phases(x, w, bias, stride, padding, output_padding, dilation, groups); y.defined()) {
    return y;
  }
  int64_t s = x.dim() - 2, N = x.size(0), Ci = x.size(1), Cog = w.size(1), Co = Cog * groups;
  // Insert stride - 1 zeros between input elements.
  std::vector<int64_t> dil_shape = {N, Ci};
  for (int64_t d = 0; d < s; ++d) {
    dil_shape.push_back(x.size(2 + d) == 0 ? 0 : (x.size(2 + d) - 1) * stride[d] + 1);
  }
  at::Tensor xd = x;
  bool strided = false;
  for (int64_t d = 0; d < s; ++d) strided = strided || stride[d] > 1;
  if (strided) {
    xd = at::zeros(dil_shape, x.options());
    at::Tensor view = xd;
    for (int64_t d = 0; d < s; ++d) {
      view = view.slice(2 + d, 0, dil_shape[2 + d], stride[d]);
    }
    view.copy_(x);
  }
  // The equivalent convolution: padding d (k - 1) - p (output_padding more at the end), flipped kernel.
  std::vector<std::pair<int64_t, int64_t>> pad;
  std::vector<int64_t> flip_dims, ones(s, 1);
  for (int64_t d = 0; d < s; ++d) {
    int64_t full = dilation[d] * (w.size(2 + d) - 1);
    pad.push_back({full - padding[d], full - padding[d] + output_padding[d]});
    flip_dims.push_back(2 + d);
  }
  // [Ci, Co/G, *K] -> per group [Co/G, Ci/G, *K], flipped.
  int64_t Cig = Ci / groups;
  std::vector<int64_t> gshape = {groups, Cig, Cog};
  gshape.insert(gshape.end(), w.sizes().begin() + 2, w.sizes().end());
  at::Tensor wt = w.reshape(gshape).transpose(1, 2);
  std::vector<int64_t> wshape = {Co, Cig};
  wshape.insert(wshape.end(), w.sizes().begin() + 2, w.sizes().end());
  wt = reversed(wt.reshape(wshape), flip_dims);
  return conv_forward(xd, wt, bias, ones, pad, dilation, groups);
}

std::vector<std::pair<int64_t, int64_t>> symmetric(at::IntArrayRef padding) {
  std::vector<std::pair<int64_t, int64_t>> pad;
  for (int64_t p : padding) pad.push_back({p, p});
  return pad;
}

// Expands 1-element parameter lists to one per spatial dim.
std::vector<int64_t> per_dim(at::IntArrayRef v, int64_t s) {
  if (static_cast<int64_t>(v.size()) == s) return v.vec();
  TORCH_CHECK(v.size() == 1, "torch_vulkan: convolution parameter has ", v.size(), " values for ", s, " dims");
  return std::vector<int64_t>(s, v[0]);
}

// Whether vk computes the convolution; `half`: float16 and bfloat16 too
// (the forward: shaders/conv4s.slang in 16-bit storage where it applies,
// else in float32 and rounded once; other ops reach these dtypes through
// half_via_float).
bool conv_ok(const at::Tensor& x, const at::Tensor& w, bool half = false) {
  at::ScalarType t = x.scalar_type();
  bool t16 = half && (t == at::kHalf || t == at::kBFloat16) && storage_label(t) != nullptr;
  return is_vk(x) && is_vk(w) && x.dim() >= 3 && x.dim() <= 5 && w.dim() == x.dim() &&
         (t == at::kFloat || t == at::kInt || t == at::kLong || (t == at::kDouble && has_feature(kFeatureFloat64)) ||
          t16) &&
         w.scalar_type() == t;
}

at::Tensor convolution_compute(const at::Tensor& input, const at::Tensor& weight,
                               const std::optional<at::Tensor>& bias, at::IntArrayRef stride_,
                               at::IntArrayRef padding_, at::IntArrayRef dilation_, bool transposed,
                               at::IntArrayRef output_padding_, int64_t groups) {
  if (!conv_ok(input, weight, true) || (bias.has_value() && bias->defined() && bias->scalar_type() != input.scalar_type())) {
    // (convolution_overrideable has no CPU kernel; aten::convolution does.)
    return on_cpu("aten::convolution", "", input, weight, bias, stride_, padding_, dilation_, transposed,
                  output_padding_, groups)[0]
        .toTensor();
  }
  int64_t s = input.dim() - 2;
  auto stride = per_dim(stride_, s), padding = per_dim(padding_, s), dilation = per_dim(dilation_, s);
  auto output_padding = per_dim(output_padding_, s);
  if (transposed) {
    return conv_transpose_forward(input, weight, bias, stride, padding, output_padding, dilation, groups);
  }
  return conv_forward(input, weight, bias, stride, symmetric(padding), dilation, groups);
}

// The registered kernel: the result laid out as the meta kernel says (e.g.
// channels-last for channels-last inputs).
at::Tensor convolution(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                       at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                       at::IntArrayRef output_padding, int64_t groups) {
  at::NoTF32Guard exact;  // (its products in float32 at any torch.set_float32_matmul_precision: linalg.cpp)
  at::Tensor result =
      convolution_compute(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  // (the CPU's rule, mkldnn's and the slow path's: channels-last output if
  // the input or the weight is channels-last; the meta kernel says
  // contiguous for meta tensors)
  for (auto fmt : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::ChannelsLast3d}) {
    int64_t dims = fmt == at::MemoryFormat::ChannelsLast ? 4 : 5;
    if (input.dim() == dims && result.dim() == dims &&
        (input.suggest_memory_format() == fmt || weight.suggest_memory_format() == fmt)) {
      return result.contiguous(fmt);
    }
  }
  at::Tensor layout = at::convolution(meta_of(input), meta_of(weight), meta_of(bias), stride, padding, dilation,
                                      transposed, output_padding, groups);
  return laid_out_like(result, layout);
}

// grad_weight of a plain convolution: [Co, C/G, *K] = sum over N, L of
// grad_output x columns.
at::Tensor conv_weight_grad(const at::Tensor& x, const at::Tensor& g, at::IntArrayRef wsize, at::IntArrayRef stride,
                            const std::vector<std::pair<int64_t, int64_t>>& pad, at::IntArrayRef dilation,
                            int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), C = x.size(1), Co = g.size(1);
  std::vector<int64_t> k(wsize.begin() + 2, wsize.end()), out(g.sizes().begin() + 2, g.sizes().end());
  int64_t L = 1, Kp = 1;
  for (int64_t o : out) L *= o;
  for (int64_t v : k) Kp *= v;
  int64_t Cg = C / groups, Cog = Co / groups;
  if (N == 0 || L == 0) {
    return at::zeros(wsize, x.options());
  }
  at::Tensor cols = im2col_view(pad_spatial(x, pad), k, stride, dilation, out);      // [N, L, C*K]
  cols = cols.reshape({N * L, groups, Cg * Kp}).permute({1, 0, 2});                  // [G, N*L, Cg*K]
  at::Tensor gg = g.reshape({N, groups, Cog, L}).permute({1, 2, 0, 3}).reshape({groups, Cog, N * L});
  return at::bmm(gg, cols).reshape(wsize);                                            // [G, Cog, Cg*K]
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward_compute(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride_,
    at::IntArrayRef padding_, at::IntArrayRef dilation_, bool transposed, at::IntArrayRef output_padding_,
    int64_t groups, std::array<bool, 3> mask) {
  if (!conv_ok(input, weight) || !is_vk(grad_output)) {
    std::vector<int64_t> bias_sizes = {transposed ? weight.size(1) * groups : weight.size(0)};
    auto stack = on_cpu("aten::convolution_backward", "", grad_output, input, weight, bias_sizes, stride_, padding_,
                        dilation_, transposed, output_padding_, groups, mask);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  int64_t s = input.dim() - 2;
  auto stride = per_dim(stride_, s), padding = per_dim(padding_, s), dilation = per_dim(dilation_, s);
  auto output_padding = per_dim(output_padding_, s);
  at::Tensor g = grad_output.contiguous();
  at::Tensor grad_input, grad_weight, grad_bias;
  if (!transposed) {
    if (mask[0]) {
      // The adjoint of a strided convolution is the transposed convolution;
      // output_padding recovers input sizes the stride rounded away.
      std::vector<int64_t> op;
      for (int64_t d = 0; d < s; ++d) {
        int64_t full = (g.size(2 + d) - 1) * stride[d] - 2 * padding[d] + dilation[d] * (weight.size(2 + d) - 1) + 1;
        op.push_back(input.size(2 + d) - full);
      }
      grad_input = conv_transpose_forward(g, weight, std::nullopt, stride, padding, op, dilation, groups);
    }
    if (mask[1]) {
      grad_weight = conv_weight_grad(input, g, weight.sizes(), stride, symmetric(padding), dilation, groups);
    }
  } else {
    if (mask[0]) {
      // The adjoint of a transposed convolution is the plain convolution;
      // output_padding can make that longer than the input (positions past
      // the input's end, which no input element reached): cropped.
      grad_input = conv_forward(g, weight, std::nullopt, stride, symmetric(padding), dilation, groups);
      for (int64_t d = 0; d < s; ++d) {
        grad_input = grad_input.narrow(2 + d, 0, input.size(2 + d));
      }
      grad_input = grad_input.contiguous();
    }
    if (mask[1]) {
      // y = A_W^T x for the plain convolution A_W using W [Ci, Co/G, *K] as
      // is (Co channels in, Ci out), so dL/dW is A_W's weight gradient with
      // grad_output as its input and `input` as its output gradient.
      grad_weight = conv_weight_grad(g, input, weight.sizes(), stride, symmetric(padding), dilation, groups);
    }
  }
  if (mask[2]) {
    std::vector<int64_t> dims = {0};
    for (int64_t d = 0; d < s; ++d) dims.push_back(2 + d);
    grad_bias = g.sum(dims);
  }
  return {grad_input, grad_weight, grad_bias};
}

// The registered kernel: gradients laid out as the meta kernel says (the
// input's and weight's memory format).
std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride,
    at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> mask) {
  at::NoTF32Guard exact;  // (its products in float32 at any torch.set_float32_matmul_precision: linalg.cpp)
  auto [gi, gw, gb] = convolution_backward_compute(grad_output, input, weight, stride, padding, dilation, transposed,
                                                   output_padding, groups, mask);
  std::vector<int64_t> bias_sizes = {transposed ? weight.size(1) * groups : weight.size(0)};
  auto [mi, mw, mb] = at::convolution_backward(meta_of(grad_output), meta_of(input), meta_of(weight), bias_sizes,
                                               stride, padding, dilation, transposed, output_padding, groups, mask);
  return {gi.defined() ? laid_out_like(gi, mi) : gi, gw.defined() ? laid_out_like(gw, mw) : gw, gb};
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_overrideable", &vk::convolution);
  m.impl("convolution_backward_overrideable", &vk::convolution_backward);
}

extern "C" void vkt_set_fp16_arithmetic(int enabled) { vk::fp16_arithmetic_flag().store(enabled != 0); }
extern "C" int vkt_fp16_arithmetic() { return vk::fp16_arithmetic() ? 1 : 0; }
