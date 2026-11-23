// Scaled dot-product attention for device "vk" (docs/log/M52): a fused
// kernel without the attention matrix (shaders/attention.slang, the tiling
// of FlashAttention 2 in float32), forward and backward, chosen by
// _fused_sdp_choice through PyTorch's "overrideable" backend, the slot
// PyTorch keeps for PrivateUse1 devices:
//
//   scaled_dot_product_attention -> _fused_sdp_choice_stub (this file's
//   fused_sdp_choice) -> SDPBackend::overrideable ->
//   _scaled_dot_product_fused_attention_overrideable (forward; its
//   backward is PyTorch's autograd formula, which calls
//   _scaled_dot_product_fused_attention_overrideable_backward).
//
// Everything else (float64, nested tensors, a mask that needs a gradient,
// head dims past 256, ...) takes the math path, as it always did before.
// torch.nn.attention.sdpa_kernel picks between the two (SDPBackend.MATH,
// SDPBackend.OVERRIDEABLE), and TORCH_VK_SDPA=math or flash sets the
// preference when both are enabled.
//
// The forward's output is [B, L, H, EV] in memory (viewed as [B, H, L, EV]),
// as CUDA's flash kernel's, so the usual out.transpose(1, 2).reshape(B, L,
// H EV) is free; logsumexp is [B, H, L] in float32. Dropout draws its mask
// from Philox (seed, offset of the default generator, one offset per call),
// returned as philox_seed / philox_offset (CPU int64 scalars, as CUDA's) for
// the backward to draw the same mask again.

#include "vk.h"

#include <ATen/Context.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/transformers/attention.h>
#include <torch/library.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace vk {
namespace {

constexpr int64_t kBlock = 64;  // rows per workgroup (attention.slang)
constexpr int64_t kMax32 = (int64_t(1) << 31) - 1;

// shaders/attention.slang's Params.
struct AttnParams {
  uint64_t q, k, v, o, st, mask, w48, w56;
  uint32_t qs[4], ks[4], vs[4], ms[4];
};
static_assert(sizeof(AttnParams) == 128, "must match shaders/attention.slang");

// shaders/attention_aux.slang's Params.
struct AttnAuxParams {
  uint64_t o, go, lse, out;
  uint32_t key[2], ctr, pbits, n, gstride, H, L, W, pad;
};
static_assert(sizeof(AttnAuxParams) == 72, "must match shaders/attention_aux.slang");

// attention.slang's variant (axis T) for dtype t on this device, or null.
const char* attention_label(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kHalf: return has_feature(kFeatureFloat16) && has_feature(kFeatureInt16) ? "f16" : nullptr;
    case at::kBFloat16: return has_feature(kFeatureInt16) ? "bf16" : nullptr;
    default: return nullptr;
  }
}

// The kernel's head-dim class (axis D) for head dims e (q, k) and ev (v),
// 0 past 256.
int64_t head_class(int64_t e, int64_t ev) {
  int64_t m = std::max(e, ev);
  return m <= 64 ? 64 : m <= 128 ? 128 : m <= 256 ? 256 : 0;
}

// The largest element offset in t from its first element.
int64_t span(const at::Tensor& t) {
  int64_t s = 0;
  for (int64_t d = 0; d < t.dim(); ++d) {
    if (t.size(d) > 1) s += (t.size(d) - 1) * std::abs(t.stride(d));
  }
  return s;
}

// Why the kernel cannot take these arguments, or "" if it can (sizes in
// the kernel's 32-bit offsets and workgroup counts). With `autograd`, also
// what it cannot differentiate as the math path does (a mask that needs a
// gradient, forward-mode gradients).
std::string why_not(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    const std::optional<at::Tensor>& mask, double dropout_p, bool is_causal, bool enable_gqa,
                    bool autograd) {
  if (!has_feature(kFeatureShared32K)) return "the device has less than 32 KiB of shared memory";
  if (q.is_nested() || k.is_nested() || v.is_nested()) return "nested tensors";
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) return "query, key and value must be 4-d";
  at::ScalarType t = q.scalar_type();
  if (k.scalar_type() != t || v.scalar_type() != t) return "query, key and value differ in dtype";
  if (attention_label(t) == nullptr) return "dtype not float32, float16 or bfloat16 (on this device)";
  int64_t B = q.size(0), H = q.size(1), L = q.size(2), E = q.size(3);
  int64_t Hk = k.size(1), S = k.size(2), EV = v.size(3);
  if (k.size(0) != B || v.size(0) != B || v.size(1) != Hk || v.size(2) != S || k.size(3) != E) {
    return "query, key and value sizes need broadcasting";
  }
  if (Hk == 0 || H % Hk != 0 || (H != Hk && Hk != 1 && !enable_gqa)) return "key heads do not divide query heads";
  if (E == 0 || head_class(E, EV) == 0) return "head dim 0 or more than 256";
  if (B > kMaxGroups || H > kMaxGroups || ceil_div(L, kBlock) > kMaxGroups || ceil_div(S, kBlock) > kMaxGroups) {
    return "too many batches, heads or blocks for one launch";
  }
  if (span(q) > kMax32 || span(k) > kMax32 || span(v) > kMax32 || B * L * H * std::max(E, EV) > kMax32 ||
      B * S * H * std::max(E, EV) > kMax32 || 4 + 2 * B * H * L > kMax32) {
    return "more than 2^31 elements";
  }
  if (dropout_p < 0 || dropout_p > 1) return "dropout probability outside [0, 1]";
  if (mask.has_value() && mask->defined()) {
    const at::Tensor& m = *mask;
    if (is_causal) return "is_causal with a mask";
    if (m.scalar_type() != at::kBool && m.scalar_type() != t) return "mask dtype";
    if (m.dim() > 4 || !at::is_expandable_to(m.sizes(), {B, H, L, S})) return "mask shape";
    if (span(m) > kMax32) return "more than 2^31 mask elements";
    if (autograd && m.requires_grad()) return "the mask requires a gradient";
  }
  if (autograd) {
    for (const at::Tensor* x : {&q, &k, &v}) {
      if (x->_fw_grad(/*level=*/0).defined()) return "forward-mode gradients";
    }
  }
  return "";
}

// TORCH_VK_SDPA: 1 math, 2 flash, 0 unset (choose).
int forced_route() {
  static const int route = [] {
    const char* env = std::getenv("TORCH_VK_SDPA");
    if (env == nullptr || *env == '\0') return 0;
    if (std::strcmp(env, "math") == 0) return 1;
    if (std::strcmp(env, "flash") == 0) return 2;
    TORCH_CHECK(false, "TORCH_VK_SDPA must be math or flash, not ", env);
  }();
  return route;
}

// Which attention backend to use (a DispatchStub, also aten::_fused_sdp_choice).
int64_t fused_sdp_choice(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                         const std::optional<at::Tensor>& mask, double dropout_p, bool is_causal,
                         std::optional<double> scale, bool enable_gqa) {
  auto& ctx = at::globalContext();
  bool math = ctx.userEnabledMathSDP();
  bool fused_enabled = ctx.userEnabledOverrideableSDP();
  std::string why = fused_enabled ? why_not(q, k, v, mask, dropout_p, is_causal, enable_gqa, true) : "";
  bool fused = fused_enabled && why.empty();
  if (fused && (!math || forced_route() != 1)) {
    return static_cast<int64_t>(at::SDPBackend::overrideable);
  }
  if (math) {
    return static_cast<int64_t>(at::SDPBackend::math);
  }
  TORCH_CHECK(false, "No available kernel for scaled_dot_product_attention on vk: the math backend is disabled and ",
              fused_enabled ? "the fused kernel does not apply (" + why + ")" : std::string("so is the overrideable one"));
  return static_cast<int64_t>(at::SDPBackend::error);
}

// t with a unit stride along its last dim (copied if not).
at::Tensor unit_last(const at::Tensor& t) {
  return t.size(-1) <= 1 || t.stride(-1) == 1 ? t : t.contiguous();
}

// The additive mask (a bool one turned additive, as PyTorch does) as a
// [B, H, L, S] view of dtype t with a unit or zero stride along keys, and
// attention.slang's MASK (0 without a mask).
at::Tensor prepare_mask(const std::optional<at::Tensor>& mask, int64_t B, int64_t H, int64_t L, int64_t S,
                        at::ScalarType t, uint32_t& mode) {
  mode = 0;
  if (!mask.has_value() || !mask->defined()) return {};
  at::Tensor m = *mask;
  if (m.scalar_type() == at::kBool) {
    m = at::where(m, at::scalar_tensor(0.0, m.options().dtype(t)),
                  at::scalar_tensor(-std::numeric_limits<double>::infinity(), m.options().dtype(t)));
  } else if (m.scalar_type() != t) {
    m = m.to(t);
  }
  while (m.dim() < 4) m = m.unsqueeze(0);
  if (m.size(3) > 1 && m.stride(3) != 1) m = m.contiguous();
  m = m.expand({B, H, L, S});
  mode = S <= 1 || m.stride(3) == 1 ? 1 : 2;
  return m;
}

void set_strides(uint32_t (&dst)[4], const at::Tensor& t, int64_t last) {
  for (int d = 0; d < 3; ++d) {
    dst[d] = t.size(d) > 1 ? static_cast<uint32_t>(t.stride(d)) : 0u;
  }
  dst[3] = static_cast<uint32_t>(last);
}

uint32_t float_bits(double v) {
  float f = static_cast<float>(v);
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}

// The Philox key and counter word for a call's (seed, offset): the offset's
// high word folded into the key (offsets advance by one per call).
void philox_words(uint64_t seed, uint64_t offset, uint32_t (&key)[2], uint32_t& ctr) {
  key[0] = static_cast<uint32_t>(seed);
  key[1] = static_cast<uint32_t>(seed >> 32) ^ static_cast<uint32_t>(offset >> 32);
  ctr = static_cast<uint32_t>(offset);
}

std::string kernel_name(at::ScalarType t, int64_t E, int64_t EV) {
  return std::string("attention_") + attention_label(t) + "_d" + std::to_string(head_class(E, EV));
}

using FwdResult = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor,
                             at::Tensor, at::Tensor>;

FwdResult sdpa_forward(const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
                       const std::optional<at::Tensor>& attn_bias, double dropout_p, bool is_causal,
                       bool return_debug_mask, std::optional<double> scale) {
  std::string why = why_not(q_in, k_in, v_in, attn_bias, dropout_p, is_causal, true, false);
  TORCH_CHECK_NOT_IMPLEMENTED(why.empty(), "_scaled_dot_product_fused_attention_overrideable on vk: ", why);
  at::ScalarType t = q_in.scalar_type();
  int64_t B = q_in.size(0), H = q_in.size(1), L = q_in.size(2), E = q_in.size(3);
  int64_t Hk = k_in.size(1), S = k_in.size(2), EV = v_in.size(3);
  at::Tensor q = unit_last(q_in), k = unit_last(k_in), v = unit_last(v_in);
  uint32_t mask_mode = 0;
  at::Tensor mask = prepare_mask(attn_bias, B, H, L, S, t, mask_mode);
  at::Tensor o = at::empty({B, L, H, EV}, q.options()).transpose(1, 2);
  at::Tensor lse = at::empty({B, H, L}, q.options().dtype(at::kFloat));
  double s = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(E));
  bool dropout = dropout_p > 0;
  uint64_t seed = 0, offset = 0;
  if (dropout) std::tie(seed, offset) = next_rng_call();
  uint32_t key[2], ctr;
  philox_words(seed, offset, key, ctr);

  if (L > 0 && B > 0 && H > 0) {
    AttnParams p{};
    p.q = gpu_address(q);
    p.k = gpu_address(k);
    p.v = gpu_address(v);
    p.o = gpu_address_out(o);
    p.st = gpu_address_out(lse);
    p.mask = mask_mode != 0 ? gpu_address(mask) : 0;
    p.w48 = uint64_t(key[0]) | (uint64_t(key[1]) << 32);
    p.w56 = uint64_t(ctr) | (uint64_t(float_bits(dropout_p)) << 32);
    set_strides(p.qs, q, H);
    set_strides(p.ks, k, L);
    set_strides(p.vs, v, S);
    if (mask_mode != 0) set_strides(p.ms, mask, 0);
    p.ms[3] = float_bits(s);
    std::vector<uint32_t> spec = {0u, uint32_t(E), uint32_t(EV), is_causal ? 1u : 0u, mask_mode, dropout ? 1u : 0u,
                                  uint32_t(H / Hk)};
    dispatch(kernel_name(t, E, EV), spec, p, static_cast<uint32_t>(ceil_div(L, kBlock)), static_cast<uint32_t>(H),
             static_cast<uint32_t>(B));
  }

  at::Tensor debug;
  if (return_debug_mask && dropout) {
    int64_t n = B * H * L * S;
    TORCH_CHECK(n < (int64_t(1) << 32), "torch_vulkan: dropout mask of more than 2^32 elements");
    at::Tensor keep = at::empty({B, H, L, S}, q.options().dtype(at::kInt));
    if (n > 0) {
      AttnAuxParams p{};
      p.out = gpu_address_out(keep);
      p.key[0] = key[0];
      p.key[1] = key[1];
      p.ctr = ctr;
      p.pbits = float_bits(dropout_p);
      p.n = static_cast<uint32_t>(n);
      p.H = static_cast<uint32_t>(H);
      p.L = static_cast<uint32_t>(L);
      p.W = static_cast<uint32_t>(S);
      uint32_t groups = 0;
      check(vkt_grid_1d(static_cast<uint64_t>(n), &groups, &p.gstride));
      dispatch(std::string("attention_aux_") + attention_label(t), {1u}, p, groups);
    }
    debug = keep.ne(0);
  }
  at::Tensor seed_t = at::scalar_tensor(static_cast<int64_t>(seed), at::dtype(at::kLong));
  at::Tensor offset_t = at::scalar_tensor(static_cast<int64_t>(offset), at::dtype(at::kLong));
  return {o, lse, at::Tensor(), at::Tensor(), c10::SymInt(L), c10::SymInt(S), seed_t, offset_t, debug};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_backward(
    const at::Tensor& grad_out, const at::Tensor& q_in, const at::Tensor& k_in, const at::Tensor& v_in,
    const at::Tensor& attn_bias, std::array<bool, 4> grad_input_mask, const at::Tensor& out, const at::Tensor& lse,
    const at::Tensor&, const at::Tensor&, c10::SymInt, c10::SymInt, double dropout_p, bool is_causal,
    const at::Tensor& philox_seed, const at::Tensor& philox_offset, std::optional<double> scale) {
  if (!grad_out.defined()) return {};
  std::optional<at::Tensor> bias = attn_bias.defined() ? std::optional<at::Tensor>(attn_bias) : std::nullopt;
  std::string why = why_not(q_in, k_in, v_in, bias, dropout_p, is_causal, true, false);
  TORCH_CHECK_NOT_IMPLEMENTED(why.empty(), "_scaled_dot_product_fused_attention_overrideable_backward on vk: ", why);
  TORCH_CHECK_NOT_IMPLEMENTED(!grad_input_mask[3],
                              "_scaled_dot_product_fused_attention_overrideable_backward on vk: the mask's gradient");
  at::ScalarType t = q_in.scalar_type();
  int64_t B = q_in.size(0), H = q_in.size(1), L = q_in.size(2), E = q_in.size(3);
  int64_t Hk = k_in.size(1), S = k_in.size(2), EV = v_in.size(3);
  at::Tensor q = unit_last(q_in), k = unit_last(k_in), v = unit_last(v_in);
  uint32_t mask_mode = 0;
  at::Tensor mask = prepare_mask(bias, B, H, L, S, t, mask_mode);
  // out and grad_out in the forward's layout ([B, L, H, EV] in memory)
  auto o_layout = [&](const at::Tensor& x) {
    at::Tensor y = at::empty({B, L, H, EV}, q.options()).transpose(1, 2);
    if (x.scalar_type() == t && x.strides() == y.strides()) return x;
    y.copy_(x);
    return y;
  };
  at::Tensor o = o_layout(out), go = o_layout(grad_out);
  at::Tensor lse_c = lse.to(at::kFloat).contiguous();
  bool dropout = dropout_p > 0;
  uint64_t seed = dropout ? static_cast<uint64_t>(philox_seed.item<int64_t>()) : 0;
  uint64_t offset = dropout ? static_cast<uint64_t>(philox_offset.item<int64_t>()) : 0;
  uint32_t key[2], ctr;
  philox_words(seed, offset, key, ctr);
  double s = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(E));

  // (lse, delta) per row after the dropout's header
  int64_t rows = B * H * L;
  at::Tensor stats = at::empty({4 + 2 * rows}, q.options().dtype(at::kFloat));
  {
    AttnAuxParams p{};
    p.o = gpu_address(o);
    p.go = gpu_address(go);
    p.lse = gpu_address(lse_c);
    p.out = gpu_address_out(stats);
    p.key[0] = key[0];
    p.key[1] = key[1];
    p.ctr = ctr;
    p.pbits = float_bits(dropout_p);
    p.n = static_cast<uint32_t>(rows);
    p.H = static_cast<uint32_t>(H);
    p.L = static_cast<uint32_t>(L);
    p.W = static_cast<uint32_t>(EV);
    uint32_t groups = 0;
    check(vkt_grid_1d(static_cast<uint64_t>(std::max<int64_t>(rows, 1)), &groups, &p.gstride));
    dispatch(std::string("attention_aux_") + attention_label(t), {0u}, p, groups);
  }

  at::Tensor dq = at::empty({B, L, H, E}, q.options());
  at::Tensor dk = at::empty({B, S, Hk, E}, q.options());
  at::Tensor dv = at::empty({B, S, Hk, EV}, q.options());
  AttnParams p{};
  p.q = gpu_address(q);
  p.k = gpu_address(k);
  p.v = gpu_address(v);
  p.o = gpu_address(go);
  p.st = gpu_address(stats);
  p.mask = mask_mode != 0 ? gpu_address(mask) : 0;
  set_strides(p.qs, q, H);
  set_strides(p.ks, k, L);
  set_strides(p.vs, v, S);
  if (mask_mode != 0) set_strides(p.ms, mask, 0);
  p.ms[3] = float_bits(s);
  std::vector<uint32_t> spec = {1u, uint32_t(E), uint32_t(EV), is_causal ? 1u : 0u, mask_mode, dropout ? 1u : 0u,
                                uint32_t(H / Hk)};
  std::string name = kernel_name(t, E, EV);
  bool want_q = grad_input_mask[0], want_kv = grad_input_mask[1] || grad_input_mask[2];
  if (want_q && L > 0 && B > 0 && H > 0) {
    p.w48 = gpu_address_out(dq);
    p.w56 = 0;
    dispatch(name, spec, p, static_cast<uint32_t>(ceil_div(L, kBlock)), static_cast<uint32_t>(H),
             static_cast<uint32_t>(B));
  }
  if (want_kv && S > 0 && B > 0 && Hk > 0) {
    spec[0] = 2u;
    p.w48 = gpu_address_out(dk);
    p.w56 = gpu_address_out(dv);
    dispatch(name, spec, p, static_cast<uint32_t>(ceil_div(S, kBlock)), static_cast<uint32_t>(Hk),
             static_cast<uint32_t>(B));
  }
  return {want_q ? dq.transpose(1, 2) : at::Tensor(), grad_input_mask[1] ? dk.transpose(1, 2) : at::Tensor(),
          grad_input_mask[2] ? dv.transpose(1, 2) : at::Tensor(), at::Tensor()};
}

}  // namespace
}  // namespace vk

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &vk::fused_sdp_choice)
}  // namespace at::native

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fused_sdp_choice", &vk::fused_sdp_choice);
  m.impl("_scaled_dot_product_fused_attention_overrideable", &vk::sdpa_forward);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &vk::sdpa_backward);
}
