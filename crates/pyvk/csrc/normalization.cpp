// Batch normalization and attention entry points for device "vk" (M12),
// composed from existing kernels (reductions, element-wise ops, matmul).
//
// Batch norm has several aten entry points (native_batch_norm,
// _native_batch_norm_legit*, _batch_norm_with_update/_no_update) that
// differ in which statistics they update and return; all share
// batch_norm_impl, which matches the CPU's outputs (including empty saved
// statistics in eval mode and the unbiased running-variance update).

#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

namespace vk {

namespace {

// Shape [1, C, 1, ...] for broadcasting per-channel values over x.
std::vector<int64_t> channel_shape(const at::Tensor& x) {
  std::vector<int64_t> shape(x.dim(), 1);
  shape[1] = x.size(1);
  return shape;
}

std::vector<int64_t> reduce_dims(const at::Tensor& x) {
  std::vector<int64_t> dims = {0};
  for (int64_t d = 2; d < x.dim(); ++d) dims.push_back(d);
  return dims;
}

// (output, save_mean, save_invstd); updates running stats in place when
// training and they are given.
std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_impl(const at::Tensor& x,
                                                               const std::optional<at::Tensor>& weight,
                                                               const std::optional<at::Tensor>& bias,
                                                               const std::optional<at::Tensor>& running_mean,
                                                               const std::optional<at::Tensor>& running_var,
                                                               bool training, double momentum, double eps) {
  auto shape = channel_shape(x);
  // Statistics in float32 at least (as the CPU does for reduced precision).
  at::ScalarType acc = x.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  at::Tensor xf = x.to(acc);
  at::Tensor mean, invstd, save_mean, save_invstd;
  if (training) {
    auto dims = reduce_dims(x);
    int64_t n = x.numel() / x.size(1);
    mean = xf.mean(dims);
    at::Tensor var = (xf - mean.view(shape)).square().mean(dims);  // biased
    invstd = at::rsqrt(var + eps);
    save_mean = mean;
    save_invstd = invstd;
    if (running_mean.has_value() && running_mean->defined()) {
      running_mean->mul_(1 - momentum).add_(mean.to(running_mean->scalar_type()), momentum);
    }
    if (running_var.has_value() && running_var->defined()) {
      at::Tensor unbiased = n > 1 ? var * (static_cast<double>(n) / (n - 1)) : var;
      running_var->mul_(1 - momentum).add_(unbiased.to(running_var->scalar_type()), momentum);
    }
  } else {
    TORCH_CHECK(running_mean.has_value() && running_mean->defined() && running_var.has_value() &&
                    running_var->defined(),
                "running_mean and running_var must be defined in evaluation mode");
    mean = running_mean->to(acc);
    invstd = at::rsqrt(running_var->to(acc) + eps);
    save_mean = at::empty({0}, x.options().dtype(acc));
    save_invstd = at::empty({0}, x.options().dtype(acc));
  }
  at::Tensor scale = invstd, shift = -mean * invstd;
  if (weight.has_value() && weight->defined()) {
    scale = scale * weight->to(acc);
    shift = shift * weight->to(acc);
  }
  if (bias.has_value() && bias->defined()) {
    shift = shift + bias->to(acc);
  }
  at::Tensor out = (xf * scale.view(shape) + shift.view(shape)).to(x.scalar_type());
  return {out, save_mean, save_invstd};
}

bool bn_ok(const at::Tensor& x) {
  at::ScalarType t = x.scalar_type();
  return is_vk(x) && x.dim() >= 2 && (at::isFloatingType(t) && storage_label(t) != nullptr);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var, bool training,
    double momentum, double eps) {
  if (!bn_ok(x)) {
    auto s = on_cpu("aten::native_batch_norm", "", x, weight, bias, running_mean, running_var, training, momentum,
                    eps);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  return batch_norm_impl(x, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    at::Tensor& running_mean, at::Tensor& running_var, bool training, double momentum, double eps) {
  return batch_norm_impl(x, weight, bias, running_mean, running_var, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit_no_stats(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    bool training, double momentum, double eps) {
  return batch_norm_impl(x, weight, bias, std::nullopt, std::nullopt, training, momentum, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_legit_no_training(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var, double momentum, double eps) {
  return batch_norm_impl(x, weight, bias, running_mean, running_var, false, momentum, eps);
}

// (output, save_mean, save_invstd, reserve)
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_with_update(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    at::Tensor& running_mean, at::Tensor& running_var, double momentum, double eps) {
  auto [out, mean, invstd] = batch_norm_impl(x, weight, bias, running_mean, running_var, true, momentum, eps);
  return {out, mean, invstd, at::empty({0}, x.options().dtype(at::kByte))};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_no_update(
    const at::Tensor& x, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var, double momentum,
    double eps) {
  bool have_stats = running_mean.has_value() && running_mean->defined();
  auto [out, mean, invstd] =
      batch_norm_impl(x, weight, bias, running_mean, running_var, !have_stats, momentum, eps);
  return {out, mean, invstd, at::empty({0}, x.options().dtype(at::kByte))};
}

// ---------------------------------------------------------------- attention

// The CPU's fused attention kernel as math: (output, logsumexp [B, H, L]).
std::tuple<at::Tensor, at::Tensor> sdpa_flash_for_cpu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                      double dropout_p, bool is_causal,
                                                      const std::optional<at::Tensor>& attn_mask,
                                                      std::optional<double> scale) {
  double s = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(q.size(-1)));
  at::Tensor scores = at::matmul(q, k.transpose(-2, -1)) * s;
  if (is_causal) {
    int64_t L = q.size(-2), S = k.size(-2);
    at::Tensor keep = at::ones({L, S}, q.options().dtype(at::kBool)).tril();
    scores = scores.masked_fill(at::logical_not(keep), -std::numeric_limits<double>::infinity());
  }
  if (attn_mask.has_value() && attn_mask->defined()) {
    scores = attn_mask->scalar_type() == at::kBool
                 ? scores.masked_fill(at::logical_not(*attn_mask), -std::numeric_limits<double>::infinity())
                 : scores + *attn_mask;
  }
  at::Tensor lse = at::logsumexp(scores, -1);
  at::Tensor attn = at::exp(scores - lse.unsqueeze(-1));
  if (dropout_p > 0) {
    attn = at::dropout(attn, dropout_p, true);
  }
  // The CPU kernel returns logsumexp as [B, H, L] in float32 for float inputs.
  return {at::matmul(attn, v), lse.to(q.scalar_type() == at::kDouble ? at::kDouble : at::kFloat)};
}

// Which fused attention kernel to use: always the math path on "vk".
int64_t fused_sdp_choice(const at::Tensor&, const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
                         double, bool, std::optional<double>, bool) {
  return 0;  // sdp::SDPBackend::math
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("native_batch_norm", &vk::native_batch_norm);
  m.impl("_native_batch_norm_legit", &native_batch_norm_legit);
  m.impl("_native_batch_norm_legit.no_stats", &native_batch_norm_legit_no_stats);
  m.impl("_native_batch_norm_legit_no_training", &native_batch_norm_legit_no_training);
  m.impl("_batch_norm_with_update", &batch_norm_with_update);
  m.impl("_batch_norm_no_update", &batch_norm_no_update);
  m.impl("_scaled_dot_product_flash_attention_for_cpu", &sdpa_flash_for_cpu);
  m.impl("_fused_sdp_choice", &fused_sdp_choice);
}
