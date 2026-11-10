// Layer norm for device "vk": one workgroup per row (shaders/layernorm.slang)
// for the output and the input gradient, where PyTorch's decomposition took
// 12 kernels forward and 30 backward. The weight and bias gradients are sums
// over the rows (generic reductions). float32/float64 run here; other
// dtypes go through run_fallback (16-bit floats in float32 on the GPU).

#include "opcodes.h"
#include "vk.h"

#include <ATen/OpMathType.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>

namespace vk {
namespace {

struct LayerNormParams {
  uint64_t x, g, w, b, out, mean, rstd;
  uint32_t m, n, ngroups, pad;
  uint64_t eps;
};
static_assert(sizeof(LayerNormParams) == 80 && offsetof(LayerNormParams, eps) == 72,
              "must match shaders/layernorm.slang");

constexpr int64_t kMaxGroups = 65535;

// The CPU's _check_layer_norm_inputs, with its messages.
void check_inputs(const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& weight,
                  const at::Tensor& bias) {
  int64_t k = static_cast<int64_t>(normalized_shape.size());
  TORCH_CHECK(k >= 1, "Expected normalized_shape to be at least 1-dimensional, i.e., ",
              "containing at least one element, but got normalized_shape = ", normalized_shape);
  TORCH_CHECK(!weight.defined() || weight.sizes().equals(normalized_shape),
              "Expected weight to be of same shape as normalized_shape, but got ", "weight of shape ", weight.sizes(),
              " and normalized_shape = ", normalized_shape);
  TORCH_CHECK(!bias.defined() || bias.sizes().equals(normalized_shape),
              "Expected bias to be of same shape as normalized_shape, but got ", "bias of shape ", bias.sizes(),
              " and normalized_shape = ", normalized_shape);
  if (input.dim() < k || !input.sizes().slice(input.dim() - k).equals(normalized_shape)) {
    std::stringstream ss;
    ss << "Given normalized_shape=" << normalized_shape << ", expected input with shape [*";
    for (int64_t size : normalized_shape) {
      ss << ", " << size;
    }
    ss << "], but got input of size" << input.sizes();
    TORCH_CHECK(false, ss.str());
  }
}

bool kernel_ok(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  at::ScalarType t = x.scalar_type();
  bool dtype = t == at::kFloat || (t == at::kDouble && vkt_has_float64());
  for (const at::Tensor* p : {&w, &b}) {
    if (p->defined() && (!is_vk(*p) || p->scalar_type() != t)) {
      return false;
    }
  }
  return is_vk(x) && dtype && x.numel() < (int64_t(1) << 31);
}

std::vector<int64_t> stat_shape(const at::Tensor& input, int64_t k) {
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end() - k);
  shape.insert(shape.end(), k, 1);
  return shape;
}

// An empty input (no rows, or rows of no elements), as the CPU: the output
// empty, the statistics of rows of nothing mean 0 and rstd NaN (the
// statistics in float32 for 16-bit inputs, as the non-empty path's).
bool empty_ok(const at::Tensor& input, const at::Tensor& w, const at::Tensor& b) {
  if (!is_vk(input) || !at::isFloatingType(input.scalar_type())) return false;
  for (const at::Tensor* p : {&w, &b}) {
    if (p->defined() && !is_vk(*p)) return false;
  }
  return true;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> empty_layer_norm(const at::Tensor& input, int64_t k) {
  at::ScalarType st = at::isReducedFloatingType(input.scalar_type()) ? at::kFloat : input.scalar_type();
  auto stats = input.options().dtype(st);
  return {at::empty(input.sizes(), input.options()), at::zeros(stat_shape(input, k), stats),
          at::full(stat_shape(input, k), std::numeric_limits<double>::quiet_NaN(), stats)};
}

void launch(uint32_t op, LayerNormParams& p, at::ScalarType t, bool has_w, bool has_b) {
  int64_t groups = std::min<int64_t>(p.m, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("layernorm_") + dtype_label(t), {op, has_w ? 1u : 0u, has_b ? 1u : 0u}, p,
           static_cast<uint32_t>(groups));
}

// 16-bit inputs: computed in float32 and rounded once. Parameters may be of
// the input's dtype or float32 (the CPU's mixed rule). The saved statistics
// stay float32, as PyTorch's meta kernel (and CUDA) gives them and as
// PyTorch's tests (_refs.native_layer_norm) expect, where the CPU gives the
// input's dtype.
bool half_ok(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b) {
  if (!is_vk(x) || !at::isReducedFloatingType(x.scalar_type())) {
    return false;
  }
  for (const at::Tensor* p : {&w, &b}) {
    if (p->defined() && (!is_vk(*p) || (p->scalar_type() != x.scalar_type() && p->scalar_type() != at::kFloat))) {
      return false;
    }
  }
  return true;
}

at::Tensor widened(const at::Tensor& t) {
  return t.defined() ? t.to(at::kFloat) : t;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm(const at::Tensor& input,
                                                                 at::IntArrayRef normalized_shape,
                                                                 const std::optional<at::Tensor>& weight_opt,
                                                                 const std::optional<at::Tensor>& bias_opt,
                                                                 double eps) {
  at::Tensor weight = weight_opt.value_or(at::Tensor()), bias = bias_opt.value_or(at::Tensor());
  check_inputs(input, normalized_shape, weight, bias);
  int64_t k = static_cast<int64_t>(normalized_shape.size());
  int64_t n = c10::multiply_integers(normalized_shape);
  int64_t m = n == 0 ? 0 : input.numel() / n;
  if (half_ok(input, weight, bias) && m > 0 && n > 0) {
    auto [out, mean, rstd] = vk::native_layer_norm(widened(input), normalized_shape, widened(weight), widened(bias), eps);
    return {out.to(input.scalar_type()), mean, rstd};
  }
  if ((m == 0 || n == 0) && empty_ok(input, weight, bias)) return empty_layer_norm(input, k);
  if (!kernel_ok(input, weight, bias) || m == 0 || n == 0) {
    auto s = on_cpu("aten::native_layer_norm", "", input, normalized_shape, weight_opt, bias_opt, eps);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  at::Tensor x = input.contiguous();
  at::Tensor w = weight.defined() ? weight.contiguous() : weight, b = bias.defined() ? bias.contiguous() : bias;
  at::Tensor out = at::empty(input.sizes(), input.options());
  at::Tensor mean = at::empty(stat_shape(input, k), input.options());
  at::Tensor rstd = at::empty(stat_shape(input, k), input.options());
  LayerNormParams p{};
  p.x = gpu_address(x);
  p.w = w.defined() ? gpu_address(w) : 0;
  p.b = b.defined() ? gpu_address(b) : 0;
  p.out = gpu_address_out(out);
  p.mean = gpu_address_out(mean);
  p.rstd = gpu_address_out(rstd);
  p.m = static_cast<uint32_t>(m);
  p.n = static_cast<uint32_t>(n);
  p.eps = pack_param(eps, input.scalar_type());
  launch(op::LAYERNORM_FORWARD, p, input.scalar_type(), w.defined(), b.defined());
  return {out, mean, rstd};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> mask) {
  at::Tensor weight = weight_opt.value_or(at::Tensor()), bias = bias_opt.value_or(at::Tensor());
  check_inputs(input, normalized_shape, weight, bias);
  int64_t k = static_cast<int64_t>(normalized_shape.size());
  int64_t n = c10::multiply_integers(normalized_shape);
  int64_t m = n == 0 ? 0 : input.numel() / n;
  if (half_ok(input, weight, bias) && m > 0 && n > 0 && is_vk(grad_out) && is_vk(mean) && is_vk(rstd) &&
      grad_out.scalar_type() == input.scalar_type() && grad_out.sizes().equals(input.sizes())) {
    auto [gx, gw, gb] = vk::native_layer_norm_backward(widened(grad_out), widened(input), normalized_shape, widened(mean),
                                                   widened(rstd), widened(weight), widened(bias), mask);
    return {gx.defined() ? gx.to(input.scalar_type()) : gx, gw.defined() ? gw.to(weight.scalar_type()) : gw,
            gb.defined() ? gb.to(bias.scalar_type()) : gb};
  }
  if ((m == 0 || n == 0) && empty_ok(input, weight, bias) && is_vk(grad_out)) {
    // (no rows: zero parameter gradients; rows of nothing: empty ones)
    auto zeros = [](const at::Tensor& p) { return at::zeros(p.sizes(), p.options()); };
    return {mask[0] ? at::empty(input.sizes(), input.options()) : at::Tensor(),
            mask[1] && weight.defined() ? zeros(weight) : at::Tensor(),
            mask[2] && bias.defined() ? zeros(bias) : at::Tensor()};
  }
  if (!kernel_ok(input, weight, bias) || m == 0 || n == 0 || !is_vk(grad_out) ||
      grad_out.scalar_type() != input.scalar_type() || mean.scalar_type() != input.scalar_type() ||
      rstd.scalar_type() != input.scalar_type() || !grad_out.sizes().equals(input.sizes())) {
    auto s = on_cpu("aten::native_layer_norm_backward", "", grad_out, input, normalized_shape, mean, rstd, weight_opt,
                    bias_opt, mask);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  at::Tensor x = input.contiguous(), g = grad_out.contiguous();
  at::Tensor mu = mean.contiguous(), rs = rstd.contiguous();
  at::Tensor gx, gw, gb;
  if (mask[0]) {
    at::Tensor w = weight.defined() ? weight.contiguous() : weight;
    gx = at::empty(input.sizes(), input.options());
    LayerNormParams p{};
    p.x = gpu_address(x);
    p.g = gpu_address(g);
    p.w = w.defined() ? gpu_address(w) : 0;
    p.out = gpu_address_out(gx);
    p.mean = gpu_address(mu);
    p.rstd = gpu_address(rs);
    p.m = static_cast<uint32_t>(m);
    p.n = static_cast<uint32_t>(n);
    launch(op::LAYERNORM_BACKWARD, p, input.scalar_type(), w.defined(), false);
  }
  at::Tensor g2 = g.view({m, n});
  if (mask[1] && weight.defined()) {
    at::Tensor xhat = (x.view({m, n}) - mu.view({m, 1})) * rs.view({m, 1});
    gw = (g2 * xhat).sum(0).view(normalized_shape);
  }
  if (mask[2] && bias.defined()) {
    gb = g2.sum(0).view(normalized_shape);
  }
  return {gx, gw, gb};
}

// PyTorch's rms_norm_composite (not exported), with tensor ops: in the
// opmath dtype, rstd = rsqrt(mean(x^2) + eps) over the last dims.
std::tuple<at::Tensor, at::Tensor> rms_norm_composite(const at::Tensor& input, at::IntArrayRef normalized_shape,
                                                      const std::optional<at::Tensor>& weight_opt,
                                                      std::optional<double> eps) {
  std::vector<int64_t> dims;
  for (size_t i = 0; i < normalized_shape.size(); ++i) dims.push_back(input.dim() - static_cast<int64_t>(i) - 1);
  at::ScalarType ot = at::toOpMathType(input.scalar_type());
  at::Tensor up = input.to(ot);
  bool dbl = ot == at::kDouble || ot == at::kComplexDouble;
  double e = eps.value_or(dbl ? std::numeric_limits<double>::epsilon()
                              : static_cast<double>(std::numeric_limits<float>::epsilon()));
  at::Tensor rstd = at::rsqrt(at::pow(up, 2).mean(dims, true).add_(e));
  at::Tensor out = up.mul(rstd);
  if (weight_opt.has_value() && weight_opt->defined()) out = out.mul(*weight_opt);
  return {out.contiguous().to(input.scalar_type()), rstd.contiguous()};
}

// RMS norm (rms_norm calls _fused_rms_norm, whose autograd formula calls
// _fused_rms_norm_backward): one workgroup per row as layer norm's, the
// weight gradient a sum over the rows. 16-bit inputs in float32 (rstd
// float32, as PyTorch's composite gives it); other cases PyTorch's composite.
std::tuple<at::Tensor, at::Tensor> fused_rms_norm(const at::Tensor& input, at::IntArrayRef normalized_shape,
                                                  const std::optional<at::Tensor>& weight_opt,
                                                  std::optional<double> eps) {
  at::Tensor weight = weight_opt.value_or(at::Tensor());
  int64_t k = static_cast<int64_t>(normalized_shape.size());
  int64_t n = c10::multiply_integers(normalized_shape);
  int64_t m = n == 0 ? 0 : input.numel() / n;
  bool half = at::isReducedFloatingType(input.scalar_type());
  at::Tensor x = half ? input.to(at::kFloat) : input, w = half ? widened(weight) : weight;
  if (!kernel_ok(x, w, at::Tensor()) || m == 0 || n == 0 || k < 1) {
    return rms_norm_composite(input, normalized_shape, weight_opt, eps);
  }
  double e = eps.value_or(x.scalar_type() == at::kDouble ? std::numeric_limits<double>::epsilon()
                                                         : static_cast<double>(std::numeric_limits<float>::epsilon()));
  x = x.contiguous();
  if (w.defined()) w = w.contiguous();
  at::Tensor out = at::empty(input.sizes(), x.options());
  at::Tensor rstd = at::empty(stat_shape(input, k), x.options());
  LayerNormParams p{};
  p.x = gpu_address(x);
  p.w = w.defined() ? gpu_address(w) : 0;
  p.out = gpu_address_out(out);
  p.rstd = gpu_address_out(rstd);
  p.m = static_cast<uint32_t>(m);
  p.n = static_cast<uint32_t>(n);
  p.eps = pack_param(e, x.scalar_type());
  launch(op::LAYERNORM_RMS_FORWARD, p, x.scalar_type(), w.defined(), false);
  return {half ? out.to(input.scalar_type()) : out, rstd};
}

std::tuple<at::Tensor, at::Tensor> fused_rms_norm_backward(const at::Tensor& grad_out, const at::Tensor& input,
                                                           at::IntArrayRef normalized_shape, const at::Tensor& rstd,
                                                           const std::optional<at::Tensor>& weight_opt,
                                                           std::array<bool, 2> mask) {
  at::Tensor weight = weight_opt.value_or(at::Tensor());
  int64_t n = c10::multiply_integers(normalized_shape);
  int64_t m = n == 0 ? 0 : input.numel() / n;
  bool half = at::isReducedFloatingType(input.scalar_type());
  at::Tensor x = half ? input.to(at::kFloat) : input, w = half ? widened(weight) : weight;
  at::Tensor g = grad_out.to(x.scalar_type()).contiguous(), rs = rstd.to(x.scalar_type()).contiguous();
  x = x.contiguous();
  at::Tensor gx, gw;
  if (mask[0]) {
    if (kernel_ok(x, w, at::Tensor()) && m > 0 && n > 0 && is_vk(g) && is_vk(rs)) {
      gx = at::empty(input.sizes(), x.options());
      at::Tensor wc = w.defined() ? w.contiguous() : w;
      LayerNormParams p{};
      p.x = gpu_address(x);
      p.g = gpu_address(g);
      p.w = wc.defined() ? gpu_address(wc) : 0;
      p.out = gpu_address_out(gx);
      p.rstd = gpu_address(rs);
      p.m = static_cast<uint32_t>(m);
      p.n = static_cast<uint32_t>(n);
      launch(op::LAYERNORM_RMS_BACKWARD, p, x.scalar_type(), wc.defined(), false);
    } else {  // (the formula with tensor ops)
      at::Tensor x2 = x.view({m, n}), g2 = g.view({m, n}), r2 = rs.view({m, 1});
      at::Tensor gw2 = w.defined() ? g2 * w.reshape({1, n}) : g2;
      gx = (r2 * gw2 - x2 * r2 * r2 * r2 * (gw2 * x2).sum(1, true) / n).view(input.sizes());
    }
    if (half) gx = gx.to(input.scalar_type());
  }
  if (mask[1] && weight.defined()) {
    gw = (g.view({m, n}) * x.view({m, n}) * rs.view({m, 1})).sum(0).view(normalized_shape);
    if (half) gw = gw.to(weight.scalar_type());
  }
  return {gx, gw};
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_layer_norm", &vk::native_layer_norm);
  m.impl("native_layer_norm_backward", &vk::native_layer_norm_backward);
  m.impl("_fused_rms_norm", &vk::fused_rms_norm);
  m.impl("_fused_rms_norm_backward", &vk::fused_rms_norm_backward);
}
