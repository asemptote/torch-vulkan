// Layer norm for device "vk": one workgroup per row (shaders/layernorm.slang)
// for the output and the input gradient, where PyTorch's decomposition took
// 12 kernels forward and 30 backward. The weight and bias gradients are sums
// over the rows (generic reductions). float32/float64 run here; other
// dtypes go through run_fallback (16-bit floats in float32 on the GPU).

#include "opcodes.h"
#include "vk.h"

#include <torch/library.h>

#include <algorithm>
#include <cstddef>
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

void launch(uint32_t op, LayerNormParams& p, at::ScalarType t, bool has_w, bool has_b) {
  int64_t groups = std::min<int64_t>(p.m, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("layernorm_") + dtype_label(t), {op, has_w ? 1u : 0u, has_b ? 1u : 0u}, p,
           static_cast<uint32_t>(groups));
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

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_layer_norm", &vk::native_layer_norm);
  m.impl("native_layer_norm_backward", &vk::native_layer_norm_backward);
}
