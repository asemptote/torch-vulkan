// Fused optimiser steps for device "vk" (torch.optim's fused=True):
// _fused_adam_, _fused_adamw_, _fused_sgd_ and _fused_adagrad_, one launch
// of shaders/optim.slang per parameter tensor. As the CPU's kernels, the
// step counts, grad_scale, found_inf and a tensor lr are read on the host
// (here in one transfer for all tensors), the per-tensor coefficients are
// computed in double precision, and found_inf = 1 skips the step. float32
// and float64 (where the device has it) are computed in their own type,
// float16 and bfloat16 in float32 copies rounded back (the CPU's opmath);
// anything else goes to the CPU.

#include "opcodes.h"
#include "vk.h"

#include <torch/library.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

namespace vk {
namespace {

struct OptimParams {
  uint64_t param, grad, buf1, buf2, buf3;
  uint32_t n, stride, flags, pad;
  uint64_t c[9];
};
static_assert(sizeof(OptimParams) == 128 && offsetof(OptimParams, c) == 56, "must match shaders/optim.slang");

enum : uint32_t { kScaled = 1, kMaximize = 2, kDecay = 4, kAmsgrad = 8, kMomentum = 16, kFirst = 32, kNesterov = 64 };

using OptTensor = std::optional<at::Tensor>;

// The dtype the kernel computes `t` in, or nullopt (the CPU then).
std::optional<at::ScalarType> compute_type(at::ScalarType t) {
  if (t == at::kFloat || t == at::kHalf || t == at::kBFloat16) return at::kFloat;
  if (t == at::kDouble && vkt_has_float64()) return at::kDouble;
  return std::nullopt;
}

// True if every tensor of the lists is a vk tensor of `t` with fewer than
// 2^31 elements, the shape of the parameter it belongs to.
bool on_gpu(at::ScalarType t, at::TensorList params, std::initializer_list<at::TensorList> lists) {
  for (size_t i = 0; i < params.size(); ++i) {
    const at::Tensor& p = params[i];
    if (!is_vk(p) || p.scalar_type() != t || p.numel() >= (int64_t(1) << 31)) return false;
    for (at::TensorList l : lists) {
      if (l.empty()) continue;
      if (!is_vk(l[i]) || l[i].scalar_type() != t || !l[i].sizes().equals(p.sizes())) return false;
    }
  }
  return true;
}

// The values the CPU reads on the host, as it reads them (float32), with
// one transfer for the vk tensors among them.
std::vector<double> host_values(const std::vector<at::Tensor>& tensors) {
  std::vector<double> values(tensors.size());
  std::vector<at::Tensor> on_device;
  std::vector<size_t> where;
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (is_vk(tensors[i])) {
      on_device.push_back(tensors[i].reshape({1}).to(at::kFloat));
      where.push_back(i);
    } else {
      values[i] = tensors[i].item<float>();
    }
  }
  if (!on_device.empty()) {
    at::Tensor host = at::cat(on_device).cpu();
    const float* data = host.data_ptr<float>();
    for (size_t k = 0; k < where.size(); ++k) values[where[k]] = data[k];
  }
  return values;
}

// The step counts, grad_scale and found_inf as the CPU reads them: steps
// (one per tensor; empty for SGD), then the scale (or nullopt), and whether
// found_inf says to skip the step.
struct HostScalars {
  std::vector<double> steps;
  std::optional<double> scale;
  bool skip = false;
};

HostScalars read_scalars(at::TensorList steps, const OptTensor& grad_scale, const OptTensor& found_inf) {
  std::vector<at::Tensor> tensors(steps.begin(), steps.end());
  if (grad_scale.has_value()) tensors.push_back(*grad_scale);
  if (found_inf.has_value()) tensors.push_back(*found_inf);
  std::vector<double> values = host_values(tensors);
  HostScalars s;
  s.steps.assign(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(steps.size()));
  size_t k = steps.size();
  if (grad_scale.has_value()) s.scale = values[k++];
  if (found_inf.has_value()) s.skip = values[k] == 1.0;
  return s;
}

// A tensor the kernel can use for `x` (contiguous, of the compute type):
// x itself, or a copy written back by finish().
struct Operand {
  at::Tensor original, used;
  Operand(const at::Tensor& x, at::ScalarType ct) : original(x), used(x.contiguous().to(ct)) {}
  uint64_t address() { return gpu_address_out(used); }
  void finish() {
    if (!used.is_same(original)) original.copy_(used);
  }
};

void launch(uint32_t op, at::ScalarType ct, const at::Tensor& param, const at::Tensor& grad, const at::Tensor* b1,
            const at::Tensor* b2, const at::Tensor* b3, uint32_t flags, const std::array<double, 9>& c) {
  if (param.numel() == 0) return;
  Operand p(param, ct), g(grad, ct);
  std::optional<Operand> o1, o2, o3;
  OptimParams k{};
  k.param = p.address();
  k.grad = g.address();
  if (b1) k.buf1 = o1.emplace(*b1, ct).address();
  if (b2) k.buf2 = o2.emplace(*b2, ct).address();
  if (b3) k.buf3 = o3.emplace(*b3, ct).address();
  k.n = static_cast<uint32_t>(param.numel());
  k.flags = flags;
  for (size_t i = 0; i < c.size(); ++i) k.c[i] = pack_param(c10::Scalar(c[i]), ct);
  uint32_t grid = 0;
  check(vkt_grid_1d(k.n, &grid, &k.stride));
  dispatch(std::string("optim_") + dtype_label(ct), {op}, k, grid);
  p.finish();
  if (flags & kScaled) g.finish();
  for (auto* o : {&o1, &o2, &o3}) {
    if (o->has_value()) (*o)->finish();
  }
}

template <bool AdamW>
void fused_adam(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2,
                double weight_decay, double eps, bool amsgrad, bool maximize, const OptTensor& grad_scale,
                const OptTensor& found_inf) {
  const char* name = AdamW ? "aten::_fused_adamw_" : "aten::_fused_adam_";
  size_t n = params.size();
  TORCH_CHECK(grads.size() == n);
  TORCH_CHECK(exp_avgs.size() == n);
  TORCH_CHECK(exp_avg_sqs.size() == n);
  if (amsgrad) {
    TORCH_CHECK(max_exp_avg_sqs.size() == n);
  } else {
    TORCH_CHECK(max_exp_avg_sqs.empty());
  }
  TORCH_CHECK(state_steps.size() == n);
  if (n == 0) return;
  auto ct = compute_type(params[0].scalar_type());
  if (!ct || !on_gpu(params[0].scalar_type(), params, {grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs})) {
    on_cpu(name, "", params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2,
           weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
    return;
  }
  HostScalars s = read_scalars(state_steps, grad_scale, found_inf);
  if (s.skip) return;
  uint32_t flags = (s.scale ? kScaled : 0) | (maximize ? kMaximize : 0) | (weight_decay != 0.f ? kDecay : 0) |
                   (amsgrad ? kAmsgrad : 0);
  for (size_t i = 0; i < n; ++i) {
    double step = s.steps[i];
    double bias_correction1 = 1 - std::pow(beta1, step);
    double bias_correction2 = 1 - std::pow(beta2, step);
    std::array<double, 9> c = {s.scale.value_or(1.0),
                               weight_decay,
                               1 - lr * weight_decay,
                               1 - beta1,
                               beta2,
                               1 - beta2,
                               std::sqrt(bias_correction2),
                               eps,
                               -(lr / bias_correction1)};
    launch(AdamW ? op::OPTIM_ADAMW : op::OPTIM_ADAM, *ct, params[i], grads[i], &exp_avgs[i], &exp_avg_sqs[i],
           amsgrad ? &max_exp_avg_sqs[i] : nullptr, flags, c);
  }
}

template <bool AdamW>
void fused_adam_tensor_lr(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs,
                          at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps,
                          const at::Tensor& lr, double beta1, double beta2, double weight_decay, double eps,
                          bool amsgrad, bool maximize, const OptTensor& grad_scale, const OptTensor& found_inf) {
  fused_adam<AdamW>(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1,
                    beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}

void fused_sgd(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffers, double weight_decay,
               double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step,
               const OptTensor& grad_scale, const OptTensor& found_inf) {
  size_t n = params.size();
  TORCH_CHECK(grads.size() == n);
  bool has_buffers = momentum != 0.0;
  if (has_buffers) {
    TORCH_CHECK(momentum_buffers.size() == n);
  } else {
    TORCH_CHECK(momentum_buffers.empty());
  }
  if (n == 0) return;
  auto ct = compute_type(params[0].scalar_type());
  if (!ct || !on_gpu(params[0].scalar_type(), params, {grads, momentum_buffers})) {
    on_cpu("aten::_fused_sgd_", "", params, grads, momentum_buffers, weight_decay, momentum, lr, dampening, nesterov,
           maximize, is_first_step, grad_scale, found_inf);
    return;
  }
  HostScalars s = read_scalars({}, grad_scale, found_inf);
  if (s.skip) return;
  uint32_t flags = (s.scale ? kScaled : 0) | (maximize ? kMaximize : 0) | (weight_decay != 0.0 ? kDecay : 0) |
                   (has_buffers ? kMomentum : 0) | (is_first_step ? kFirst : 0) | (nesterov ? kNesterov : 0);
  std::array<double, 9> c = {s.scale.value_or(1.0), weight_decay, momentum, 1 - dampening, -lr, 0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    launch(op::OPTIM_SGD, *ct, params[i], grads[i], has_buffers ? &momentum_buffers[i] : nullptr, nullptr, nullptr,
           flags, c);
  }
}

void fused_sgd_tensor_lr(at::TensorList params, at::TensorList grads, at::TensorList momentum_buffers,
                         double weight_decay, double momentum, const at::Tensor& lr, double dampening, bool nesterov,
                         bool maximize, bool is_first_step, const OptTensor& grad_scale, const OptTensor& found_inf) {
  fused_sgd(params, grads, momentum_buffers, weight_decay, momentum, lr.item<double>(), dampening, nesterov, maximize,
            is_first_step, grad_scale, found_inf);
}

void fused_adagrad(at::TensorList params, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps,
                   double lr, double lr_decay, double weight_decay, double eps, bool maximize,
                   const OptTensor& grad_scale, const OptTensor& found_inf) {
  size_t n = params.size();
  TORCH_CHECK(grads.size() == n);
  TORCH_CHECK(state_sums.size() == n);
  TORCH_CHECK(state_steps.size() == n);
  if (n == 0) return;
  auto ct = compute_type(params[0].scalar_type());
  if (!ct || !on_gpu(params[0].scalar_type(), params, {grads, state_sums})) {
    on_cpu("aten::_fused_adagrad_", "", params, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps,
           maximize, grad_scale, found_inf);
    return;
  }
  HostScalars s = read_scalars(state_steps, grad_scale, found_inf);
  if (s.skip) return;
  uint32_t flags = (s.scale ? kScaled : 0) | (maximize ? kMaximize : 0) | (weight_decay != 0.0 ? kDecay : 0);
  for (size_t i = 0; i < n; ++i) {
    double clr = lr / (1.0 + (s.steps[i] - 1.0) * lr_decay);
    std::array<double, 9> c = {s.scale.value_or(1.0), weight_decay, eps, clr, 0, 0, 0, 0, 0};
    launch(op::OPTIM_ADAGRAD, *ct, params[i], grads[i], &state_sums[i], nullptr, nullptr, flags, c);
  }
}

void fused_adagrad_tensor_lr(at::TensorList params, at::TensorList grads, at::TensorList state_sums,
                             at::TensorList state_steps, const at::Tensor& lr, double lr_decay, double weight_decay,
                             double eps, bool maximize, const OptTensor& grad_scale, const OptTensor& found_inf) {
  fused_adagrad(params, grads, state_sums, state_steps, lr.item<double>(), lr_decay, weight_decay, eps, maximize,
                grad_scale, found_inf);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fused_adam_", &vk::fused_adam<false>);
  m.impl("_fused_adam_.tensor_lr", &vk::fused_adam_tensor_lr<false>);
  m.impl("_fused_adamw_", &vk::fused_adam<true>);
  m.impl("_fused_adamw_.tensor_lr", &vk::fused_adam_tensor_lr<true>);
  m.impl("_fused_sgd_", &vk::fused_sgd);
  m.impl("_fused_sgd_.tensor_lr", &vk::fused_sgd_tensor_lr);
  m.impl("_fused_adagrad_", &vk::fused_adagrad);
  m.impl("_fused_adagrad_.tensor_lr", &vk::fused_adagrad_tensor_lr);
}
