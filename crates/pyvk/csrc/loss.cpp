// nll_loss forward and backward for device "vk" (shaders/loss.slang): one
// thread per sample for the per-sample losses and weights (forward, then a
// sum for the "sum" and "mean" reductions) or for the gradient's one entry
// per row (backward), where PyTorch's decompositions took 16 and 12 kernels.
// The ops' meta kernels run first for PyTorch's checks and result shapes.
// float32/float64 inputs with int64 or uint8 targets run here; others go
// through on_cpu.

#include "opcodes.h"
#include "vk.h"

#include <ATen/TensorUtils.h>
#include <torch/library.h>

#include <cstddef>

namespace vk {
namespace {

struct NllParams {
  uint64_t x, target, weight, out, wsum, gout, error;
  uint32_t n, c;
  int64_t ignore;
  uint32_t stride, pad;
};
static_assert(sizeof(NllParams) == 80 && offsetof(NllParams, ignore) == 64, "must match shaders/loss.slang");

constexpr int64_t kNone = 0, kMean = 1;

bool kernel_ok(const at::Tensor& self, const at::Tensor& target, const std::optional<at::Tensor>& weight) {
  at::ScalarType t = self.scalar_type();
  bool dtype = t == at::kFloat || (t == at::kDouble && vkt_has_float64());
  bool w = !weight.has_value() || !weight->defined() || (is_vk(*weight) && weight->scalar_type() == t);
  return is_vk(self) && is_vk(target) && dtype && w &&
         (target.scalar_type() == at::kLong || target.scalar_type() == at::kByte) &&
         self.numel() < (int64_t(1) << 31);
}

// The CPU's checks (nll_loss_forward's and _backward's meta functions), with
// its messages. (PyTorch's meta kernels for these ops are its decompositions,
// whose messages differ.) The forward takes a 1-d input with a target of one
// element, 0-d or 1-d; the backward only a 0-d one.
void check_shapes(const at::Tensor& self, const at::Tensor& target, bool forward) {
  TORCH_CHECK(self.dim() > 0 && self.dim() <= 2, "input tensor should be 1D or 2D");
  TORCH_CHECK(target.dim() <= 1, "0D or 1D target tensor expected, multi-target not supported");
  if (forward && self.dim() == 1 && target.dim() == 1) {
    TORCH_CHECK_VALUE(target.size(0) == 1, "For 1D input, 1D target must have size 1, but got target size: ",
                      target.size(0));
  }
  bool no_batch_dim = self.dim() == 1 && (target.dim() == 0 || forward);
  TORCH_CHECK(no_batch_dim || self.size(0) == target.size(0), "size mismatch (got input: ", self.sizes(),
              ", target: ", target.sizes(), ")");
}

NllParams params(const at::Tensor& x, const at::Tensor& target, const at::Tensor& weight, int64_t n, int64_t c,
                 int64_t ignore_index) {
  NllParams p{};
  p.x = gpu_address(x);
  p.target = gpu_address(target);
  p.weight = weight.defined() ? gpu_address(weight) : 0;
  p.error = device_error_address();
  p.n = static_cast<uint32_t>(n);
  p.c = static_cast<uint32_t>(c);
  p.ignore = ignore_index;
  return p;
}

void launch(uint32_t op, NllParams& p, at::ScalarType t, bool has_weight, bool per_sample) {
  uint32_t groups = 0;
  check(vkt_grid_1d(p.n, &groups, &p.stride));
  dispatch(std::string("loss_") + dtype_label(t), {op, has_weight ? 1u : 0u, per_sample ? 1u : 0u}, p, groups);
}

std::tuple<at::Tensor, at::Tensor> nll_loss_forward(const at::Tensor& self, const at::Tensor& target,
                                                    const std::optional<at::Tensor>& weight, int64_t reduction,
                                                    int64_t ignore_index) {
  if (!kernel_ok(self, target, weight)) {
    if (is_vk(self)) {
      // (the CPU's checks, in its order: shapes, weight, target dtype)
      check_shapes(self, target, true);
      int64_t classes = self.size(-1);
      const at::Tensor w = weight.has_value() ? *weight : at::Tensor();
      TORCH_CHECK(!w.defined() || (w.dim() == 1 && w.numel() == classes), "weight tensor should be defined either for all ",
                  classes, " classes or no classes but got weight tensor of shape: ", w.defined() ? w.sizes() : at::IntArrayRef{});
      TORCH_CHECK(target.scalar_type() == at::kLong || target.scalar_type() == at::kByte,
                  "expected target dtype to be Long or Byte, but got ", target.scalar_type());
    }
    auto s = on_cpu("aten::nll_loss_forward", "", self, target, weight, reduction, ignore_index);
    return {s[0].toTensor(), s[1].toTensor()};
  }
  check_shapes(self, target, true);
  int64_t classes = self.size(-1);
  TORCH_CHECK(!weight.has_value() || !weight->defined() || (weight->dim() == 1 && weight->numel() == classes),
              "weight tensor should be defined either for all ", classes, " classes or no classes but got weight "
              "tensor of shape: ", weight->sizes());
  int64_t c = self.size(-1), n = self.dim() == 1 ? 1 : self.size(0);
  at::Tensor x = self.contiguous(), tg = target.to(at::kLong).contiguous();
  at::Tensor w = weight.has_value() && weight->defined() ? weight->contiguous() : at::Tensor();
  at::Tensor loss = at::empty({n}, self.options()), wsum = at::empty({n}, self.options());
  if (n > 0) {
    NllParams p = params(x, tg, w, n, c, ignore_index);
    p.out = gpu_address_out(loss);
    p.wsum = gpu_address_out(wsum);
    launch(op::NLL_FORWARD, p, self.scalar_type(), w.defined(), false);
  }
  if (reduction == kNone && self.dim() == 2) {
    return {loss, at::zeros({}, self.options())};
  }
  // (one sample, 1-d input: total_weight is its weight for every reduction, as on the CPU)
  at::Tensor total = self.dim() == 1 ? wsum.view(at::IntArrayRef{}) : wsum.sum();
  at::Tensor out = self.dim() == 1 ? loss.view(at::IntArrayRef{}) : loss.sum();
  if (reduction == kMean) {
    out = out / total;
  }
  return {out, total};
}

at::Tensor nll_loss_backward(const at::Tensor& grad_output, const at::Tensor& self, const at::Tensor& target,
                             const std::optional<at::Tensor>& weight, int64_t reduction, int64_t ignore_index,
                             const at::Tensor& total_weight) {
  if (!kernel_ok(self, target, weight) || !is_vk(grad_output) || !is_vk(total_weight) ||
      grad_output.scalar_type() != self.scalar_type() || total_weight.scalar_type() != self.scalar_type()) {
    return on_cpu("aten::nll_loss_backward", "", grad_output, self, target, weight, reduction, ignore_index,
                  total_weight)[0]
        .toTensor();
  }
  check_shapes(self, target, false);
  TORCH_CHECK(total_weight.numel() == 1, "expected total_weight to be a  single element tensor, got: ",
              total_weight.sizes(), " (", total_weight.numel(), " elements)");
  TORCH_CHECK(!weight.has_value() || !weight->defined() || weight->numel() == self.size(-1),
              "weight tensor should be defined either for all or no classes");
  if (reduction == kNone && self.dim() == 2) {
    at::check_dim_size(grad_output, 1, 0, self.size(0));
  } else {
    TORCH_CHECK(grad_output.dim() <= 1 && grad_output.numel() == 1,
                "Expected a single element grad_output tensor, but got: ", grad_output.sizes());
  }
  int64_t c = self.size(-1), n = self.dim() == 1 ? 1 : self.size(0);
  at::Tensor gx = at::zeros(self.sizes(), self.options());
  if (n == 0) {
    return gx;
  }
  at::Tensor tg = target.to(at::kLong).contiguous();
  at::Tensor w = weight.has_value() && weight->defined() ? weight->contiguous() : at::Tensor();
  bool per_sample = reduction == kNone && self.dim() == 2;
  // (as the CPU: -(grad_output / total_weight) for "mean", -grad_output for "sum")
  at::Tensor g = per_sample ? grad_output.contiguous()
                            : at::neg(reduction == kMean ? grad_output.reshape({1}) / total_weight.reshape({1})
                                                         : grad_output.reshape({1}));
  NllParams p = params(self, tg, w, n, c, ignore_index);
  p.x = 0;
  p.out = gpu_address_out(gx);
  p.gout = gpu_address(g);
  launch(op::NLL_BACKWARD, p, self.scalar_type(), w.defined(), per_sample);
  return gx;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("nll_loss_forward", &vk::nll_loss_forward);
  m.impl("nll_loss_backward", &vk::nll_loss_backward);
}
