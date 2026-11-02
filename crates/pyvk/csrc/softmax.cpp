// Softmax and log-softmax, forward and backward, for device "vk": one
// workgroup per row along the softmax dim (shaders/softmax.slang), where
// PyTorch's decomposition took 7 to 18 kernels. The ops' meta kernels run
// first: PyTorch's checks (dim range, matching sizes) and result dtypes.
// float32/float64 run here; other dtypes go through run_fallback (16-bit
// floats in float32 on the GPU).

#include "opcodes.h"
#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <algorithm>
#include <cstddef>

namespace vk {
namespace {

struct SoftmaxParams {
  uint64_t a, b, out;
  uint32_t rows, n, inner, ngroups;
};
static_assert(sizeof(SoftmaxParams) == 40 && offsetof(SoftmaxParams, rows) == 24, "must match shaders/softmax.slang");

constexpr int64_t kMaxGroups = 65535;

bool kernel_dtype(const at::Tensor& t) {
  return is_vk(t) && (t.scalar_type() == at::kFloat || (t.scalar_type() == at::kDouble && vkt_has_float64()));
}

// op over a (and b) along dim; a contiguous result of a's dtype and shape.
at::Tensor run(uint32_t op, const at::Tensor& a_in, const at::Tensor* b_in, int64_t dim) {
  at::Tensor a = a_in.contiguous();
  at::Tensor b = b_in != nullptr ? b_in->contiguous() : at::Tensor();
  at::Tensor out = at::empty(a.sizes(), a.options());
  if (a.numel() == 0) {
    return out;
  }
  int64_t d = at::maybe_wrap_dim(dim, std::max<int64_t>(a.dim(), 1));
  int64_t outer = 1, n = 1, inner = 1;
  for (int64_t i = 0; i < a.dim(); ++i) {
    (i < d ? outer : i == d ? n : inner) *= a.size(i);
  }
  TORCH_CHECK(a.numel() < (int64_t(1) << 31), "softmax: more elements than 32-bit offsets address");
  SoftmaxParams p{};
  p.a = gpu_address(a);
  p.b = b.defined() ? gpu_address(b) : 0;
  p.out = gpu_address_out(out);
  p.rows = static_cast<uint32_t>(outer * inner);
  p.n = static_cast<uint32_t>(n);
  p.inner = static_cast<uint32_t>(inner);
  int64_t groups = std::min<int64_t>(outer * inner, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("softmax_") + dtype_label(a.scalar_type()), {op}, p, static_cast<uint32_t>(groups));
  return out;
}

void assign_out(const at::Tensor& r, at::Tensor& out) {
  check_out_device(r, out);
  TORCH_CHECK(out.scalar_type() == r.scalar_type(), "Expected out tensor to have dtype ", r.scalar_type(),
              ", but got ", out.scalar_type(), " instead");
  at::native::resize_output(out, r.sizes());
  out.copy_(r);
}

template <bool Log>
at::Tensor softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  // (the CPU kernel's check first, as its message is what the CPU raises)
  TORCH_CHECK(!half_to_float, Log ? "log_softmax" : "softmax",
              " with half to float conversion is not supported on CPU");
  (Log ? at::_log_softmax : at::_softmax)(meta_of(self), dim, half_to_float);  // PyTorch's checks
  if (!kernel_dtype(self)) {
    return on_cpu(Log ? "aten::_log_softmax" : "aten::_softmax", "", self, dim, half_to_float)[0].toTensor();
  }
  return run(Log ? op::SOFTMAX_LOG_FORWARD : op::SOFTMAX_FORWARD, self, nullptr, dim);
}

template <bool Log>
at::Tensor& softmax_out(const at::Tensor& self, int64_t dim, bool half_to_float, at::Tensor& out) {
  assign_out(softmax<Log>(self, dim, half_to_float), out);
  return out;
}

template <bool Log>
at::Tensor softmax_backward(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
                            at::ScalarType input_dtype) {
  // (PyTorch's checks: same sizes, the dim)
  (Log ? at::_log_softmax_backward_data : at::_softmax_backward_data)(meta_of(grad_output), meta_of(output), dim,
                                                                       input_dtype);
  at::ScalarType h = grad_output.scalar_type();
  if (at::isReducedFloatingType(h) && output.scalar_type() == h && input_dtype == h && is_vk(grad_output) &&
      is_vk(output)) {
    // (16-bit floats: in float32, rounded once)
    at::Tensor out_f = output.to(at::kFloat);
    return run(Log ? op::SOFTMAX_LOG_BACKWARD : op::SOFTMAX_BACKWARD, grad_output.to(at::kFloat), &out_f, dim).to(h);
  }
  if (!kernel_dtype(grad_output) || output.scalar_type() != grad_output.scalar_type() ||
      grad_output.scalar_type() != input_dtype || !is_vk(output)) {
    return on_cpu(Log ? "aten::_log_softmax_backward_data" : "aten::_softmax_backward_data", "", grad_output,
                  output, dim, input_dtype)[0]
        .toTensor();
  }
  return run(Log ? op::SOFTMAX_LOG_BACKWARD : op::SOFTMAX_BACKWARD, grad_output, &output, dim);
}

template <bool Log>
at::Tensor& softmax_backward_out(const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
                                 at::ScalarType input_dtype, at::Tensor& grad_input) {
  assign_out(softmax_backward<Log>(grad_output, output, dim, input_dtype), grad_input);
  return grad_input;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_softmax", &vk::softmax<false>);
  m.impl("_softmax.out", &vk::softmax_out<false>);
  m.impl("_log_softmax", &vk::softmax<true>);
  m.impl("_log_softmax.out", &vk::softmax_out<true>);
  m.impl("_softmax_backward_data", &vk::softmax_backward<false>);
  m.impl("_softmax_backward_data.out", &vk::softmax_backward_out<false>);
  m.impl("_log_softmax_backward_data", &vk::softmax_backward<true>);
  m.impl("_log_softmax_backward_data.out", &vk::softmax_backward_out<true>);
}
