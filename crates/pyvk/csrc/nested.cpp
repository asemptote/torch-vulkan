// Nested tensors for device "vk".
//
// Jagged layout (torch.nested with layout=torch.jagged): the conversions
// between packed values and a padded dense tensor that most of its ops use
// (_jagged_to_padded_dense_forward, _padded_dense_to_jagged_forward), as one
// gather or scatter of rows with each row's batch found by a sorted search
// in the offsets, where the CPU loops over the batch.
//
// Strided layout (torch.nested.nested_tensor's default): PyTorch's
// NestedTensor kernels are written once from ordinary ops on the buffer and
// the size/stride/offset tensors, but registered for the built-in devices
// only, so NestedTensorPrivateUse1 had none: every op on a vk nested tensor
// failed, and in autograd (which asks for _nested_tensor_size) aborted the
// process. The ops whose CPU and CUDA kernels are one function in PyTorch's
// native_functions.yaml (tools/dev/nested_ops.py lists them) get that kernel
// by redispatching to NestedTensorCPU; its ops then run on the vk buffer as
// usual. Layer norm, whose kernels call the CPU's LayerNormKernel stub
// directly, runs as the dense op on the buffer.

#include "vk.h"

#include <ATen/NestedTensorImpl.h>
#include <ATen/native/nested/NestedTensorMath.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <torch/library.h>

#include <cmath>
#include <limits>
#include <vector>

namespace vk {
namespace {

// The CPU's _get_padding_value: an infinite padding for an integer dtype
// is the dtype's largest or lowest value.
at::Scalar padding_scalar(double padding_value, at::ScalarType t) {
  if (!std::isinf(padding_value) || at::isFloatingType(t)) return padding_value;
  at::Scalar s;
  AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, t, "padding_value", [&] {
    s = padding_value > 0 ? std::numeric_limits<scalar_t>::max() : std::numeric_limits<scalar_t>::lowest();
  });
  return s;
}

// Each packed row's batch (offsets[b] <= row < offsets[b + 1]; -1 or
// batches outside them) and position in it.
std::pair<at::Tensor, at::Tensor> rows_in_batches(const at::Tensor& offsets, int64_t rows) {
  at::Tensor o = offsets.to(at::kLong);
  at::Tensor r = at::arange(rows, o.options());
  at::Tensor b = at::searchsorted(o, r, /*out_int32=*/false, /*right=*/true) - 1;
  int64_t batches = o.size(0) - 1;
  at::Tensor pos = r - o.index_select(0, at::clamp(b, 0, std::max<int64_t>(batches, 0)));
  return {b, pos};
}

at::Tensor jagged_to_padded_dense(const at::Tensor& values, at::TensorList offsets_list, at::IntArrayRef max_lengths,
                                  double padding_value) {
  bool ok = offsets_list.size() == 1 && max_lengths.size() == 1 && offsets_list[0].dim() == 1 &&
            offsets_list[0].size(0) >= 1 && values.dim() >= 1 && is_vk(values) && is_vk(offsets_list[0]) &&
            !at::isComplexType(values.scalar_type());
  if (!ok) {
    return on_cpu("aten::_jagged_to_padded_dense_forward", "", values, offsets_list, max_lengths, padding_value)[0]
        .toTensor();
  }
  const at::Tensor& offsets = offsets_list[0];
  int64_t batches = offsets.size(0) - 1, length = max_lengths[0], total = values.size(0);
  std::vector<int64_t> shape = {batches, length};
  shape.insert(shape.end(), values.sizes().begin() + 1, values.sizes().end());
  int64_t inner = 1;
  for (size_t d = 2; d < shape.size(); ++d) inner *= shape[d];
  at::Scalar fill = padding_scalar(padding_value, values.scalar_type());
  if (total == 0 || batches == 0 || length == 0 || inner == 0) return at::full(shape, fill, values.options());
  // Rows past their batch's max length (or outside every batch) go to a
  // spare row, dropped after.
  auto [b, pos] = rows_in_batches(offsets, total);
  at::Tensor keep = at::logical_and(at::logical_and(at::ge(b, 0), at::lt(b, batches)), at::lt(pos, length));
  at::Tensor dest = at::where(keep, b * length + pos, batches * length);
  at::Tensor out = at::full({batches * length + 1, inner}, fill, values.options());
  out.index_copy_(0, dest, values.reshape({total, inner}));
  return out.narrow(0, 0, batches * length).view(shape);
}

at::Tensor padded_dense_to_jagged(const at::Tensor& padded, at::TensorList offsets_list, std::optional<int64_t> total_L) {
  bool ok = offsets_list.size() == 1 && offsets_list[0].dim() == 1 && offsets_list[0].size(0) >= 1 &&
            padded.dim() >= 2 && is_vk(padded) && is_vk(offsets_list[0]) &&
            offsets_list[0].size(0) - 1 <= padded.size(0);
  const at::Tensor& offsets = offsets_list[0];
  int64_t batches = ok ? offsets.size(0) - 1 : 0, width = ok ? padded.size(1) : 0;
  int64_t final_offset = 0;
  if (ok) {
    // (the CPU's checks, from one transfer: the final offset and the longest batch)
    at::Tensor o = offsets.to(at::kLong);
    at::Tensor stats = batches > 0 ? at::stack({o[-1], (o.narrow(0, 1, batches) - o.narrow(0, 0, batches)).max()})
                                   : o.narrow(0, 0, 1);
    at::Tensor host = stats.cpu();
    final_offset = host[0].item<int64_t>();
    int64_t longest = batches > 0 ? host[1].item<int64_t>() : 0;
    ok = longest <= width && (!total_L.has_value() || *total_L == final_offset) && final_offset >= 0;
  }
  if (!ok) {
    return on_cpu("aten::_padded_dense_to_jagged_forward", "", padded, offsets_list, total_L)[0].toTensor();
  }
  int64_t total = total_L.value_or(final_offset);
  std::vector<int64_t> shape = {total};
  shape.insert(shape.end(), padded.sizes().begin() + 2, padded.sizes().end());
  int64_t inner = 1;
  for (size_t d = 1; d < shape.size(); ++d) inner *= shape[d];
  if (total == 0 || inner == 0 || batches == 0 || width == 0) return at::empty(shape, padded.options());
  auto [b, pos] = rows_in_batches(offsets, total);
  at::Tensor src = at::clamp(b, 0, batches - 1) * width + at::clamp(pos, 0, width - 1);
  return padded.narrow(0, 0, batches).reshape({batches * width, inner}).index_select(0, src).view(shape);
}

// ---------------------------------------------------------------- strided nested tensors

// NestedTensorCPU kernels shared with NestedTensorCUDA (device-generic), plus
// to_padded_tensor and bmm, whose CPU kernels are generic ATen code too.
const char* const kNestedOps[] = {
    "native_dropout",
    "native_dropout_backward",
    "abs",
    "abs_",
    "sgn",
    "sgn_",
    "add.Tensor",
    "add_.Tensor",
    "all.dim",
    "logical_not",
    "logical_not_",
    "cat",
    "chunk",
    "copy_",
    "cos",
    "div.Tensor",
    "div.Scalar",
    "embedding",
    "empty_like",
    "fill_.Scalar",
    "fill_.Tensor",
    "isnan",
    "is_same_size",
    "linear",
    "linear_backward",
    "matmul",
    "matmul_backward",
    "matmul.out",
    "value_selecting_reduction_backward",
    "mul.Tensor",
    "mul_.Tensor",
    "mul.Scalar",
    "mul_.Scalar",
    "narrow",
    "ones_like",
    "neg",
    "neg_",
    "relu",
    "relu_",
    "gelu_",
    "gelu",
    "gelu_backward",
    "select.int",
    "_nested_select_backward",
    "silu",
    "silu_",
    "silu_backward",
    "sin",
    "detach",
    "_softmax",
    "_softmax_backward_data",
    "split_with_sizes",
    "squeeze",
    "squeeze.dim",
    "squeeze.dims",
    "sqrt",
    "tanh",
    "tanh_",
    "threshold_backward",
    "transpose.int",
    "_nested_tensor_size",
    "_nested_tensor_strides",
    "_nested_tensor_storage_offsets",
    "_nested_from_padded_and_nested_example",
    "unsqueeze",
    "where.self",
    "where.self_out",
    "clone",
    "zero_",
    "sub.Tensor",
    "values",
    "unbind.int",
    "_to_copy",
    "masked_fill.Scalar",
    "view",
    "eq.Scalar",
    "eq.Tensor",
    "ge.Scalar",
    "gt.Scalar",
    "normal_",
    "alias",
    "isinf",
    "isposinf",
    "isneginf",
    "_test_autograd_multiple_dispatch.fullcoverage",
    "_test_autograd_multiple_dispatch.ntonly",
    "_safe_softmax",
    "_transformer_encoder_layer_fwd",
    "to_padded_tensor",
    "bmm",
};

// Ops on dense tensors that build nested ones, whose CPU kernel is PyTorch's
// generic one (shared with CUDA, or named so): for vk inputs, the same kernel
// through the CPU key (a vk buffer makes a vk nested tensor).
const char* const kDenseOps[] = {
    "_nested_view_from_buffer",
    "_nested_from_padded",
    "_nested_tensor_from_mask",
    "_nested_tensor_from_mask_left_aligned",
};

void to_nested_cpu(const c10::OperatorHandle& op, c10::DispatchKeySet, torch::jit::Stack* stack) {
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::NestedTensorCPU), stack);
}

void to_cpu_kernel(const c10::OperatorHandle& op, c10::DispatchKeySet, torch::jit::Stack* stack) {
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
}

// Layer norm of a contiguous nested tensor over its trailing dims: the dense
// op on the buffer seen as [M, N] (the CPU's nested kernel's checks first).
std::tuple<at::Tensor, at::Tensor, at::Tensor> nested_layer_norm(const at::Tensor& input,
                                                                 at::IntArrayRef normalized_shape,
                                                                 const std::optional<at::Tensor>& weight_opt,
                                                                 const std::optional<at::Tensor>& bias_opt,
                                                                 double eps) {
  TORCH_CHECK_VALUE(weight_opt && bias_opt, "NestedTensor layer_norm requires weight and bias");
  TORCH_CHECK(!weight_opt->is_nested(), "NestedTensor weight not supported for layer_norm");
  TORCH_CHECK(!bias_opt->is_nested(), "NestedTensor bias not supported for layer_norm");
  auto* nt = at::native::get_nested_tensor_impl(input);
  TORCH_CHECK(at::native::nested_tensor_impl_is_contiguous(nt));
  auto [M, N] = at::native::_check_nested_layer_norm_inputs(*nt, normalized_shape, *weight_opt, *bias_opt);
  at::Tensor buffer = nt->get_buffer().view({M, N});
  auto [out, mean, rstd] = at::native_layer_norm(buffer, {N}, weight_opt->reshape({N}), bias_opt->reshape({N}), eps);
  return {at::native::wrap_buffer(out.reshape({-1}), nt->get_nested_sizes()), mean.reshape({M}), rstd.reshape({M})};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> nested_layer_norm_backward(
    const at::Tensor& grad, const at::Tensor& input, at::IntArrayRef normalized_shape, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight_opt, const std::optional<at::Tensor>& bias_opt,
    std::array<bool, 3> mask) {
  TORCH_CHECK_VALUE(weight_opt.has_value() && bias_opt.has_value(),
                    "NestedTensor layer_norm requires weight and bias");
  auto* nt = at::native::get_nested_tensor_impl(input);
  auto* ng = at::native::get_nested_tensor_impl(grad);
  TORCH_CHECK(at::native::nested_tensor_impl_is_contiguous(nt) && at::native::nested_tensor_impl_is_contiguous(ng));
  auto [M, N] = at::native::_check_nested_layer_norm_inputs(*nt, normalized_shape, *weight_opt, *bias_opt);
  auto [gx, gw, gb] = at::native_layer_norm_backward(
      ng->get_buffer().view({M, N}), nt->get_buffer().view({M, N}), {N}, mean.reshape({M, 1}), rstd.reshape({M, 1}),
      weight_opt->reshape({N}), bias_opt->reshape({N}), mask);
  return {gx.defined() ? at::native::wrap_buffer(gx.reshape({-1}), nt->get_nested_sizes()) : gx,
          gw.defined() ? gw.view(weight_opt->sizes()) : gw, gb.defined() ? gb.view(bias_opt->sizes()) : gb};
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_jagged_to_padded_dense_forward", TORCH_FN(vk::jagged_to_padded_dense));
  m.impl("_padded_dense_to_jagged_forward", TORCH_FN(vk::padded_dense_to_jagged));
  for (const char* name : vk::kDenseOps) {
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&vk::to_cpu_kernel>());
  }
}

TORCH_LIBRARY_IMPL(aten, NestedTensorPrivateUse1, m) {
  for (const char* name : vk::kNestedOps) {
    m.impl(name, torch::CppFunction::makeFromBoxedFunction<&vk::to_nested_cpu>());
  }
  m.impl("native_layer_norm", TORCH_FN(vk::nested_layer_norm));
  m.impl("native_layer_norm_backward", TORCH_FN(vk::nested_layer_norm_backward));
}
