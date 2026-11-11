// Foreach ops for device "vk" where an accelerator's kernel checks more than
// PyTorch's composite (slow-path) kernel, which vk otherwise runs (each
// tensor's op is a vk kernel):
//
//   _foreach_add.Tensor / _foreach_add_.Tensor   (tensors + alpha * scalar)
//     CUDA's and XPU's kernels (ATen/native/cuda/ForeachBinaryOpScalarTensor.cu)
//     take their fast route when the list is on one device in one dtype, the
//     scalar tensor's too, and there require the scalar tensor on that device
//     ("scalar tensor expected to be on ..."); the slow path accepts a CPU
//     scalar. vk raises as they do, then computes through the slow path.

#include "vk.h"

#include <ATen/native/ForeachUtils.h>
#include <ATen/ops/_foreach_add_native.h>
#include <torch/library.h>

namespace vk {
namespace {

// (ForeachUtils.h can_use_fast_route, for one tensor list: one device, one
// dtype, strided and dense)
bool fast_route(at::TensorList tensors, const at::Tensor& scalar) {
  if (tensors.empty() || tensors[0].scalar_type() != scalar.scalar_type()) {
    return false;
  }
  const auto& first = tensors[0];
  for (const auto& t : tensors) {
    if (!is_vk(t) || t.device() != first.device() || t.scalar_type() != first.scalar_type() ||
        t.layout() != at::kStrided || !t.is_non_overlapping_and_dense()) {
      return false;
    }
  }
  return true;
}

std::vector<at::Tensor> foreach_add_tensor(at::TensorList tensors, const at::Tensor& scalar, const at::Scalar& alpha) {
  at::native::check_foreach_api_restrictions(tensors);
  if (fast_route(tensors, scalar)) {
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1, "scalar tensor expected to be 0 dim but it has ", scalar.dim(),
                " dimensions and ", scalar.numel(), " elements.");
    TORCH_CHECK(tensors[0].device() == scalar.device(), "scalar tensor expected to be on ", tensors[0].device(),
                " but is on ", scalar.device());
  }
  return at::native::foreach_tensor_add_tensor_kernel_slow(tensors, scalar, alpha);
}

void foreach_add_tensor_(at::TensorList tensors, const at::Tensor& scalar, const at::Scalar& alpha) {
  at::native::check_foreach_api_restrictions(tensors);
  if (fast_route(tensors, scalar)) {
    TORCH_CHECK(scalar.dim() == 0 && scalar.numel() == 1, "scalar tensor expected to be 0 dim but has ", scalar.dim(),
                " dimensions and ", scalar.numel(), " elements.");
    TORCH_CHECK(tensors[0].device() == scalar.device(), "scalar tensor is expected to be on ", tensors[0].device(),
                " but is on ", scalar.device());
  }
  at::native::foreach_tensor_add_tensor_kernel_slow_(tensors, scalar, alpha);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_foreach_add.Tensor", &vk::foreach_add_tensor);
  m.impl("_foreach_add_.Tensor", &vk::foreach_add_tensor_);
}
