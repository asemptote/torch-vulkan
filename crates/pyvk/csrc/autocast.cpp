// Autocast for device "vk" (torch.autocast("vk", dtype=...)): PyTorch's
// policy for CUDA and XPU (its shared op lists in ATen/autocast_mode.h),
// registered for the AutocastPrivateUse1 key. Matrix products and
// convolutions run in the autocast dtype (float16 by default, or bfloat16),
// ops that lose accuracy in 16 bits (exp, log, norms, losses, softmax, sums)
// in float32, and ops with several inputs (addcmul, atan2, dot, ...) in the
// widest of their dtypes. Every other op falls through.

#include <ATen/autocast_mode.h>
#include <torch/library.h>

namespace vk {
namespace {

at::Tensor binary_cross_entropy_banned(const at::Tensor&, const at::Tensor&, const std::optional<at::Tensor>&,
                                       int64_t) {
  TORCH_CHECK(false,
              "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
              "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
              "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
              "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
              "safe to autocast.");
}

}  // namespace
}  // namespace vk

// (the op lists' signatures name these unqualified, as within at::autocast)
using at::IntArrayRef;
using at::Scalar;
using at::ScalarType;
using at::Tensor;

TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
#define VK_KERNEL_LOW_PRECISION_FP(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, lower_precision_fp)
  AT_FORALL_LOWER_PRECISION_FP(VK_KERNEL_LOW_PRECISION_FP)
#define VK_KERNEL_FP32(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32)
  AT_FORALL_FP32(VK_KERNEL_FP32)
#define VK_KERNEL_FP32_SET_OPT_DTYPE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, fp32_set_opt_dtype)
  AT_FORALL_FP32_SET_OPT_DTYPE(VK_KERNEL_FP32_SET_OPT_DTYPE)
  AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE(KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_PRIVATEUSEONE)
#define VK_KERNEL_PROMOTE(...) KERNEL_PRIVATEUSEONE(__VA_ARGS__, promote)
  AT_FORALL_PROMOTE(VK_KERNEL_PROMOTE)
  m.impl(TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"), TORCH_FN((&vk::binary_cross_entropy_banned)));
}
