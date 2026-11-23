// torchvision's detection ops for device "vk" (docs/log/M57): nms,
// roi_align and its backward, as kernels of their own (shaders/nms.slang,
// shaders/roialign.slang) registered for PrivateUse1 under the torchvision
// namespace. torchvision defines the ops (and their autograd and meta
// kernels) when it is imported; these registrations are boxed kernels, so
// they need neither torchvision's headers nor its schemas, and wait in the
// dispatcher until it is (or never, without torchvision).
//
// They match torchvision's CPU kernels (torchvision/csrc/ops/cpu): the same
// operations in the same order, so nms keeps the same boxes (ties in score
// broken by index, as its stable sort), and roi_align's values agree to the
// last bit where the CPU's compiler keeps that order. float32, and float64
// where the device has it; other dtypes go to run_fallback (float16 and
// bfloat16 then in float32 on the GPU, rounded once; others the CPU's error
// or the CPU).

#include "opcodes.h"
#include "vk.h"

#include <torch/library.h>

#include <cmath>
#include <cstddef>
#include <cstring>

namespace vk {
namespace {

bool vision_dtype(at::ScalarType t) {
  return t == at::kFloat || (t == at::kDouble && vkt_has_float64());
}

// A threshold or scale as the kernel's T: double bits for float64, else
// float bits in the low word (pack_param).
uint64_t param_bits(double v, at::ScalarType t) {
  return pack_param(c10::Scalar(v), t);
}

// ---------------------------------------------------------------- nms

struct NmsParams {
  uint64_t boxes, mask, order, keep, count;
  uint64_t thr;
  uint32_t n, words, stride, pad;
};
static_assert(sizeof(NmsParams) == 64, "must match shaders/nms.slang");

// shaders/nms.slang's MAX_WORDS: the scan keeps a bit per box in shared memory.
constexpr int64_t kNmsMaxBoxes = 32 * 2048;

// The kernel's threshold for float32 boxes: the CPU compares the float IoU
// with the double threshold, and for a float x, x > t exactly when x > the
// largest float not above t.
float float_threshold(double t) {
  float f = static_cast<float>(t);
  if (static_cast<double>(f) > t) f = std::nextafter(f, -INFINITY);
  return f;
}

// torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor
void nms_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  size_t base = stack->size() - 3;
  const at::Tensor dets = (*stack)[base].toTensor();
  const at::Tensor scores = (*stack)[base + 1].toTensor();
  double iou_threshold = (*stack)[base + 2].toDouble();
  // (the CPU's checks, with its messages)
  TORCH_CHECK(dets.dim() == 2, "boxes should be a 2d tensor, got ", dets.dim(), "D");
  TORCH_CHECK(dets.size(1) == 4, "boxes should have 4 elements in dimension 1, got ", dets.size(1));
  TORCH_CHECK(scores.dim() == 1, "scores should be a 1d tensor, got ", scores.dim(), "D");
  TORCH_CHECK(dets.size(0) == scores.size(0), "boxes and scores should have same number of elements in ",
              "dimension 0, got ", dets.size(0), " and ", scores.size(0));
  at::ScalarType t = dets.scalar_type();
  int64_t n = dets.size(0);
  if (!is_vk(dets) || !is_vk(scores) || scores.scalar_type() != t || !vision_dtype(t) || n > kNmsMaxBoxes) {
    run_fallback(op, stack, false);
    return;
  }
  at::Tensor keep;
  if (n == 0) {
    keep = at::empty({0}, dets.options().dtype(at::kLong));
  } else {
    // boxes in score order: descending, ties by index (the CPU's stable sort)
    at::Tensor order = sort_rows(scores.reshape({1, n}).contiguous(), /*descending=*/true).view({n}).to(at::kLong);
    at::Tensor boxes = dets.index_select(0, order).contiguous();
    int64_t words = ceil_div(n, 32);
    at::Tensor mask = at::empty({n, words}, dets.options().dtype(at::kInt));
    at::Tensor out = at::empty({n}, dets.options().dtype(at::kLong));
    at::Tensor count = at::empty({1}, dets.options().dtype(at::kInt));
    NmsParams p{};
    p.boxes = gpu_address(boxes);
    p.mask = gpu_address(mask);
    p.order = gpu_address(order);
    p.keep = gpu_address_out(out);
    p.count = gpu_address_out(count);
    if (t == at::kDouble) {
      p.thr = param_bits(iou_threshold, t);
    } else {
      float f = float_threshold(iou_threshold);
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      p.thr = bits;
    }
    p.n = static_cast<uint32_t>(n);
    p.words = static_cast<uint32_t>(words);
    std::string kernel = std::string("nms_") + dtype_label(t);
    uint32_t groups = 0;
    check(vkt_grid_1d(static_cast<uint64_t>(n * words), &groups, &p.stride));
    dispatch(kernel, {op::NMS_MASK}, p, groups);
    dispatch(kernel, {op::NMS_SCAN}, p, 1);
    keep = out.narrow(0, 0, count.item<int64_t>());  // (the result's size: one read back)
  }
  stack->erase(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  stack->emplace_back(keep);
}

// ---------------------------------------------------------------- roi_align

struct RoiParams {
  uint64_t input, rois, out, key, weight, order, error;
  uint64_t scale;
  uint32_t batch, c, h, w, ph, pw;
  int32_t sampling;
  uint32_t aligned, total, stride, gh, gw, k, pad;
};
static_assert(sizeof(RoiParams) == 120 && offsetof(RoiParams, batch) == 64, "must match shaders/roialign.slang");

RoiParams roi_params(const at::Tensor& rois, double spatial_scale, int64_t batch, int64_t channels, int64_t height,
                     int64_t width, int64_t ph, int64_t pw, int64_t sampling_ratio, bool aligned, at::ScalarType t) {
  RoiParams p{};
  p.rois = gpu_address(rois);
  p.error = device_error_address();
  p.scale = param_bits(spatial_scale, t);
  p.batch = static_cast<uint32_t>(batch);
  p.c = static_cast<uint32_t>(channels);
  p.h = static_cast<uint32_t>(height);
  p.w = static_cast<uint32_t>(width);
  p.ph = static_cast<uint32_t>(ph);
  p.pw = static_cast<uint32_t>(pw);
  p.sampling = static_cast<int32_t>(sampling_ratio);
  p.aligned = aligned ? 1u : 0u;
  p.k = static_cast<uint32_t>(rois.size(0));
  return p;
}

void roi_launch(uint32_t opcode, RoiParams& p, int64_t threads, at::ScalarType t) {
  p.total = static_cast<uint32_t>(threads);
  uint32_t groups = 0;
  check(vkt_grid_1d(static_cast<uint64_t>(threads), &groups, &p.stride));
  dispatch(std::string("roialign_") + dtype_label(t), {opcode}, p, groups);
}

// Sizes the kernels take: offsets and thread counts in 32 bits, below 2^31.
constexpr int64_t kVisionOffsets = int64_t(1) << 31;
bool roi_fits(std::initializer_list<int64_t> counts) {
  for (int64_t c : counts) {
    if (c < 0 || c >= kVisionOffsets) return false;
  }
  return true;
}

// torchvision::roi_align(Tensor input, Tensor rois, float spatial_scale, SymInt pooled_height,
//                        SymInt pooled_width, int sampling_ratio, bool aligned) -> Tensor
void roi_align_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  size_t base = stack->size() - 7;
  const at::Tensor input = (*stack)[base].toTensor();
  const at::Tensor rois = (*stack)[base + 1].toTensor();
  double spatial_scale = (*stack)[base + 2].toDouble();
  int64_t ph = (*stack)[base + 3].toInt(), pw = (*stack)[base + 4].toInt();
  int64_t sampling_ratio = (*stack)[base + 5].toInt();
  bool aligned = (*stack)[base + 6].toBool();
  at::ScalarType t = input.scalar_type();
  bool kernel = is_vk(input) && is_vk(rois) && input.dim() == 4 && rois.dim() == 2 && rois.size(1) == 5 &&
                rois.scalar_type() == t && vision_dtype(t) && ph > 0 && pw > 0;
  if (kernel) {
    int64_t K = rois.size(0), C = input.size(1);
    kernel = roi_fits({input.numel(), K * C * ph * pw, K * 5});
  }
  if (!kernel) {
    run_fallback(op, stack, false);
    return;
  }
  // (the CPU's checks)
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");
  int64_t K = rois.size(0), B = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
  at::Tensor out = at::zeros({K, C, ph, pw}, input.options());
  if (out.numel() > 0) {
    at::Tensor x = input.contiguous(), r = rois.contiguous();
    RoiParams p = roi_params(r, spatial_scale, B, C, H, W, ph, pw, sampling_ratio, aligned, t);
    p.input = gpu_address(x);
    p.out = gpu_address_out(out);
    roi_launch(op::ROI_FORWARD, p, out.numel(), t);
  }
  stack->erase(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  stack->emplace_back(out);
}

// torchvision::_roi_align_backward(Tensor grad, Tensor rois, float spatial_scale, SymInt pooled_height,
//     SymInt pooled_width, SymInt batch_size, SymInt channels, SymInt height, SymInt width,
//     int sampling_ratio, bool aligned) -> Tensor
void roi_align_backward_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  size_t base = stack->size() - 11;
  const at::Tensor grad = (*stack)[base].toTensor();
  const at::Tensor rois = (*stack)[base + 1].toTensor();
  double spatial_scale = (*stack)[base + 2].toDouble();
  int64_t ph = (*stack)[base + 3].toInt(), pw = (*stack)[base + 4].toInt();
  int64_t B = (*stack)[base + 5].toInt(), C = (*stack)[base + 6].toInt();
  int64_t H = (*stack)[base + 7].toInt(), W = (*stack)[base + 8].toInt();
  int64_t sampling_ratio = (*stack)[base + 9].toInt();
  bool aligned = (*stack)[base + 10].toBool();
  at::ScalarType t = grad.scalar_type();
  int64_t K = rois.dim() == 2 ? rois.size(0) : 0;
  bool kernel = is_vk(grad) && is_vk(rois) && rois.dim() == 2 && rois.size(1) == 5 && rois.scalar_type() == t &&
                vision_dtype(t) && grad.dim() == 4 && grad.size(0) == K && grad.size(1) == C && grad.size(2) == ph &&
                grad.size(3) == pw && B >= 0 && H > 0 && W > 0 && roi_fits({B * C * H * W, B * H * W, grad.numel()});
  int64_t gh = sampling_ratio, gw = sampling_ratio;
  at::Tensor r;
  RoiParams p{};
  if (kernel && grad.numel() > 0 && sampling_ratio <= 0) {
    // (adaptive sampling: the largest grid any region takes, one read back)
    r = rois.contiguous();
    p = roi_params(r, spatial_scale, B, C, H, W, ph, pw, sampling_ratio, aligned, t);
    at::Tensor grids = at::empty({K, 2}, rois.options().dtype(at::kInt));
    p.key = gpu_address_out(grids);
    roi_launch(op::ROI_GRID, p, K, t);
    at::Tensor most = grids.amax(0).cpu();
    gh = std::max<int64_t>(most[0].item<int>(), 0);
    gw = std::max<int64_t>(most[1].item<int>(), 0);
  }
  int64_t entries = K * ph * pw * std::max<int64_t>(gh, 0) * std::max<int64_t>(gw, 0) * 4;
  kernel = kernel && roi_fits({entries, entries * C});
  if (!kernel) {
    run_fallback(op, stack, false);
    return;
  }
  at::Tensor grad_input = at::zeros({B, C, H, W}, grad.options());
  if (grad.numel() > 0 && entries > 0) {
    at::Tensor nhwc = at::zeros({B, H, W, C}, grad.options());  // (channels side by side in ROI_GATHER)
    if (!r.defined()) {
      r = rois.contiguous();
      p = roi_params(r, spatial_scale, B, C, H, W, ph, pw, sampling_ratio, aligned, t);
    }
    at::Tensor g = grad.permute({0, 2, 3, 1}).contiguous();  // [K, PH, PW, C]
    at::Tensor keys = at::empty({1, entries}, grad.options().dtype(at::kInt));
    at::Tensor weights = at::empty({entries}, grad.options());
    p.gh = static_cast<uint32_t>(gh);
    p.gw = static_cast<uint32_t>(gw);
    at::Tensor counts = at::empty({K}, grad.options());
    p.key = gpu_address_out(keys);
    p.weight = gpu_address_out(weights);
    p.out = gpu_address_out(counts);
    roi_launch(op::ROI_PLAN, p, entries / 4, t);
    // (entries with one input position together, in entry order: the CPU's order of additions)
    at::Tensor order = sort_rows(keys, /*descending=*/false);
    at::Tensor sorted = at::take(keys, order.to(at::kLong));
    p.input = gpu_address(g);
    p.rois = gpu_address(counts);
    p.out = gpu_address_out(nhwc);
    p.key = gpu_address(sorted);
    p.order = gpu_address(order);
    roi_launch(op::ROI_GATHER, p, entries * C, t);
    grad_input.copy_(nhwc.permute({0, 3, 1, 2}));
  }
  stack->erase(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  stack->emplace_back(grad_input);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
  m.impl("nms", torch::CppFunction::makeFromBoxedFunction<&vk::nms_boxed>());
  m.impl("roi_align", torch::CppFunction::makeFromBoxedFunction<&vk::roi_align_boxed>());
  m.impl("_roi_align_backward", torch::CppFunction::makeFromBoxedFunction<&vk::roi_align_backward_boxed>());
}
