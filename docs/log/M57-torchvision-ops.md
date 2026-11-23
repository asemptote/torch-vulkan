# torchvision's nms and roi_align (M57)

TorchBench's vision_maskrcnn stopped on vk with the CPU fallback off at
`torchvision::nms` (#143); with the fallback on it ran, 10 calls of `nms` and
40 of `roi_align` going to the CPU, at 559 ms an inference against CUDA's 23.
These are torchvision's operators, which it registers per backend
(`torchvision/csrc/ops`): vk needs kernels of its own, as CUDA and MPS have.

## Registration without a dependency

`csrc/vision.cpp` registers `nms`, `roi_align` and `_roi_align_backward` for
PrivateUse1 in a `TORCH_LIBRARY_IMPL(torchvision, ...)` block, as boxed
kernels: they read their arguments from the stack, so they need neither
torchvision's headers nor its schemas, and the dispatcher keeps them until
torchvision is imported and defines the ops (with their autograd and meta
kernels, which then wrap these). Without torchvision they are never
reached. torchvision stays out of the package's dependencies; the tests
(`tests/python/test_vision.py`) skip without it.

What the kernels do not take goes to `run_fallback`, as an op without a
kernel would: float16 and bfloat16 `roi_align` then runs in float32 on the
GPU and is rounded once (`half_via_float`), and anything else (`nms` in 16
bits, mismatched dtypes, malformed shapes) gets the CPU's error, or runs on
the CPU where the fallback is on.

## nms

torchvision's CPU kernel sorts the boxes by score (descending, stable), then
walks them: a box not yet suppressed is kept and suppresses every later box
whose IoU with it is above the threshold. CUDA's computes a bitmask of the
IoUs above the threshold in blocks of 64 and walks it on the host. Here
(`shaders/nms.slang`):

1. `sort_rows` orders the scores (descending, ties in index order: the
   CPU's stable sort), and the boxes are gathered in that order.
2. `NMS_MASK`: a thread per (box i, word of 32 later boxes) sets bit j where
   box j's IoU with box i is above the threshold. The IoU is the CPU's: the
   same operations in T, in its order (the kernels are compiled without
   contraction), the division correctly rounded (`m_div`), `std::max` and
   `std::min` as their comparisons (NaN coordinates give what they give on
   the CPU). The CPU compares the float IoU with the double threshold; for
   float32 boxes the kernel compares with the largest float not above the
   threshold, the same test.
3. `NMS_SCAN`: one workgroup of 256 walks the sorted boxes with a bit per box
   in shared memory: where box i is not suppressed, it is kept and its mask
   row is or-ed in. The decision is uniform (every thread reads the same
   word), so only kept boxes cost a barrier.
4. The kept count is read back (the result's size; nonzero does the same).

The suppression is the CPU's whatever the order its comparisons ran in, so
the kept indices are the same, ties included. The mask takes n² / 8 bytes
and the scan's shared memory 2,048 words: up to 65,536 boxes (512 MiB of
mask) run here, more on the CPU. Mask R-CNN calls it with a few thousand.

## roi_align

`shaders/roialign.slang` follows torchvision's CPU kernels
(`roi_align_kernel.cpp`, `roi_align_common.h`) operation by operation: each
region's start, bin size and adaptive grid (`ceil(height / pooled_height)`),
each sample's position `start + ph bin + (iy + 0.5) bin / grid`, the
bilinear corners with their clamps, and the weights; the divisions correctly
rounded, `1. - ly` in T (exact either way), samples outside the image as the
CPU's zero weights at position 0 (so a NaN or infinity there propagates as
on the CPU).

- **Forward** (`ROI_FORWARD`): a thread per output (n, c, ph, pw), summing
  its bin's samples in the CPU's order and dividing by the count.
- **Backward**, without atomics: torchvision's CUDA kernel adds each
  sample's four corner contributions into the input gradient with atomic
  adds, in no fixed order. Here `ROI_PLAN` writes, for each sample slot of
  every region (n, ph, pw, iy, ix, the largest grid's) and each of its four
  corners, the input position as a key (or a skip key) and the bilinear
  weight (and each region's count); `sort_rows` sorts the keys stably;
  `ROI_GATHER` (a thread per sorted entry and channel, where a run of equal
  keys starts) adds the run's `g[n, c, ph, pw] * weight / count` in entry
  order and writes the sum. The channels are a warp's lanes (g and the
  gradient channels-last, permuted around the launch), so a warp's lanes
  take one branch and their loads coalesce: with the entries as the lanes,
  each warp waited for its longest run, 10 times slower (70 ms against 6.8
  below). For
  one channel, the CPU adds into an input element in exactly that order
  (regions, then ph, pw, iy, ix, the corners), from zero.
  Deterministic, and the same bits as the CPU. With adaptive sampling the
  largest grid is read back first (`ROI_GRID`, one read).
- A region's batch index outside [0, batch) contributes nothing and sets the
  device's error word, raised as an `IndexError` when the device next
  synchronises, where the CPU raises at once.
- An empty set of regions (Mask R-CNN's mask head when nothing is detected)
  gives an empty result, on the GPU.

Forward and backward agree with the CPU bit for bit on lavapipe and the H100
(float32 and float64, aligned or not, fixed and adaptive sampling, regions
smaller than a pixel, inverted, outside the image).

## Mask R-CNN's ops

torchvision's `maskrcnn_resnet50_fpn`, one training step and one inference
on vk (lavapipe, 256 x 256), with a `TorchDispatchMode` counting
torchvision's ops: `nms` (3 calls), `roi_align` (17) and
`_roi_align_backward` (8), and nothing else of torchvision's; with the
fallback on, `fallback_counts()` stayed empty. The other ops torchvision
registers per backend (`roi_pool`, `ps_roi_align`, `ps_roi_pool`,
`deform_conv2d`) are not used by Mask R-CNN (nor by the other TorchBench
models: #130's sweep met only these two) and still have no Vulkan kernels.

## Numbers (H100 NVL)

Timings (best of 3 x 20 calls on vk, 5 on the CPU: the job's 4
cores, torchvision's kernels, which are single-threaded):

| | vk | CPU | same result |
|---|--:|--:|---|
| nms, 1,000 boxes | 0.54 ms | 2.6 ms | yes |
| nms, 5,000 boxes | 2.2 ms | 64 ms | yes |
| nms, 20,000 boxes | 15.1 ms | 973 ms | yes |
| roi_align, [2, 256, 200, 272], 1,000 regions, 7 x 7, sampling 2 | 1.4 ms | 275 ms | bit for bit |
| its backward | 6.8 ms | | |
| roi_align, 100 regions, 14 x 14 | 0.37 ms | 70 ms | bit for bit |
| its backward | 3.0 ms | | |

nms's time is mostly the scan (one workgroup, a barrier per kept box).
`maskrcnn_resnet50_fpn` (untrained, one 800 x 800 image) runs inference on vk
with the fallback off and nothing falling back, in about 100 ms (not the TorchBench
runner's setting, so not comparable with #143's 559 ms and CUDA's 23 ms;
#130's sweep will measure that).
