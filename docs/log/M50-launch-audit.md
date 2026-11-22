# Composites of many launches: an audit, and four kernels (M50)

This cleanup looked for ops built from many small kernel launches. It
started from #110, which added one launch per tap for convolution weights
and depthwise convolutions; #112 (flip) and M48 (the depthwise kernel)
replaced those. Each launch costs the host a few microseconds to record,
plus a barrier where it depends on the one before. On the Mali-G57,
launches are expensive: an op of a few hundred launches costs milliseconds
before it computes anything. Some of the composites found here were slow
for another reason as well. They scatter through PyTorch's
decompositions, which sort an int64 key per element.

This note lists the composites that launch the most for typical shapes. It
fixes four of them with a kernel each, where one kernel was clearly
simpler and much faster, and says what is left for graph capture (#78) or
for kernels of their own.

## Counting

`benchmarks/launches.py` counts and times each case:

- It runs a call once to warm up, then counts the launches of one call
  with the runtime's dispatch counter, and times it.
- With `--ops N` it also shows the N aten ops that launched the most. A
  TorchDispatchMode sees each op together with the launches of the ops
  its kernel runs inside it.
- `TORCH_VK_TRACE=1` names each launch's kernel instead.

The cases cover the composites in `python/torch_vulkan/_kernels.py`, the
decompositions vk takes from `torch._decomp`, and csrc's loops over a
small dim, at typical shapes:

- images [8, 64, 56, 56];
- 28 x 28 grids;
- 50 to 200 time steps;
- 50 parameters.

## The worst offenders (HD 530, main)

Launches per call (forward / backward, where the backward includes the
forward). The times are from one run while other jobs shared the GPU, so
they are high by up to 2-3x; the launch counts are exact.

| op (shape) | launches | ms | why |
|---|---|---|---|
| `ctc_loss`, T = 200, S = 10 | 6,223 / 14,696 | 170 / 415 | Python loop over time steps: 31 launches per step forward, 42 backward |
| `flip` of a 7-dim weight view (#112) | 32,772 | 202 | offset table, one launch per outer index |
| depthwise convolution with a multiplier (M48) | 4,106 / 4,123 | 31 / 226 | one multiply-add per tap, over 5-dim views looped on the host |
| depthwise convolution k3 stride 2 (M48) | 67 / 1,628 | 16 / 77 | the same, through the transposed convolution's phases |
| `fractional_max_pool2d` | 1,060 / 1,116 | 10 / 29 | Python loop over output positions' window starts, per dim |
| bicubic `grid_sample` | 233 / 691 | 136 / 2,691 | composite of gathers; backward by autograd through it |
| `upsample_bicubic2d` (`interpolate`, bicubic) | 193 / 590 | 499 / 532 | PyTorch's decomposition; the backward builds interpolation matrices from it |
| `adaptive_max_pool2d` | 522 / 554 | 24 / 28 | gathers per window offset |
| Adam step, 50 tensors (foreach) | 350 | 5 | one launch per tensor per foreach op |
| LSTM cell, 16 steps | 290 / 930 | 6 / 20 | 16 launches per cell forward, 35 backward |
| reflection pad 2-D backward | 212 (194 for the backward) | 113 | PyTorch's decomposition: gathers and an index_add |
| `fold` (col2im) | 139 | 4,901 | PyTorch's decomposition: an index_put with accumulation, one sort key per element |
| `unfold` backward (col2im) | 152 | 2,554 | the same |
| replication pad 3-D backward | 137 | 10 | an index_add per padded dim |
| `pdist` backward | 68 | 11 | composite |
| `embedding_bag` | 44 / 70 | 4 / 5 | composite |
| bilinear `grid_sample` backward | 43 | 22 | the sort that sums grad_input in the CPU's order |

## Fixed here

ms per call, main -> this change (HD 530 timed in one quiet run of
`benchmarks/launches.py`; lavapipe on a cluster workstation with 4
threads; on the Mali-G57, main's paths are timed by calling them directly
on this build, where they remain as fallbacks; backward rows include the
forward, as `launches.py` times them):

| op | launches | HD 530 | lavapipe | Mali-G57 |
|---|---|---|---|---|
| bicubic `grid_sample` [8, 64, 56, 56], 28 x 28 grid | 233 -> 1 | 75 -> 4.8 | 337 -> 4.4 | 179 -> 7.5 |
| its backward | 691 -> 58 | 1,322 -> 29 | 10,489 -> 221 | 3,387 -> 95 |
| `upsample_bicubic2d` x2 of [8, 64, 56, 56] | 193 -> 4 | 501 -> 14.5 | 1,893 -> 26 | 821 -> 54 |
| its backward | 590 -> 21 | 535 -> 48 | 2,215 -> 168 | -> 58 (backward alone) |
| `fold` k3, [8, 576, 3136] columns | 139 -> 1 | 2,389 -> 7.8 | 19,173 -> 88 | 6,009 -> 35 |
| `unfold` k3 backward | 152 -> 14 | 2,459 -> 75 | 19,571 -> 478 | -> 143 |
| reflection pad 2-D backward, [8, 64, 56, 56] | 212 -> 20 | 112 -> 12 | 680 -> 81 | 192 (backward alone) -> 31 |
| replication pad 3-D backward, [2, 8, 12, 12, 12] | 137 -> 13 | 11 -> 2.9 | 87 -> 2.4 | -> 6.2 |

### Bicubic grid_sample: shaders/gridsample.slang, mode 2

`grid_sample`'s bicubic mode was a composite of gathers, written to follow
PyTorch 2.15's CPU kernels (resolve_cubic_taps), with a backward through
autograd. It is now a mode of gridsample.slang, with the same arithmetic:

- per dim, the Keys coefficients of the unnormalized coordinate's
  fractional part, the unnormalization fused as the CPU's compiler fuses
  it;
- four taps around the floor, each resolved by the padding mode; a dropped
  tap reads nothing;
- the sums in the CPU's order: 2-D along x per row and then the rows, 3-D
  the 64 terms in turn.

The backward writes grad_grid (the CPU's sums of the coefficients'
derivatives) and the 16 or 64 taps' offsets and weights. grad_input is the
bilinear backward's scatter: one sort of the taps' keys, all channels at
once.

The kernel reads the input through its strides. An input with an extent
of 2^31 or more (only an expanded view can have one) keeps the composite,
since the kernel's sizes are 32-bit.

On the way: the reflection counted its spans with the GPU's division, and
Intel's rounds 455 / 455 just below 1. That folded a tap a whole span away
(PyTorch's reference inputs, 456 wide with align_corners, hit it). Both the
bicubic taps and the bilinear coordinate now divide correctly rounded
(`m_div`).

### upsample_bicubic2d: shaders/gridsample.slang, mode 3

`interpolate(mode="bicubic")` on floats ran PyTorch's decomposition: 16
gathers through int64 indices and clamps, 193 launches. Its backward
multiplied by interpolation matrices built by running that decomposition
on an identity, on every call. It now runs on the same taps:

- The host computes each output row's and column's source coordinate with
  the CPU's float arithmetic (area_pixel_compute_scale and
  area_pixel_compute_source_index), Ho + Wo values, which the GPU stacks
  into a grid.
- The kernel applies guard_index_and_lambda and clamps the taps into the
  input. The sums are cubic_interp1d's, along x and then y.
- The backward is the grid backward's tap weights and its scatter.

uint8 keeps its fixed-point kernel.

### fold (col2im): shaders/col2im.slang

PyTorch's decomposition scatters the columns into the image with
`index_put(accumulate=True)`, which on vk sorts one int64 key per column
element. That is 14.5M keys for the case above: 4.9 s on the HD 530 and
19 s on lavapipe. In the kernel, each output element gathers the values
its blocks hold, in (kh, kw) order from zero. That is the order in which
the CPU's col2im loops add them, so the results are the CPU's bit for bit.

### Padding's backward: shaders/padgrad.slang

Reflection and replication padding's backward work the same way: each input
element gathers the outputs that read it. Along each dim these are at most
three positions (reflection) or one range (replication), found by
inverting the CPU's maps and checked against them, in ascending order. The
sum therefore goes in the CPU's order and is the CPU's bit for bit. This
covers float32 and float64; other dtypes keep the previous paths.

## Left for graph capture (#78), or for kernels of their own

- **CTC loss** (`_ctc_loss`, `_ctc_loss_backward` in `_kernels.py`): 31 and
  42 launches per time step. The alpha and beta recursions are sequential
  in time, so the natural fix is a kernel with the time loop inside, one
  workgroup per sample, as CUDA's. Graph capture would cut the host's cost
  but not the barriers. This is the largest count found, and it grows with
  the sequence.
- **Optimisers and foreach ops** (`foreach.cpp`, `optim.cpp`): 7 launches
  per parameter for a foreach Adam step, and one per parameter for the
  fused ones. A multi-tensor-apply kernel (a table of addresses and sizes)
  would make each foreach op one launch. Graph capture of a training step
  covers them too.
- **Recurrent cells** (`_thnn_fused_lstm_cell` and the GRU cell, with
  their backward): 16 launches per LSTM cell forward and 35 backward,
  called once per time step by PyTorch's RNN loop. This is #75's item 3 (a
  fused cell); the time loop itself is graph capture's case.
- **fractional_max_pool2d/3d and adaptive_max_pool2d/3d**: 500 to 1,100
  launches, building window indices per dim with gathers. pool.slang's
  max-pooling kernel with per-output window bounds would take them; they
  are cheap in time (10-30 ms on the HD 530).
- **pdist, cdist, embedding_bag**: 20-70 launches of small composites, not
  worth a kernel each.
- **The bilinear grid_sample backward's sort** (41 launches): the price of
  summing grad_input in the CPU's order.
- **Dense linear algebra** (LU, QR, eigh, SVD and the triangular solves in
  `dense.cpp`): panel and sweep loops that are iterative by nature, and
  graph capture's case.

## Tests

- tests/python/test_nn.py:
  - `test_grid_sample_bicubic_one_launch`: 2-D and 3-D; contiguous,
    strided and expanded inputs; one launch.
  - `test_grid_sample_reflection_whole_spans`: coordinates whole spans
    from the edge, bilinear and bicubic.
  - `test_upsample_bicubic_kernel`: sizes and scale factors up and down,
    align_corners, forward and backward against the CPU.
  - `test_fold_unfold`: bit for bit against the CPU and one launch;
    unbatched input; dilation, padding and stride.
  - The existing bicubic tests (non-finite coordinates and voxels, the
    unnormalization, the wide expanded axis) now reach the kernel, all but
    the last.
- tests/python/test_indexing.py: `test_padding_backward` covers 1-3 dims,
  negative padding, float32 and float64, bit for bit and in at most two
  launches.
- PyTorch's tests on the HD 530:
  - `test_nn.py -k "onv or flip or pad or fold or grid_sample"` (210
    passed);
  - `test_nn.py -k "bicubic or nterpolate or psampl"`: no failures but the
    9 that fail on main too (uint8 "nonsupported dtypes" that vk supports,
    and a CUDA launch-failure test);
  - `test_ops.py` and `test_ops_gradients.py` for `grid_sample`,
    `grid_sampler`, `fold`, `unfold`, `pad` and `interpolate`: no failures
    but the FakeTensor ones main has (PyTorch's meta kernel refuses 3-D
    bicubic).
