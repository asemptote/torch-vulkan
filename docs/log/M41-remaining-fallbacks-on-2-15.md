# The last CPU fallbacks and edge cases on PyTorch 2.15 (M41)

Issue #91. After the move to PyTorch 2.15 (M36), a re-run of the failing
tests of PyTorch's 36 test files on lavapipe still reached the CPU in a few
places, and left some failures that were not about sparse tensors (#89) or
fused Adam with mixed dtypes (#90). This closes them, as a prerequisite for
running without the CPU fallback by default (#101).

## CPU fallbacks

- **`_masked_softmax_backward`** (MaskedTensor's softmax): a composite in
  `_kernels.py`, as the CPU's `host_softmax_backward` with a mask: along
  the dim, s = the sum of grad * out over the unmasked positions, the
  gradient out * (grad - s) there and 0 where masked. A row with every
  position masked (whose forward output is NaN) gets 0, not NaN, as on the
  CPU.
- **float8**. PyTorch 2.15's `test_isinf_isfinite_float8` covers every
  float8 dtype, `float8_e8m0fnu` included, which vk did not know: its casts
  and comparisons went to the CPU. The CPU's float8 support is narrow: of
  the element-wise ops it has only `abs`, `mul`, the comparisons `eq`/`ne`,
  `isnan`/`isinf`/`isfinite`, `where` and `any`/`all` (probed on 2.15; most
  others raise `"..._cpu" not implemented for 'Float8_e5m2'`). vk now runs
  each of these on the GPU:
  - `float8_e8m0fnu` (a power of two, 2^-127 to 2^127, or NaN at 0xff; no
    sign and no zero) joins the conversion tables of M25: each code's value
    from the CPU's own conversion, and float32 to code by a search over the
    thresholds where the CPU's conversion changes code. The code is all 8
    bits, and the float's sign is ignored, as c10 does.
  - `abs` through float32 (exact) and back; e8m0fnu, which the CPU has no
    `abs` for, raises the CPU's error.
  - `mul` of two float8 tensors through float32, where the product of two
    float8 values is exact, rounded once to the result, as c10's arithmetic.
    With a Python scalar the CPU raises (`mul_cpu_reduced_float`); so does
    vk.
  - comparisons, `isnan` and `any`/`all` through float32, for e8m0fnu too.
    (`isinf` and `isfinite` are composites of these and `abs`.)
- **CPU calls that only raised**: `baddbmm` with an `out=` of another dtype
  went to the CPU for its meta function's error; vk raises the two dtype
  checks itself, with the CPU's messages. The other two in that run,
  `_fused_adam(w)_` with bfloat16 parameters and float32 state, and
  `_validate_compressed_sparse_indices`, belong to #90 and #89.

## Grid sampling

- **2-D grid sampling runs on vk's own kernels.** `grid_sampler_2d` went
  through PyTorch's decomposition, which unnormalises a grid coordinate as
  `g * mul + ofs`, rounding differently from the CPU's `(g + 1) * scale -
  0.5`. At half-way coordinates nearest sampling then rounded differently
  from 3-D (vk's `gridsample.slang`, the CPU's arithmetic), which
  `test_grid_sample_nearest_neighbor_rounding_mode_consistency` compares
  bit for bit. The decomposition is now on `_decomp.py`'s deny list, and
  `grid_sampler_2d` runs `gridsample.slang` (which already took 2-D grids,
  for the backward) for bilinear and nearest. The CPU's 2-D kernel scales
  by `size / 2` and the 3-D one by `size` then halves; a factor of two is
  exact, so the two give the same float, and the shader's one formula
  matches both.
- **2-D bicubic follows the CPU's taps.** The decomposition gathered a
  dropped tap's voxel (zeros padding, the tap outside the input) and
  multiplied it by a zero weight, so an infinite voxel next to the border
  gave NaN where the CPU gives 0 (`test_grid_sample_3d_bicubic_non_finite`
  compares 2-D with 3-D). M36's 3-D composite (`resolve_cubic_taps` on
  tensors, a dropped tap reading nothing) now covers both ranks
  (`_bicubic_grid_sample`); in 2-D it sums in the CPU's order, four values
  along x per row and then the rows along y. The bicubic backward is
  autograd of the same composite at both ranks.
- **The CPU's unnormalisation rounds once.** With the composite, one
  element of 943,920 in PyTorch's `test_compare_cpu` for 2-D bicubic
  float32 (0-255 samples, align_corners off) was 0.0016 off, on a value of
  -0.5 where the 16 taps cancel. Its cause was the source coordinate: the
  CPU computes `(g + 1) * (size / 2) - 0.5` (and in 3-D `((g + 1) * size -
  1) / 2`), which its compiler contracts into a fused multiply-add, one
  rounding; vk rounded the product and the difference separately, and one
  ulp of a coordinate near 128 moves such a sum by 0.002. The composite now
  computes the coordinate rounded once from separately rounded operations
  (`_fused_multiply_add`: Dekker's two-product and Knuth's two-sum, the
  errors added back before the last rounding), which gives the
  float64-exact value for 100,000 random coordinates at five sizes, and
  every grid_sample OpInfo sample is within the test's tolerance. The
  correction carries no gradient, so autograd still sees `(g + 1) * size /
  2`. (`gridsample.slang` keeps two roundings: its bilinear and nearest
  results are within the tolerances, and no test showed a difference.)
- **A very wide expanded axis.** `test_grid_sample_3d_bicubic_last_voxel_of_a_wide_axis`
  samples the last voxel of a 1-element volume expanded to 2^53 + 1. The
  indexing kernel (`index.slang`) bounds indices in int32, and took the
  size's low word, so any index into a dim wider than int32 failed; a
  dimension that wide exists only with stride 0 (vk's offsets are 32-bit).
  `index_offsets` now bounds such a dim at int32's top, so indices below
  2^31 are in range (offset 0), and the bicubic composite reads an
  expanded axis at index 0, whatever the tap's index.
- `grid_sampler` with a CPU grid for a vk input raises the CPU's device
  error from vk's own check (the decomposition used to).

## Autocast

`test_checkpointing_non_reentrant_autocast` runs matrix products under
`torch.autocast(device, dtype=torch.bfloat16)` and expects bfloat16 results.
vk had no autocast policy: `torch.vk.get_amp_supported_dtype()` returned
float32 only, so `torch.autocast("vk", ...)` warned and did nothing.
`csrc/autocast.cpp` now registers PyTorch's policy for CUDA and XPU (the op
lists `ATen/autocast_mode.h` shares between backends) for the
AutocastPrivateUse1 key: products, convolutions and the recurrent cells in
the autocast dtype, ops that lose accuracy in 16 bits in float32, ops with
several inputs promoted to the widest; `binary_cross_entropy` is refused, as
on CUDA. `get_amp_supported_dtype()` lists float16, bfloat16 and float32.
The default dtype is PyTorch's for PrivateUse1, float16. This is a change in
behaviour: code that used `torch.autocast("vk")` and got float32 now gets
16-bit products (as it would on CUDA).

## Recorded in `EXPECTED`

- `test_lookup_does_not_deadlock_with_python_dispatch`: its subprocess
  script hard-codes `device="cuda"`. With "vk" in its place the backward
  completes; about one run in ten then aborts at interpreter exit, in
  PyTorch's autograd device thread (no vk frames on the stack): it drops the
  last reference to the graph task, which holds a Python object, after
  finalisation has begun.
- `test_cpu_tensor_pow_device_scalar_tensor`: an in-place `pow_` of a 0-dim
  CPU tensor by a vk tensor raises the device error, as on CUDA and XPU; the
  test expects that error from those two devices by name only.
- `test_tensor_lr_num_dim_*_LBFGS`: the `maximize_strong_wolfe` input
  maximises a quadratic, so the weights grow without bound (1.5e19 after two
  steps), and the runs with a float lr and a tensor lr, which round
  differently, part ways. A replica of the test on the CPU does the same for
  other inputs (seed 19: 70% apart); vk's generator draws other numbers for
  the test's seed than the CPU's.

## Testing

(see the pull request)
