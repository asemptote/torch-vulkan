# PyTorch 2.15 (M36)

torch-vulkan 0.2.0 was compiled and tested against PyTorch 2.14 only. This
moves it to 2.15 (#69): `tools/torch-version` is 2.15.0 and `pyproject.toml`
asks for `torch>=2.15,<2.16`.

## One release, not two

The extension's C++ layer is compiled against one release's C++ ABI, so
supporting 2.14 and 2.15 side by side would mean a wheel per PyTorch minor
and a CI and release matrix per minor. The tests compare with the CPU's
results and errors, which differ between the two (below), so each wheel
would also need its own expected results. Nothing in 2.15 makes keeping
2.14 worthwhile: no op vk uses was removed or changed signature. 0.3.0 moves
to 2.15 only; 0.2.0's wheels stay for 2.14.

PyTorch 2.15 publishes no Python 3.10 wheels, so the project's floor is now
Python 3.11 too (`requires-python`, and pyo3's `abi3-py311`: the wheels are
tagged `cp311-abi3`).

## What changed in the CPU's results

- **float16 and bfloat16 `linspace`** (pytorch/pytorch#198415). 2.14 rounded
  the step to the output dtype and computed `start + step * i` in c10::Half
  arithmetic, rounding the product and the sum separately; 2.15 rounds only
  the endpoints to the dtype, then computes the step and each element in
  float32 (the dtype's opmath type) and rounds once. vk now does the same:
  `linspace_out` rounds the endpoints to the dtype and computes the step in
  float32, and iota.slang's float16 variant no longer rounds the product.
  bfloat16, which goes through the float32 kernel, gets its endpoints rounded
  to bfloat16 first, which it did not before. A new test compares float16
  and bfloat16 `linspace` bit for bit over ranges whose elements fall between
  16-bit values. This difference was why CI was pinned to 2.14.
- **Exception types** (pytorch/pytorch#192350, #192351, #192352). Ops
  undefined for complex inputs (`sort`, `topk`, `argmax`, `max` with `dim`,
  `maximum`, `fmax`, `aminmax`, `isposinf`, `view_as_real` of a real tensor
  ...) now raise `TypeError`; dtypes not implemented (`heaviside` and
  `rad2deg` of complex, `topk` of bool, integer `mean`, `std`, `var`,
  `norm`, and the linalg ops' integer inputs, `matrix_exp` among them) raise
  `NotImplementedError`. The messages stayed the same. vk repeats four of
  those checks in its own kernels (complex `sort`, `isposinf`/`isneginf`,
  `heaviside`, `nansum`), which now raise the new types (`nansum` also the
  CPU's message, which it did not before); the others run in PyTorch's own
  code before vk's kernels (meta functions, composite ops). A script
  comparing the CPU's and vk's exception for each op the release notes
  name, 44 cases, finds them all the same, with nothing run on the CPU.
- **The CPU dtype-error table** (`cpu_dtype_errors.inc`, M25) recorded each
  entry as `RuntimeError` or `NotImplementedError`. Its entries now carry the
  exception's type, `TypeError` included. Regenerated on 2.15, 119 of its
  1419 signatures changed from `RuntimeError` to `NotImplementedError`
  (`std` and `var` of integers, `embedding_renorm_`'s norm, integer
  `matrix_exp`); no message changed. The table's own signatures are no longer
  recorded by a run (vk raises them without the CPU), so
  `tools/dev/cpu_dtype_errors.py` now probes them again from their recorded
  examples, adding any new ones from the run directories it is given, and
  without arguments rebuilds the table from them alone.

`native_functions.yaml` between v2.14.1 and v2.15.0 adds no op that a
PrivateUse1 backend must implement: `_philox_randint_` (CPU and CUDA only,
like `_philox_uniform_` before it), `_scaled_addmm` (CUDA only) and
`_cudnn_attention_forward_no_dropout_inplace`; `arange` gained a Meta
kernel; the rest is MPS and XPU dispatch entries. `Math.h`, which
`_special.slang` is ported from, did not change.

## What 2.15's own tests found

A lavapipe run of the 36 test files on 2.15 (before the fixes below) had
414 vk-only failures that the last 2.14 run did not have, and new CPU
fallbacks. Most came from three changes on PyTorch's side:

- **Custom-op and FakeTensor alias checks.** 2.15's `torch._refs.fft`
  goes through a custom op whose output may not alias its input, and the
  FakeTensor cross-checks compare aliasing. vk's `_fft_c2c` (and r2c, c2r)
  returned the input itself where a transform does nothing (an empty or
  length-1 dimension). They now return a copy (222 failures).
- **Bicubic `grid_sample` of 5-D input** (pytorch/pytorch#194787), new on
  the CPU, CUDA and MPS. vk computes it from gathers
  (`_bicubic_grid_sample_3d` in `_kernels.py`): 2.15's `resolve_cubic_taps`
  on tensors (Keys coefficients, the four taps per axis around the
  unclipped index, padding, a dropped tap reading nothing), then the 64 taps
  of every output point in one gather through the input's strides (an
  expanded input is not copied), each term formed as the CPU's, in float32
  or float64, and summed. (A first version, one gather per tap, made
  PyTorch's gradgrad test of it run for over 20 minutes on lavapipe.) Its
  backward is autograd of the same composite, as for 2-D bicubic.
  PyTorch's meta function refuses 5-D bicubic on devices other than the CPU
  and CUDA, so `_layout.py` no longer runs FakeTensor for `grid_sampler_3d`
  (it logged an error on every first call) and the two FakeTensor tests are
  in the runner's `EXPECTED` list.
- **Tests that used to be CUDA-only** now run on every accelerator
  (`@onlyAccelerator`). Most exposed differences in vk worth fixing:
  `zero_` writes zero bits, as the CPU's memset, so dtypes with no fill
  kernel (`bits8`, `float4_e2m1fn_x2`) can be zeroed; float to int32 casts
  give int32 min for NaN, infinities and out-of-range values, as x86 does
  (the GPU saturated), and float64 to int8/int16/uint8 their low bits;
  `exponential_` never returns 0 (CUDA keeps -log(1 - u) at least half the
  dtype's epsilon); `multinomial` checks devices first; `randperm` with n =
  0 or a generator of another device type; `cat` gives a resized `out=` the
  inputs' memory format; complex `x ** 2`, `** 3` and `** -2` are products,
  as the CPU computes them (`(-1 + inf i) ** 2` is `-inf - inf i`). Some
  are CUDA-only by nature (CUDA graphs, peak-memory statistics, streams,
  DataParallel, SyncBatchNorm's CUDA-only ops) and are in `EXPECTED`.

The rest were changes in the CPU's behaviour:

- `clamp` and `hardtanh` saturate integer scalar bounds outside the dtype's
  range (pytorch/pytorch#193169): a bound that cannot change an element is
  dropped, the opposite raises; `hardtanh` no longer rejects negative
  bounds for unsigned types. vk follows (`clamp_bound` in elementwise.cpp),
  and clamps uint16, uint32 and uint64 (new on the CPU, #189953) on the GPU
  as the signed type that orders them the same way (uint16 and uint32
  widened, uint64 with its top bit flipped). This turned up an older bug:
  an int32 or 8/16-bit `clamp` with only one bound used int64's extremes for
  the missing one, whose low words the 32-bit kernel read as 0 and -1, so
  `torch.clamp(int32_tensor, min=10)` gave -1 everywhere.
- Padding and pooling checks moved to shared headers with new messages
  (`Padding.h`, `PoolingChecks.h`); vk's checks follow them, and the
  reflection and replication pad backwards check `grad_output`'s channels.
- `aminmax` raises `ValueError` for overlapping `min` and `max` outputs;
  `histc`'s dtype error names `torch.histogram`; `i0` and the other special
  functions now refuse a partially overlapping `out=`, as every CPU unary op
  does (2.15's test lists `i0`).
- `torch.equal` of uint16/32/64 and `addmv` with an empty matrix (any dtype)
  ran on the CPU; they run on the GPU now.

## Tooling

`tools/cluster/setup.sh` installed torch 2.14.1 and checked out PyTorch's
tests at 2.14.1's commit, both written into the script. It now installs
`tools/torch-version`'s release and checks out the tests at that torch's
own commit (`torch.version.git_version`), re-checking them out when it
changes, so a version bump needs no edit there. It also works from a
checkout other than `~/torch-vulkan`, as `build.sh` does.

## Testing

- tests/python with `TORCH_VK_STRICT=1`: 1607 passed, 9 skipped on the Intel
  HD 530 (Mesa ANV) and on lavapipe.
- The strict OpInfo sweep: 689 ops, all on the GPU, none on the CPU, on both.
- `tools/dev/cpu_dtype_errors.py --check --all`: 1419 signatures, none
  mismatched.
- PyTorch's 36 test files on lavapipe (121,587 tests collected): the full
  run gave 107,003 passed and 418 vk-only failures before the fixes; the
  439 failing or falling-back tests run again after them leave 61 vk-only
  failures and 2 crashes.

What is left:

- Tests 2.15 runs on accelerators for the first time that show gaps older
  than 2.15: sparse compressed tensors (`bmm`, `baddbmm`,
  `_sparse_mm_reduce_impl`, `sampled_addmm`'s checks, `addmm` into a dense
  result, the device-mismatch messages of the constructors; 37 tests), fused
  Adam/AdamW with bfloat16 parameters and float32 state (9), 2-D bicubic
  `grid_sample` with zeros padding beside a non-finite voxel (PyTorch's
  decomposition multiplies it by a zero weight; 2-D and 3-D then differ),
  nearest `grid_sample`'s rounding at half-way coordinates, a 2^53 + 1 wide
  axis, autocast's dtype under checkpointing, a Python-dispatch deadlock
  test, and a CPU-scalar `pow` case.
- Fallbacks in those tests: `_masked_softmax_backward`, `abs` of
  float8_e5m2 and `eq`/`copy_` of float8_e8m0fnu, and CPU errors for fused
  Adam's mixed dtypes, `_validate_compressed_sparse_indices` and
  `baddbmm`'s `out=` dtype.
- The three `test_tensor_lr_*_LBFGS` failures and `grid_sampler_3d`'s
  gradgrad crash were there with 2.14.
