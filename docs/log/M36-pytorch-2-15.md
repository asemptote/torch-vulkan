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

## Tooling

`tools/cluster/setup.sh` installed torch 2.14.1 and checked out PyTorch's
tests at 2.14.1's commit, both written into the script. It now installs
`tools/torch-version`'s release and checks out the tests at that torch's
own commit (`torch.version.git_version`), re-checking them out when it
changes, so a version bump needs no edit there. It also works from a
checkout other than `~/torch-vulkan`, as `build.sh` does.

## Testing

RESULTS
