# Releasing torch-vulkan

The steps for a release, in order. Each step's result is checked before the
next one starts.

## 1. The code is ready

- CI on `main` is green, and the release's milestone issues are closed.
- PyTorch's test files have been run with fallback recording
  (`tools/pytorch_tests.py`, sharded with `tools/cluster/`): no test computes
  or raises on the CPU, apart from failures that are listed with a reason in
  the script's `EXPECTED` table. A targeted rerun of the tests that fell back
  in the last full run is enough when that run's other results stand.
- The strict OpInfo sweep (`python tools/opinfo_coverage.py --strict`) passes.

## 2. Devices

Run `tests/python` (with `TORCH_VK_VALIDATE=1` where the validation layer is
installed) on each kind of device the README lists as tested:

- lavapipe: CI;
- an NVIDIA GPU (and one with cooperative matrices, for the float16 GEMM);
- Intel under both Windows and Mesa (Linux);
- an Arm Mali phone under Termux, built with `tools/build-termux.sh` from the
  release candidate's SPIR-V, as a Termux user would.

Record anything new in the README's hardware table or limitations.

## 3. Documentation

- README: the PyTorch version, what runs on the GPU, the hardware table,
  benchmark figures (re-measured, with the machine named), limitations, and
  the installation commands' version numbers.
- A release notes draft: what's new since the previous release, taken from
  the `docs/log/` files added since then.

## 4. Version and tag

1. Set `version` in `pyproject.toml`, and `version` and `date-released` in
   `CITATION.cff`, and merge that to `main`.
2. Tag the merge: `git tag -a vX.Y.Z -m "torch-vulkan X.Y.Z"` and push the
   tag. The release workflow builds the Windows and Linux wheels and the
   SPIR-V bundle for Termux, tests the Linux wheel on lavapipe outside the
   source tree, and creates a **draft** release. It fails if the tag doesn't
   match `pyproject.toml`'s version.
3. Check the draft: three assets (two wheels, `torch-vulkan-spirv-vX.Y.Z.tar.gz`),
   and install a wheel on a real GPU.

## 5. Publish

1. Paste the release notes into the draft and publish it.
2. Announce where appropriate.

A tag that was never published can be deleted and redone; a published release
is not changed afterwards (fixes go in the next patch release).
