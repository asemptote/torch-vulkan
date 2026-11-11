# Contributing to torch-vulkan

Thanks for your interest. Bug reports, results from GPUs and drivers not yet
tested, and pull requests are all welcome.

## Reporting a bug

Open an issue with:

- what you ran (a minimal script if possible) and what happened, including the
  full traceback;
- your GPU and driver (`vulkaninfo --summary`), operating system, Python and
  PyTorch versions, and the torch-vulkan commit;
- whether it also happens with `TORCH_VK_VALIDATE=1` (and any validation
  messages it prints), and whether the CPU gives the result you expected.

A wrong result is a bug even when it is small: the goal is to match PyTorch's
CPU results within the tolerances PyTorch's own tests use. Results that differ
because a driver is less accurate than the Vulkan specification promises, or
because a GPU lacks a feature, are worth reporting too.

## Asking a question

Open an issue or a GitHub discussion. There is no chat channel.

## Design principles

- **All numerical work on "vk" tensors runs on the GPU**, in this project's
  Slang kernels. There are no BLAS, LAPACK, FFT or other vendor libraries,
  and no routing of work to them on the CPU. Algorithms may follow LAPACK's
  (blocked factorisations, Householder reductions, tridiagonal solvers); the
  code that runs them is ours, on the GPU.
- **The host only** builds small tables that do not grow with the problem's
  work (FFT twiddles, rotation schedules), makes control decisions from a few
  scalars read back (convergence, pivot errors), and reproduces PyTorch's
  errors where PyTorch rejects an input.
- **PyTorch's CPU fallback is a safety net, not a design choice.** It is
  counted by `torch_vulkan.fallback_counts()` and is an error under
  `TORCH_VK_STRICT=1`; an op that falls back is a bug or a listed gap.
  `TORCH_VK_LINALG=cpu` is a reference for testing and comparison, not a
  performance option: a slow GPU kernel is fixed on the GPU.
- **Match PyTorch's CPU results** within the tolerances PyTorch's own tests
  use, on every supported GPU, including those without float64 or int64.

## Making a change

1. Build as described in the README's [Building](README.md#building) section.
2. Run the checks, cheapest first:
   - `python -m pytest -q` (the project's tests, `tests/python`);
   - the PyTorch tests that cover the op you changed, for example
     `python tools/dev/pt.py test_linalg.py -k "solve"`;
   - `python tools/opinfo_coverage.py --jobs 4` for the OpInfo sweep;
   - `cargo clippy --release --all-targets -- -D warnings` and
     `cargo test --release -p vkcore` if you changed Rust code;
   - all of the above with `TORCH_VK_VALIDATE=1` before opening a pull
     request: the validation layer must report nothing.
3. Add a test that fails without your change, comparing `"vk"` against the
   CPU (see `tests/python/_util.py`).
4. Open a pull request. CI builds the project on Linux and runs every test on
   lavapipe (Mesa's CPU Vulkan driver) with the validation layer.

New kernels are Slang files in `crates/vkcore/shaders`, templated per dtype,
and their aten entry points live in `crates/pyvk/csrc`. Lavapipe is a good
second device: it finds missing barriers and out-of-range loads that real GPUs
hide. The [development log](docs/development-log.md) records the design
decisions and their reasons, one file per milestone in `docs/log/`; a change
that deserves a design note adds a new `docs/log/MNN-topic.md` rather than
editing an earlier one.

`tools/cluster/` runs PyTorch's test files in shards on a PBS cluster; it
needs a `tools/cluster/cluster.env` (see `cluster.env.example`) and is
optional.

## Pull requests

- Keep a pull request to one topic. Leave the README's version numbers,
  benchmark figures and release notes to the maintainer, who updates them at
  release time; other README changes are welcome.
- CI runs on the pull request merged with `main` as it was when the run
  started. If `main` has since changed files the pull request also changes, it
  is rebased and CI runs again before merging; otherwise it is merged as it is.
- Pull requests are merged with a merge commit, so the history shows how the
  work was done.

## Upgrading PyTorch

The extension's C++ layer is compiled against one PyTorch release, and the
tests compare with that release's CPU results. That release is named in one
place, `tools/torch-version`; `pyproject.toml` allows only its minor version,
and CI's `tools/check_torch_version.py` fails if the two disagree. To move to
a new release:

1. Update `tools/torch-version` and the torch requirement in
   `pyproject.toml`, and rebuild.
2. Regenerate the files derived from PyTorch's own code:
   `python tools/port_special.py` (`shaders/_special.slang`, from ATen's
   `Math.h`) and `python tools/dev/cpu_dtype_errors.py` (the CPU's dtype
   errors that vk raises itself, from a fallback-detail run). Review their
   diffs.
3. Run `tests/python`, the strict OpInfo sweep and PyTorch's test files
   (`tools/pytorch_tests.py`); differences in the CPU's results between
   releases (2.15's float16 `linspace`, for example) show up here.
4. Update the PyTorch version in the README's installation and building
   sections.

## Maintenance and support

torch-vulkan is maintained by one person, in their own time. Issues and pull
requests are read and answered on a best-effort basis, usually within a week
or two; there is no guaranteed response time. Design decisions are made by the
maintainer, discussed in issues first for anything large (a new dependency, a
change to the device's behaviour or to the public API).

## Licence

By contributing, you agree that your contributions are licensed under the
project's terms: MIT OR Apache-2.0, at the user's option (see the README's
[Licence](README.md#licence) section).
