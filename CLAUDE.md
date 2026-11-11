# Notes for coding agents

Read [CONTRIBUTING.md](CONTRIBUTING.md) first, especially its design
principles: all numerical work on "vk" tensors runs on the GPU in this
project's Slang kernels. Do not route work to LAPACK, BLAS or any other CPU
or vendor library, and do not recommend `TORCH_VK_LINALG=cpu` as a fix: it is
a testing reference. When a GPU kernel is slow, the fix is a better GPU
kernel or algorithm.

- Build: `maturin develop --release` (with `--uv` on Windows) from the
  project's virtual environment; `slangc` comes from `SLANGC`, `VULKAN_SDK` or
  `PATH`.
- Checks, cheapest first: `python -m pytest -q`; the PyTorch tests for the op
  (`python tools/dev/pt.py test_linalg.py -k ...`);
  `python tools/opinfo_coverage.py --jobs 4`; clippy and `cargo test` for Rust
  changes; everything again with `TORCH_VK_VALIDATE=1`.
- Kernels: `crates/vkcore/shaders/*.slang`, templated per dtype; aten entry
  points: `crates/pyvk/csrc`. Design decisions go in a new file,
  `docs/log/MNN-topic.md` (see `docs/development-log.md`).
- Don't change the README's version numbers, benchmark figures or release
  notes: the maintainer updates them at release time.
- Docs use Australian spelling (-ise, "licence" as the noun).
