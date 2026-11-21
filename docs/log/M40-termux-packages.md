# Termux packages (M40)

Releases now attach wheels for Termux on Android, so a phone needs no Rust,
clang or on-device build (#82):

    pkg install python-torch vulkan-loader-android
    pip install <torch_vulkan-X.Y.Z-cp311-abi3-android_24_arm64_v8a.whl URL>

(`armeabi_v7a` for a 32-bit userspace). Building from source with
`tools/build-termux.sh` and the release's SPIR-V stays the fallback, for any
other `python-torch`; it now makes the same kind of wheel.

- **Built in Termux itself, in CI.** The release workflow's `wheel-termux`
  jobs run the official `termux/termux-docker` image (a Termux userland with
  Android's bionic linker, no emulation) on GitHub's arm64 Linux runners,
  free for public repositories. They install Termux's own `python-torch`,
  `rust` and `clang` with apt, check that `python-torch` is
  `tools/torch-version`'s release, and run `tools/build-termux.sh` against
  the spirv job's kernels (`VKCORE_PREBUILT_SPIRV`; Termux has no slangc).
  The runners' Neoverse N2 cores also execute 32-bit ARM (`lscpu`: "CPU
  op-mode(s): 32-bit, 64-bit"; the Azure kernel has `CONFIG_COMPAT`), so the
  arm image runs natively too, and both wheels come from CI. Two details:
  32-bit bionic keeps thread IDs in 16 bits, so the job lowers the runner's
  `kernel.pid_max` to 65535 (as termux-docker's README says), and the image
  needs the `personality()` syscall (`--security-opt seccomp=unconfined`).
- **Termux's libtorch links `libvulkan.so`** (it is built with PyTorch's old
  mobile Vulkan backend) without depending on a Vulkan loader package, so
  `import torch` fails in a bare Termux. Users get one from
  `vulkan-loader-android`, which they want anyway: it makes Termux's
  `libvulkan.so` a link to Android's, which reaches the phone's GPU, where
  Termux's default `vulkan-loader-generic` finds only Mesa's drivers
  (llvmpipe on the CPU). It conflicts with `vulkan-loader-generic`, so on a
  device where other packages need Mesa's loader (ffmpeg, mesa), a directory
  with `libvulkan.so -> /system/lib64/libvulkan.so` (`/system/lib` on
  32-bit) first on `LD_LIBRARY_PATH` does the same for torch-vulkan's
  processes only.
- **A wheel rather than a `.deb`.** A `.deb` depending on
  `python-torch (= 2.15.0)` would make apt hold back or remove it at a
  `python-torch` upgrade, but it is two steps to install (download, then
  `pkg install ./file.deb`), its files would sit in one Python's
  site-packages directory, and it needs its own control files and a second
  test path. A wheel installs with one `pip install <URL>`, as on Windows and
  Linux, and pip handles uninstalling and upgrades. Termux's pip accepts
  `cp311-abi3-android_24_arm64_v8a` and `..._armeabi_v7a`
  (`pip debug --verbose` lists `abi3` tags for API levels up to 24;
  `sysconfig.get_platform()` is `android-24-arm64_v8a`, API level 24 being
  Termux's minimum). `tools/termux_wheel.py` writes the wheel with the
  standard library (a zip with `METADATA`, `WHEEL` and `RECORD`), from the
  module build-termux.sh built, rather than through maturin: build-termux.sh
  already calls cargo directly, and the tags are spelled out rather than
  left to maturin's Android support.
- **Tied to one python-torch, failing clearly otherwise.** The compiled
  module links libtorch's C++ API, whose ABI changes between releases, so a
  Termux wheel requires exactly the torch it was compiled against
  (`Requires-Dist: torch==2.15.0`). Termux's `python-torch` registers itself
  with pip as `torch 2.15.0`, so pip accepts it; with another torch pip looks
  for torch 2.15.0 on PyPI, which has no Android builds, and fails. pip
  cannot stop a later `pkg upgrade` of `python-torch`, so build-termux.sh
  also writes `_build_info.py` (`TORCH_VERSION = "2.15.0"`), and importing
  torch_vulkan under any other torch raises an ImportError naming both
  versions, before the compiled module is loaded (a mismatched libtorch
  could otherwise fail to resolve symbols, or load and crash). The workflow
  checks this by importing with `torch.__version__` set to another release.
  A Termux rebuild of the same PyTorch release (a new package revision) keeps
  `torch.__version__` and is accepted; its C++ ABI is the same release's.
- **Testing.** In a second, fresh container the workflow installs
  `python-torch`, Mesa's lavapipe (`mesa-vulkan-icd-swrast`) and the wheel
  with pip, then runs `tools/release_smoke.py` (strict) and tests/python
  outside the source tree. On aarch64 the smoke test passes (float64 QR
  included: lavapipe has float64) and tests/python gives 1870 passed, 14
  skipped in 5 minutes on the runner's 4 cores. Not run there, each confirmed
  by a diagnostic run in the container:
  - lavapipe on aarch64 flushes subnormals to zero, float32 and float64
    (`x * 1.0` of 5e-324 or 1e-40 is 0; on x86 it keeps them), which the
    float64 `exp`, `exp2`, `tanh` accuracy tests and the correctly rounded
    float64 `sqrt` test see;
  - the aarch64 CPU, the tests' reference, quantises NaN and infinities to
    other integers than x86's (`quantize_per_tensor`,
    `fake_quantize_per_tensor_affine_cachemask`), and returns 0 for
    `torch.dot(a.conj(), b.conj())` of complex64 vectors (vk's result is the
    sum of the products);
  - Termux's PyTorch has KleidiAI, so on aarch64
    `_dyn_quant_pack_4bit_weight` packs weights in KleidiAI's layout, which
    vk's `_dyn_quant_matmul_4bit` does not read (it raises): the one
    difference a 64-bit Termux user could meet, when 4-bit weights packed on
    the CPU are moved to "vk".

  The CPU's complex `log2(0)` is -inf+nanj on Android as on 32-bit ARM, and
  that test now expects it there too. On 32-bit ARM, lavapipe's own shaders
  fault (M17: an aligned NEON store to 8-byte aligned memory), so the arm
  wheel is only installed, imported and refused under another torch in CI;
  its GPU check is a phone's (RELEASING.md). The CI-built arm wheel,
  installed with pip into a venv on the Galaxy A13, passes the strict smoke
  test on the Mali-G57 (matmul error 5.7e-7, FFTs 1.8-2.5e-7, eigvalsh
  1.1e-6, 20 Adam steps 2.340 -> 0.610, no fallbacks) and tests/python:
  1402 passed, 493 skipped (float64), in 2.6 minutes.
- **Build times**: in CI the Termux build takes 2.5 minutes (cargo on 4
  cores), each Termux job about 6-10 minutes with the tests; on the A13 the
  same build takes 15 (M17).
- **Termux upgrades**: a release's Termux wheels match the `python-torch`
  Termux shipped when it was tagged (which the workflow requires to be
  `tools/torch-version`'s). When Termux moves to a newer PyTorch, users who
  upgrade get the ImportError until a torch-vulkan release for that PyTorch,
  or build from source; `pkg upgrade` itself does not know about the pip
  package. A TUR or termux-packages recipe, which Termux would rebuild with
  its torch, is the longer-term answer (#82).
