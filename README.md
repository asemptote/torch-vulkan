# torch-vulkan

[![ci](https://github.com/asemptote/torch-vulkan/actions/workflows/ci.yml/badge.svg)](https://github.com/asemptote/torch-vulkan/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#licence)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange)

A Vulkan compute backend for PyTorch aimed at **scientific** workloads:
float64, complex numbers, FFTs, dense and sparse linear algebra and special
functions, not just neural networks.

PyTorch is increasingly used for scientific computing, but its GPU backends
depend on vendor stacks (CUDA, ROCm, Metal), and its former Vulkan backend
targeted mobile neural-network inference and is no longer maintained.
torch-vulkan is a native PyTorch device, `"vk"`, built on portable Vulkan
compute. It runs the scientific side of PyTorch on integrated, discrete and
phone GPUs without vendor libraries. Where Vulkan's precision rules are looser
than PyTorch's numerical expectations, it uses driver-independent
implementations, and it is validated against PyTorch's own test suites on
Intel, NVIDIA, Arm Mali and Mesa's lavapipe.

> [!WARNING]
> **Experimental.** APIs, coverage and performance will change before 1.0.

```python
import torch
import torch_vulkan             # registers the "vk" device

torch.manual_seed(0)            # seeds the "vk" generator too
x = torch.randn(1_000_000, dtype=torch.float64, device="vk")  # Philox on the GPU
y = (x * 2 + 1).relu()          # Vulkan kernels, submitted in batches
print(y[:3], y.sum())           # sum is a Vulkan reduction
torch_vulkan.fallback_counts()  # ops that ran on the CPU instead, if any: {}
```

## What runs on the GPU

- **Element-wise ops** in float32, float64, complex64 and complex128:
  arithmetic, comparisons, logical ops, transcendental and special functions
  (Bessel, incomplete gamma, zeta/polygamma, erfcx, orthogonal polynomials),
  activations, `where`, `lerp`/`addcmul`/`addcdiv`. float16 and bfloat16 are
  computed in float32 and rounded once; 8/16-bit integers are computed in int32.
- **Reductions and scans**: sum, mean, prod, max/min, argmax, var/std, norms,
  any/all, cumsum/cumprod, cummax/cummin, logcumsumexp.
- **Matrix products**: mm, bmm, matmul, addmm, baddbmm, addmv, dot, `nn.Linear`,
  einsum; integer and 16-bit products too.
- **FFTs**: all of `torch.fft`, any length.
- **Dense linear algebra**, real and complex, any size and batch, without
  LAPACK: solve, inv, det, slogdet, LU, Cholesky, QR, eigh, SVD, least
  squares, matrix_exp, triangular solves, general eig, LDL.
- **Sparse COO and CSR tensors**: construction, coalescing, conversions,
  arithmetic and sparse @ dense products.
- **Indexing and data movement**: advanced indexing, `index_put_`, gather,
  scatter (with reductions), `masked_select`, `nonzero`, cat, flip, roll,
  padding, arange/linspace/eye, `index_add`, `bincount`, `histc`.
- **Sorting and statistics**: sort, topk, kthvalue, median, mode, unique,
  searchsorted.
- **Random numbers**: rand, randn, randint, bernoulli, multinomial, randperm,
  dropout, poisson, gamma, binomial, Dirichlet.
- **Neural-network ops** with their backward passes: convolution (1-D to 3-D,
  transposed), pooling, normalisation, softmax, losses, upsampling, grid
  sampling, embeddings, attention.

All numerical work is meant to run on the GPU (see the design principles in
[CONTRIBUTING.md](CONTRIBUTING.md#design-principles)). Anything that does not
run there yet falls back to the CPU through PyTorch's fallback and is counted
in `torch_vulkan.fallback_counts()`; `TORCH_VK_STRICT=1` turns those fallbacks
into errors. In the OpInfo sweep (`tools/opinfo_coverage.py`, PyTorch's
sample inputs for every op compared against the CPU) every op runs fully on
the GPU and matches the CPU. PyTorch's own device-generic test files
(`test_ops.py`, `test_torch.py`, `test_linalg.py`, `test_nn.py`,
`test_sparse.py` and others, run by `tools/pytorch_tests.py`) pass on `"vk"`.
Every remaining failure also fails on the CPU or has a written reason in that
script's `EXPECTED` table.

## A scientific example

```python
import torch
import torch_vulkan

torch.manual_seed(0)
dev = "vk"

# Dense linear algebra in float64, on the GPU (no LAPACK).
n = 512
A = torch.randn(n, n, dtype=torch.float64, device=dev)
A = A @ A.mT + n * torch.eye(n, dtype=torch.float64, device=dev)  # symmetric positive definite
b = torch.randn(n, dtype=torch.float64, device=dev)
x = torch.linalg.solve(A, b)
print("residual:", torch.linalg.norm(A @ x - b).item())          # ~2.5e-14
L = torch.linalg.cholesky(A)
w = torch.linalg.eigvalsh(A)

# FFTs of any length, complex128 included.
s = torch.randn(1000, dtype=torch.complex128, device=dev)
print("fft round trip:", (torch.fft.ifft(torch.fft.fft(s)) - s).abs().max().item())  # ~3e-15

# Special functions.
q = torch.linspace(1.5, 4.0, 6, dtype=torch.float64, device=dev)
print("zeta:", torch.special.zeta(q, 1.0).cpu())

print("CPU fallbacks:", torch_vulkan.fallback_counts())          # {}
```

## Hardware

torch-vulkan needs a Vulkan 1.1 driver with buffer device addresses, scalar
block layout, timeline semaphores and the Vulkan memory model (core in 1.2,
extensions on 1.1). float64 tensors need `shaderFloat64`; on GPUs without it,
creating a float64 or complex128 tensor raises a `TypeError`, as on Apple's
MPS. int64 is emulated with 32-bit words everywhere, so `shaderInt64` is not
needed.

Tested on:

| Device | Driver | Notes |
|---|---|---|
| Intel UHD Graphics 620 | Intel, Windows | main development machine; unified memory, no copies between CPU and GPU |
| NVIDIA H100 | NVIDIA, Linux | PyTorch's test files |
| Arm Mali-G57 (Galaxy A13 5G) | Arm, Android (Termux) | Vulkan 1.1, no float64, no int64 |
| Mesa lavapipe | Mesa, Linux | CI, with the validation layer; PyTorch's test files |

Apple GPUs (through MoltenVK) are untested: MoltenVK provides buffer device
addresses only on Metal 3 devices, and Apple GPUs have no float64, so at
best they would run float32 work.

## Installing

Wheels for Windows and Linux (x86_64, Python 3.10 or newer) are attached to
each [release](https://github.com/asemptote/torch-vulkan/releases). They need
PyTorch 2.14, which they are compiled against, and a Vulkan driver:

```sh
pip install "torch==2.14.*"
# Windows
pip install https://github.com/asemptote/torch-vulkan/releases/download/v0.1.0/torch_vulkan-0.1.0-cp310-abi3-win_amd64.whl
# Linux
pip install https://github.com/asemptote/torch-vulkan/releases/download/v0.1.0/torch_vulkan-0.1.0-cp310-abi3-manylinux_2_28_x86_64.whl
```

On Android (Termux), build from source with the release's prebuilt kernels:
extract `torch-vulkan-spirv-v0.1.0.tar.gz` and run `tools/build-termux.sh`
with that directory (the script describes the setup). Other platforms build
from source as below.

## Building

Requirements: Rust (stable), a C++20 compiler, Python 3.10 or newer, PyTorch
2.14, the [Slang](https://shader-slang.org)
compiler `slangc` (in the Vulkan SDK, or from Slang's releases), and a Vulkan
driver. `build.rs` finds `slangc` through `SLANGC`, then `VULKAN_SDK`, then
`PATH`. The extension is compiled against the PyTorch of the environment it is
installed into, so rebuild it after upgrading PyTorch. The tests and CI use
PyTorch 2.14.1; with 2.15, float16 `linspace` differs from the CPU's by a few
units in the last place.

Linux (as the CI does; lavapipe gives a Vulkan driver on any machine):

```sh
sudo apt-get install mesa-vulkan-drivers libvulkan1 vulkan-tools
mkdir -p ~/slang && curl -sSL https://github.com/shader-slang/slang/releases/download/v2026.20/slang-2026.20-linux-x86_64.tar.gz | tar -xz -C ~/slang
export SLANGC=~/slang/bin/slangc
python -m venv .venv && . .venv/bin/activate
pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
pip install numpy pytest maturin expecttest hypothesis
maturin develop --release
python -m pytest -q
```

Windows (Visual Studio Build Tools with C++, the Vulkan SDK, and
[uv](https://docs.astral.sh/uv/)):

```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest maturin expecttest hypothesis
.\.venv\Scripts\maturin develop --release --uv
.\.venv\Scripts\python -m pytest -q
```

Android (Termux, 32-bit ARM): see `tools/build-termux.sh`; Termux has no
`slangc`, so it uses kernels compiled elsewhere, such as a release's
`torch-vulkan-spirv-*.tar.gz`.

## Python API

```python
import torch_vulkan

torch_vulkan.is_available()     # a usable Vulkan device was found
torch_vulkan.device_count()     # 1 (one device for now)
torch_vulkan.device_name()      # e.g. "Intel(R) UHD Graphics 620"
torch_vulkan.synchronize()      # wait for queued GPU work
torch_vulkan.memory_stats()     # {"allocated_bytes", "reserved_bytes", "blocks"}
torch_vulkan.empty_cache()      # return cached device memory to the driver
torch_vulkan.fallback_counts()  # {op: count} of ops that ran on the CPU
```

Importing `torch_vulkan` also registers `torch.vk` (PyTorch's device-module
interface) and `Tensor.vk()`; `torch.manual_seed` seeds the GPU generator.

### Environment variables

| Variable | Effect |
|---|---|
| `TORCH_VK_STRICT=1` | Raise instead of running an op on the CPU. |
| `TORCH_VK_VALIDATE=1` | Enable the Khronos validation layer with synchronisation and best-practices checks; every message is printed and counted, and the tests fail on any. |
| `TORCH_VK_LINALG=cpu` | Send dense linear algebra to LAPACK on the CPU, as a reference for testing and comparison. |
| `TORCH_VK_GEMM=tiled`, `float4` or `subgroup` | Choose the matrix-multiply kernel (`float4` is the default on Arm GPUs, `subgroup` on other GPUs with subgroup size control, `tiled` elsewhere). |
| `TORCH_VK_SUBGROUP=8`, `16` or `32` | Subgroup size of the `subgroup` kernels (by default 16 for float32 and 8 for float64), within the device's range. |
| `TORCH_VK_API=1.1` | Use the Vulkan 1.1 code path on a newer driver. |
| `TORCH_VK_DISABLE=int64,float64,int16,float16,int8,storage16,storage8,denorm,subgroups` | Hide device features, to test a small GPU's code paths on a bigger one. |
| `TORCH_VK_CACHE_DIR` | Directory for the compiled-pipeline cache (default `%LOCALAPPDATA%\torch-vulkan`, or `$XDG_CACHE_HOME/torch-vulkan`, or `~/.cache/torch-vulkan`). |

## Limitations

- **Performance is not the focus yet.** Kernels favour correctness and
  portability. On the Intel UHD 620, float32 matrix products reach about
  35% of the GPU's peak (about 148 GFLOP/s at 2048^3, ahead of MKL on the
  CPU) and float64 about 30 GFLOP/s; per-op overhead is about 9 µs, close to
  the CPU's. A 2^20-point complex64 FFT takes about 12 ms (the CPU about
  6 ms); long real and prime-length transforms are ahead of the CPU.
  Cooperative-matrix GEMM is future work.
- **`torch.compile`**: the default Inductor backend has no code generator for
  `"vk"`. `backend="eager"` and `backend="aot_eager"` work (a forward and
  backward pass matches the CPU), without fusion.
- **One device**; no pinned memory.
- **Tensors of 2^31 or more elements** are not supported (32-bit element
  offsets).
- **Quantised and int4 tensors** are out of scope.
- **float32 subnormals** are flushed to zero on GPUs that flush them in
  shaders (e.g. the H100).
- **Long float64 kernels** can trip the Windows GPU watchdog on small
  integrated GPUs.

## How it works

- A Rust core (`crates/vkcore`, using [`ash`](https://crates.io/crates/ash))
  manages the device, a caching memory pool, a pipeline cache and an in-order
  command stream that submits kernels in batches. A thin C++ layer
  (`crates/pyvk/csrc`) registers the aten kernels on PyTorch's `PrivateUse1`
  dispatch key and calls the core through a C ABI, so only that layer is tied
  to PyTorch's C++ ABI.
- Tensors are buffer device addresses: a kernel gets `(pointer, sizes,
  strides)` in push constants, so views, transposes and broadcasts need no
  copies and there are no descriptor sets.
- Kernels are written in Slang, templated per dtype and compiled to SPIR-V at
  build time with `-fp-mode precise`. float32 transcendentals use Cephes'
  algorithms instead of the driver's built-ins, whose accuracy varies between
  drivers; float64 math is implemented in `_math64.slang`.
- The device is named `"vk"` because PyTorch already has a built-in `vulkan`
  device type (its former mobile backend).

The development log, milestone by milestone with design notes and
measurements, is in [docs/development-log.md](docs/development-log.md).

## Related projects

- [torchvulkan](https://github.com/zainsharief/torchvulkan): a Vulkan backend
  for PyTorch focused on neural-network training, with prebuilt wheels for
  Linux, Windows and macOS, multi-GPU support and cooperative-matrix GEMM.
- PyTorch's former Vulkan backend and the
  [ExecuTorch](https://github.com/pytorch/executorch) Vulkan delegate:
  inference on mobile GPUs.
- [pytorch_dlprim](https://github.com/artyom-beilis/pytorch_dlprim): an
  OpenCL backend, and a Vulkan port of it in progress.
- Discussion of an official Vulkan backend:
  [pytorch/pytorch#160230](https://github.com/pytorch/pytorch/issues/160230).

## Contributing

Bug reports from more GPUs and drivers are especially welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT licence](LICENSE-MIT), at your option. Code derived from PyTorch,
Cephes and the Faddeeva package keeps its own licences; see [NOTICE](NOTICE).
Unless you state otherwise, any contribution you submit for inclusion is
dual-licensed as above, without additional terms.
