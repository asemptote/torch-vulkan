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
  computed in float32 and rounded once; 8/16-bit integers are computed in int32;
  float8 converts exactly. float32 division, sqrt and rsqrt are correctly
  rounded on every driver, and float64 division too, without fused
  multiply-adds. (The CPU's own vectorised float32 sqrt is not always correctly
  rounded, so a result can differ from a particular CPU's by 1 ulp while being
  the correctly rounded one.)
- **Reductions and scans**: sum, mean, prod, max/min, argmax, var/std, norms,
  any/all, cumsum/cumprod, cummax/cummin, logcumsumexp.
- **Matrix products**: mm, bmm, matmul, addmm, baddbmm, addmv, dot, `nn.Linear`,
  einsum; integer and 16-bit products too, and float16 and bfloat16 on tensor
  cores (cooperative matrices) where the GPU has them.
- **Quantisation**: int8 weight-only and dynamic 4-bit matrix products
  (`_weight_int8pack_mm`, `_dyn_quant_matmul_4bit`), float8 `_scaled_mm`,
  per-tensor affine quantised tensors, and quantisation-aware training's fake
  quantisation.
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
  transposed), pooling, normalisation, softmax, losses, upsampling (Lanczos
  included), grid sampling, embeddings, attention and `nn.MultiheadAttention`'s
  fast path, nested tensors, `segment_reduce`.
- **Training**: the optimisers' fused (`fused=True`) and foreach updates for
  Adam, AdamW, SGD and Adagrad, and `torch.amp.GradScaler`.

All numerical work runs on the GPU (see the design principles in
[CONTRIBUTING.md](CONTRIBUTING.md#design-principles)). **No op runs on the CPU
in PyTorch's own tests**: across 36 of PyTorch's device-generic test files
(`test_ops.py`, `test_torch.py`, `test_linalg.py`, `test_nn.py`,
`test_autograd.py`, `test_optim.py`, `test_sparse.py`, `test_nestedtensor.py`
and others; about 160,000 tests each on lavapipe and an NVIDIA GPU, run by
`tools/pytorch_tests.py` with every CPU call recorded), no op computes on the
CPU or calls it to raise an error. Every remaining failure also fails on
the CPU or has a written reason in that script's `EXPECTED` table. In the
OpInfo sweep (`tools/opinfo_coverage.py --strict`, PyTorch's sample inputs for
every op compared against the CPU) every op runs on the GPU and matches the
CPU. PyTorch's CPU fallback remains as a safety net for anything the tests
don't reach: it is counted in `torch_vulkan.fallback_counts()`, and
`TORCH_VK_STRICT=1` turns it into an error.

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
| Intel HD Graphics 530 | Mesa (ANV), Linux | validation layer clean |
| NVIDIA H100 NVL, L40S | NVIDIA, Linux | PyTorch's test files; float16 on tensor cores; [benchmarks against CUDA](benchmarks/results/) |
| Arm Mali-G57 (Galaxy A13 5G) | Arm, Android (Termux) | Vulkan 1.1, no float64, no int64; the tests pass except where that phone's own CPU reference is wrong ([#13](https://github.com/asemptote/torch-vulkan/issues/13)) |
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
pip install https://github.com/asemptote/torch-vulkan/releases/download/v0.2.0/torch_vulkan-0.2.0-cp310-abi3-win_amd64.whl
# Linux
pip install https://github.com/asemptote/torch-vulkan/releases/download/v0.2.0/torch_vulkan-0.2.0-cp310-abi3-manylinux_2_28_x86_64.whl
```

On Android (Termux), build from source with the release's prebuilt kernels:
extract `torch-vulkan-spirv-v0.2.0.tar.gz` and run `tools/build-termux.sh`
with that directory (the script describes the setup). Other platforms build
from source as below.

## Building

Requirements: Rust (stable), a C++20 compiler, Python 3.10 or newer, PyTorch
2.14, the [Slang](https://shader-slang.org)
compiler `slangc` (in the Vulkan SDK, or from Slang's releases), and a Vulkan
driver. `build.rs` finds `slangc` through `SLANGC`, then `VULKAN_SDK`, then
`PATH`. The extension is compiled against the PyTorch of the environment it is
installed into, so rebuild it after upgrading PyTorch. The tests and CI use
PyTorch 2.14.1; with 2.15, two of the CPU's error messages differ from the
table generated for 2.14 (sorting complex tensors, `matrix_exp` of integers).

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
The full reference, including dtypes and every environment variable, is
[docs/api.md](docs/api.md).

### Environment variables

The ones most often needed (all of them are in [docs/api.md](docs/api.md#environment-variables)):

| Variable | Effect |
|---|---|
| `TORCH_VK_STRICT=1` | Raise `NotImplementedError` where an op would compute on the CPU (an op whose CPU check only raises gives the CPU's error). |
| `TORCH_VK_VALIDATE=1` | Enable the Khronos validation layer with synchronisation and best-practices checks; every message is printed and counted. |
| `TORCH_VK_CACHE_DIR=dir` | Directory for the compiled-pipeline cache. |
| `TORCH_VK_DISABLE=a,b,...` | Hide device features (e.g. `float64`, `coopmat`), to test a small GPU's code paths on a bigger one. |

## Limitations

- **Performance** comes after correctness and portability, and varies by
  operation. On integrated Intel GPUs (against their own CPUs:
  [HD 530](benchmarks/results/hd530-mesa.md),
  [UHD 620](benchmarks/results/uhd620-windows.md)), float32 matrix products
  reach about 95-110 GFLOP/s at 2048^3 (on the UHD 620, level with MKL on its
  4-core CPU) and float64 about 26 GFLOP/s; prime-length FFTs are 2.4-5.6x
  faster than the CPU and long real FFTs about 2-2.5x faster, and a
  2^20-point complex64 FFT takes 5 ms on the HD 530 (CPU 7.7 ms) and 12 ms on
  the UHD 620 (level with its CPU); dense linear algebra is 5-10x slower than
  the CPU's LAPACK on these small GPUs.
  Against CUDA on the same GPUs ([tables](benchmarks/results/)), as vk's time
  over CUDA's:
  - float16 and bfloat16 matrix products run on tensor cores at 155-165
    TFLOP/s at 8192^3: 1.1-1.2x on an L40S and 2.5-2.6x on an H100 NVL, where
    NVIDIA's own Vulkan cooperative-matrix sample tops out at about 170;
  - float32 products 2.3-2.5x; float64 products 0.85-0.9x on the L40S and
    about 5x on the H100, whose CUDA uses float64 tensor cores;
  - float64 `svd` 0.2-0.8x (faster than CUDA), `eigh` 1.5-3.4x, `solve`
    1.4-3.4x; `qr` of tall, skinny matrices is still 20-40x slower than CUDA
    (1000x64 on an L40S), as its Householder panel runs on one workgroup;
  - FFTs and real FFTs at 0.84-1.9x on the L40S and 1.5-1.7x on the H100 at
    2^24 points; smaller transforms (2^16-2^20 points) reach up to about 2.3x
    on the H100, where each call's fixed submission and synchronisation cost
    (about 40 us) dominates;
  - element-wise bandwidth 1-1.3x, and per-op overhead 0.86-0.9x (slightly
    lower than CUDA's).
- **`torch.compile`**: the default Inductor backend has no code generator for
  `"vk"`. `backend="eager"` and `backend="aot_eager"` work (a forward and
  backward pass matches the CPU), without fusion.
- **One device.**
- **Tensors of 2^31 or more elements** are not supported (32-bit element
  offsets).
- **Quantised tensors** are per-tensor affine only; per-channel quantised
  tensors and the int4 dtypes are not supported.
- **float32 subnormals** are flushed to zero on GPUs that flush them in
  shaders (e.g. the H100).
- **Long float64 kernels** can trip the Windows GPU watchdog on small
  integrated GPUs; on Linux's i915 (Intel Gen9), a single dispatch running
  for a second or more can lose its results without a device-lost error
  (the stream bounds batches on Intel GPUs, `TORCH_VK_MAX_BATCH_GROUPS`).

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
