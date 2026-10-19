# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M2 — Python prototype**: PyTorch tensors on device `"vk"` with
element-wise ops, casts, copies and fills running as Vulkan kernels; every other
op falls back to the CPU. Tested on an Intel UHD 620 with the validation layer clean.

```python
import torch, torch_vulkan

x = torch.randn(1_000_000, dtype=torch.float64, device="vk")  # random fill runs on the CPU
y = (x * 2 + 1).relu()          # three Vulkan kernels, batched
print(y[:3], y.sum())           # sum runs on the CPU for now (warns once per op)
torch_vulkan.fallback_counts()  # {'aten.normal_.default': 1, 'aten.sum.default': 1}
```

## Python prototype (M2)

- `python/torch_vulkan`: `VkTensor`, a `torch.Tensor` wrapper subclass on
  device `vk:0` holding a pool allocation. All aten ops reach
  `__torch_dispatch__` (below autograd, so `backward()` works) and are handled by
  a Vulkan kernel, by meta-tensor shape math for views (they share the buffer),
  or by a CPU fallback that copies operands and writes back mutated arguments.
- `torch.utils.backend_registration._setup_privateuseone_for_python_backend`
  registers the device guard, hooks and device module autograd expects; factory
  functions (`torch.zeros(..., device="vk")`, `.to("vk")`) reach small
  `torch.library` kernels on PrivateUse1.
- `crates/pyvk`: PyO3 module `torch_vulkan._native` (alloc, upload, download,
  dispatch with a packed push-constant block, synchronize, stats).
- Kernels are Slang templates compiled per dtype (`// axis T: ...` lines in the
  shader; see `build.rs`), with the op chosen by a specialization constant:
  `unary` (20 ops), `binary` (11, with scalar operands and `alpha`), `compare`
  (6, bool out), `cast` (all 25 pairs of f32/f64/i32/i64/bool; also strided
  copies), `fill`. Operands are addressed as pointer + up to 4 coalesced
  (size, stride) dims, so broadcasting, transposes and slices need no copies.
- Precision: float32 `sin/cos/tan/exp/log/...` use the GPU's built-ins, which
  Vulkan only requires to be accurate to ~2^-11 absolute (sin/cos) or a few ULP;
  tests use those bounds. SPIR-V has no 64-bit transcendentals, so float64
  `exp/log/sin/...` currently fall back to the CPU (a float64 math library is planned).

Measured (`python benchmarks/elementwise.py`, UHD 620 vs 4-thread CPU, torch 2.14.1):

| | vk | CPU |
|---|---|---|
| per-op overhead, 1024 floats | 136 µs | 22 µs |
| `relu(x*2+1)`, 16M float32 | 29 ms | 33 ms |
| `relu(x*2+1)`, 16M float64 | 40 ms | 71 ms |

Large arrays are already faster than the CPU (memory-bound, ~14–20 GB/s);
small ops are dominated by Python dispatch, which the C++ shim (M3) removes.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| GPU API | Vulkan 1.3 via [`ash`](https://crates.io/crates/ash) | float64/int64, raw device pointers, explicit memory types, low dispatch overhead; runs on Intel, AMD, NVIDIA |
| Core language | Rust (`crates/vkcore`) | Memory/queue/pipeline management; exposed through a C ABI so only a thin layer is tied to PyTorch's C++ ABI |
| PyTorch integration | Thin C++ shim on `PrivateUse1` (later), with a `__torch_dispatch__` + PyO3 prototype first | PyTorch's backend API is C++-only; the shim is the only part rebuilt per PyTorch release |
| Device name | `vk` | PyTorch already has a built-in `vulkan` device type (its old, unmaintained mobile backend) |
| Tensor addressing | `VK_KHR_buffer_device_address`: tensors are `(pointer, offset, sizes, strides)` passed in push constants | Views and strides become pointer arithmetic, as in CUDA; no descriptor sets |
| Memory | Prefer device-local + host-visible + coherent memory | On integrated GPUs (UHD 620, Strix Halo) CPU and GPU share tensors with no copies |
| Kernels | [Slang](https://shader-slang.org) → SPIR-V (`crates/vkcore/shaders`, compiled by `build.rs`) | Generics, pointers, autodiff; can later also target CUDA/Metal |
| FFT | VkFFT (planned) | Complex + float64 FFT on Vulkan |
| Matmul | Own kernels, modeled on llama.cpp's `ggml-vulkan` | No mature Vulkan BLAS exists |

## Verified on this machine (M0)

`cargo run --release --example smoke` on Intel UHD 620, driver 101.2134:

- Vulkan 1.3.215; float64, int64, int16, float16, buffer device address,
  scalar block layout, Vulkan memory model: all supported
- Subgroup size 32 (8–32), 32 KiB shared memory, max workgroups.x 65536,
  max single allocation 7.94 GiB, max buffer 4 GiB
- One 7.94 GiB heap with a device-local + host-visible memory type (unified memory)
- Not supported: int64 atomics, float32/float64 atomic add, cooperative matrix
  (tensor cores) -- reductions must use subgroup ops / shared memory instead
- Kernel `y = x² + 1/3` over 16M float64 values via device-address pointers:
  max error 1.8e-15 vs CPU (FMA contraction), 15 ms GPU vs 22 ms single-thread CPU,
  ~18 GB/s effective bandwidth

## Core runtime (M1)

- **Memory pool** (`memory.rs`): tensors are sub-allocated (256 B aligned,
  best fit, coalescing) from 64 MiB persistently mapped, device-addressable
  blocks; larger requests get their own block. Freed memory is cached and
  reused in stream order, like PyTorch's CUDA allocator; `trim()` returns
  empty blocks to Vulkan, and allocation retries after trimming when out of memory.
- **Kernel cache** (`kernels.rs`): one shared pipeline layout (no descriptor
  sets, 128-byte push-constant block). Pipelines are keyed by
  (kernel, specialization constants); compiled code persists across runs in
  `%LOCALAPPDATA%\torch-vulkan\pipelines.bin` (override with `TORCH_VK_CACHE_DIR`).
- **Stream** (`stream.rs`): in-order like a CUDA stream. Commands are recorded
  with a full memory barrier between them and submitted in batches of up to
  256, tracked by a timeline semaphore; command buffers are recycled.
  The host must `synchronize()` before touching memory that queued work uses.
- **Validation** (`context.rs`): `TORCH_VK_VALIDATE=1` enables the Khronos
  validation layer with synchronization and best-practices checks; every
  message is printed and counted, and the tests fail on any.

Measured on the UHD 620 (`cargo run --release --example dispatch_bench`):
synchronizing after every small kernel costs **486 µs/op**; batching brings it
to **5.6 µs/op** (~87× less).

## Layout

```
Cargo.toml                 workspace
crates/vkcore/             Rust Vulkan core
  src/context.rs           device + queue, capabilities, validation reporting
  src/memory.rs            caching sub-allocator (Pool, Allocation, RangeAllocator)
  src/kernels.rs           pipeline cache + embedded SPIR-V table
  src/stream.rs            batched in-order command stream
  src/lib.rs               Runtime (all of the above) + helpers
  shaders/*.slang          compute kernels (templated per dtype), compiled to SPIR-V by build.rs
  shaders/_strided.slang   shared push-constant layout + strided index math
  tests/runtime.rs         GPU tests (batching, dependencies, reuse, copies, pool, cache)
  examples/smoke.rs        hardware check
  examples/dispatch_bench.rs  per-op overhead, synchronized vs batched
crates/pyvk/               PyO3 bindings -> torch_vulkan._native
python/torch_vulkan/       VkTensor, op dispatch, kernel launch, device registration
tests/python/test_vk.py    138 tests: results on "vk" vs CPU, no unexpected fallbacks
benchmarks/elementwise.py  per-op overhead and throughput vs CPU
pyproject.toml             maturin build of the Python package
```

## Building

Requirements: Rust (rustup, MSVC toolchain), Visual Studio Build Tools with C++,
a Vulkan 1.3 driver, and the Vulkan SDK (1.4.363 here: `slangc`, validation
layers, `vkconfig`). `build.rs` finds `slangc` via `SLANGC`, then `VULKAN_SDK`, then PATH.

```powershell
$env:PATH = "$HOME\.cargo\bin;$env:PATH"
cargo run --release --example smoke
```

Tests and examples under validation (synchronization and best-practices checks included):

```powershell
$env:TORCH_VK_VALIDATE = "1"
cargo test --release
cargo run --release --example smoke
```

All 10 Rust tests (3 allocator unit tests, 7 GPU tests) and both examples run
with no validation messages.

Python package (editable install into `.venv`, CPU build of PyTorch):

```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest maturin
.\.venv\Scripts\maturin develop --release --uv
$env:TORCH_VK_VALIDATE = "1"; .\.venv\Scripts\python -m pytest -q
```

All 138 Python tests pass with no validation messages.

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. ~~**M2 Python prototype**~~: done (see above). Known gaps: ops that change a
   tensor's shape in place (`resize_`, `transpose_`, `out=` arguments of the
   wrong size) raise `NotImplementedError`; >4 non-mergeable dims, float64
   transcendentals and everything non-element-wise run on the CPU.
3. **M3 C++ shim**: real `PrivateUse1` device `vk` -- allocator, device guard,
   `empty_strided`/`copy_`/`resize_`/`as_strided`/`view`, CPU fallback
   (and a test for the empty-result bug seen with `out=` fallbacks in other backends).
4. **M4 reductions and scans**: sum/mean/var/norm/min/max, cumsum/cumprod/logcumsumexp;
   float64 math library (exp/log/sin/cos/...) for the element-wise kernels.
5. **M5 matmul** (f32/f64, tiled, subgroup ops).
6. **M6 complex dtypes + FFT** (VkFFT).
7. **M7 linear algebra**: blocked LU/Cholesky/QR with GPU trailing updates; eigh/svd via CPU initially.
8. **Testing throughout**: PyTorch OpInfo samples compared against CPU.
