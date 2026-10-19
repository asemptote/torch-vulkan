# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M3 — native PyTorch backend**: device `"vk"` is a real PyTorch
backend implemented in C++ on the PrivateUse1 dispatch key. Element-wise ops,
casts, copies and fills run as Vulkan kernels; every other op runs on the CPU
through PyTorch's boxed fallback. Tested on an Intel UHD 620 with the
validation layer clean.

```python
import torch, torch_vulkan

x = torch.randn(1_000_000, dtype=torch.float64, device="vk")  # random fill runs on the CPU
y = (x * 2 + 1).relu()          # three Vulkan kernels, batched
print(y[:3], y.sum())           # sum runs on the CPU for now (warns once per op)
torch_vulkan.fallback_counts()  # {'aten::normal_': 1, 'aten::sum.IntList_out': 1}
```

## PyTorch backend (M3)

- `crates/pyvk/csrc/backend.cpp`: a `c10::Allocator` over the Rust memory pool
  (`data_ptr()` is the host-mapped address; the GPU address is derived from the
  allocation context), a device guard, `PrivateUse1HooksInterface`, and aten
  kernels for `empty`/`empty_strided`, views (`view`, `as_strided`,
  `_reshape_alias`), `resize_` (grows storage, keeps contents), `set_`,
  `_copy_from`/`_copy_from_and_resize` (CPU <-> vk and vk <-> vk with
  broadcasting and dtype conversion), `_local_scalar_dense`, `fill_`, `zero_`.
- `crates/pyvk/csrc/ops.cpp`: the `.out` overloads of structured ops (PyTorch's
  generated functional/in-place variants call them): 19 unary ops plus `relu`,
  11 binary ops (scalar operands, `alpha`, scalar-first `sub`), 6 comparisons
  (Tensor and Scalar), all with PyTorch's type promotion. Cases without a
  kernel (e.g. float64 `exp`) run the same overload through
  `at::native::cpu_fallback`, so results stay correct.
- Every other op: a boxed PrivateUse1 fallback to the CPU, counted in
  `torch_vulkan.fallback_counts()` and warned about once per op. Its results
  are written back with `_copy_from_and_resize`, which resizes `out=`
  arguments (the empty-result bug of torch-directml and pytorch_ocl is tested).
- Strided launches merge contiguous dims; tensors with more than 4
  non-mergeable dims loop over the outer dims on the host (one dispatch each).
- `crates/pyvk/src/ffi.rs`: the C ABI from C++ to the Rust runtime (alloc, free,
  dispatch, synchronize); the C++ is compiled by `crates/pyvk/build.rs` against
  the libtorch of the target Python and linked into the same extension module.
- `python/torch_vulkan`: names the backend `vk`, registers `torch.vk`, adds
  `Tensor.vk()`, and exposes `synchronize`, `memory_stats`, `empty_cache`,
  `fallback_counts`.

The M2 prototype (a `__torch_dispatch__` tensor subclass) is replaced; its
kernels and tests carry over.

Kernels are Slang templates compiled per dtype (`// axis T: ...` lines in the
shader; see `build.rs`), with the op chosen by a specialization constant:
`unary`, `binary`, `compare`, `cast` (all 25 pairs of f32/f64/i32/i64/bool;
also strided copies) and `fill`. Operands are addressed as pointer + up to 4
coalesced (size, stride) dims, so broadcasting, transposes and slices need no copies.

Precision: float32 `sin/cos/tan/exp/log/...` use the GPU's built-ins, which
Vulkan only requires to be accurate to ~2^-11 absolute (sin/cos) or a few ULP;
tests use those bounds. SPIR-V has no 64-bit transcendentals, so float64
`exp/log/sin/...` currently run on the CPU (a float64 math library is planned).

Measured (`python benchmarks/elementwise.py`, UHD 620 vs 4-thread CPU, torch 2.14.1):

| | vk (M3, C++) | vk (M2, Python) | CPU |
|---|---|---|---|
| per-op overhead, 1024 floats | 9 µs | 136 µs | 7 µs |
| `relu(x*2+1)`, 16M float32 | 27 ms | 29 ms | 27 ms |
| `relu(x*2+1)`, 16M float64 | 37 ms | 40 ms | 51 ms |

Large arrays are memory-bound (~15–22 GB/s on this integrated GPU); per-op
overhead is now close to the CPU's.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| GPU API | Vulkan 1.3 via [`ash`](https://crates.io/crates/ash) | float64/int64, raw device pointers, explicit memory types, low dispatch overhead; runs on Intel, AMD, NVIDIA |
| Core language | Rust (`crates/vkcore`) | Memory/queue/pipeline management; exposed through a C ABI so only a thin layer is tied to PyTorch's C++ ABI |
| PyTorch integration | C++ backend on `PrivateUse1` (`crates/pyvk/csrc`), calling the Rust core through a C ABI | PyTorch's backend API is C++-only; only this layer is tied to a PyTorch release |
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
crates/pyvk/               extension module torch_vulkan._native
  csrc/backend.cpp         allocator, device guard, hooks, views, resize, copies, CPU fallback
  csrc/ops.cpp             element-wise aten kernels (strided launches)
  src/ffi.rs               C ABI from C++ to the Rust runtime
  src/lib.rs               Python functions (synchronize, stats, fallback counts)
  build.rs                 compiles csrc/ against the target Python's libtorch
python/torch_vulkan/       backend naming and torch.vk device module
tests/python/test_vk.py    140 tests: results on "vk" vs CPU, no unexpected fallbacks
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
.\.venv\Scripts\maturin develop --release --uv   # compiles csrc/ against this venv's torch
$env:TORCH_VK_VALIDATE = "1"; .\.venv\Scripts\python -m pytest -q
```

All 140 Python tests pass with no validation messages.

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. ~~**M2 Python prototype**~~: done; superseded by M3.
3. ~~**M3 native backend**~~: done (see above). Not yet: bool binary ops
   (`&`, `*` on masks), multiple devices, pinned memory, a GPU random-number
   generator (random fills run on the CPU).
4. **M4 reductions and scans**: sum/mean/var/norm/min/max, cumsum/cumprod/logcumsumexp;
   float64 math library (exp/log/sin/cos/...) for the element-wise kernels.
5. **M5 matmul** (f32/f64, tiled, subgroup ops).
6. **M6 complex dtypes + FFT** (VkFFT).
7. **M7 linear algebra**: blocked LU/Cholesky/QR with GPU trailing updates; eigh/svd via CPU initially.
8. **Testing throughout**: PyTorch OpInfo samples compared against CPU.
