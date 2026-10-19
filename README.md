# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M1 — core runtime done** (memory pool, kernel cache, batched stream),
tested on an Intel UHD 620 with the validation layer clean.

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
  shaders/*.slang          compute kernels, compiled to SPIR-V by build.rs
  tests/runtime.rs         GPU tests (batching, dependencies, reuse, copies, pool, cache)
  examples/smoke.rs        hardware check
  examples/dispatch_bench.rs  per-op overhead, synchronized vs batched
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

All 10 tests (3 allocator unit tests, 7 GPU tests) and both examples run with
no validation messages.

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. **M2 Python prototype**: PyO3 module + `__torch_dispatch__` tensor subclass;
   element-wise ops generated from one Slang template over (dtype, strides).
3. **M3 C++ shim**: real `PrivateUse1` device `vk` -- allocator, device guard,
   `empty_strided`/`copy_`/`resize_`/`as_strided`/`view`, CPU fallback
   (and a test for the empty-result bug seen with `out=` fallbacks in other backends).
4. **M4 reductions and scans**: sum/mean/var/norm/min/max, cumsum/cumprod/logcumsumexp.
5. **M5 matmul** (f32/f64, tiled, subgroup ops).
6. **M6 complex dtypes + FFT** (VkFFT).
7. **M7 linear algebra**: blocked LU/Cholesky/QR with GPU trailing updates; eigh/svd via CPU initially.
8. **Testing throughout**: PyTorch OpInfo samples compared against CPU.
