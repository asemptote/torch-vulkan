# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M0 — toolchain and core assumptions verified** on an Intel UHD 620.

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

## Layout

```
Cargo.toml                 workspace
crates/vkcore/             Rust Vulkan core
  src/lib.rs               Context (device, queue), Buffer (mapped + device address), Kernel, run()
  shaders/*.slang          compute kernels, compiled to SPIR-V at build time
  examples/smoke.rs        M0 hardware check
```

## Building

Requirements: Rust (rustup, MSVC toolchain), Visual Studio Build Tools with C++,
a Vulkan 1.3 driver, and the Vulkan SDK (1.4.363 here: `slangc`, validation
layers, `vkconfig`). `build.rs` finds `slangc` via `SLANGC`, then `VULKAN_SDK`, then PATH.

```powershell
$env:PATH = "$HOME\.cargo\bin;$env:PATH"
cargo run --release --example smoke
```

Run under validation (synchronization and best-practices checks included):

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_KHRONOS_VALIDATION_VALIDATE_SYNC = "true"
$env:VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES = "true"
cargo run --release --example smoke
```

The M0 smoke test produces no validation messages.

## Roadmap

1. **M1 core runtime**: sub-allocating memory pool, pipeline cache keyed by
   (kernel, dtype, specialization), batched command recording with timeline
   semaphores, async copies; validation-layer clean.
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
