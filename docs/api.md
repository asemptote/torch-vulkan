# API reference

torch-vulkan adds a PyTorch device, `"vk"`. Tensors and modules move to it as
to any other device (`x.to("vk")`, `model.to("vk")`, `torch.zeros(3,
device="vk")`), and ops on `"vk"` tensors run as Vulkan compute kernels. This
page lists what the package adds on top of that.

## The `torch_vulkan` module

Importing `torch_vulkan` registers the device. It also registers
`torch.vk`, PyTorch's device-module interface for the backend (the same
functions, reached as `torch.vk.synchronize()` and so on), and the
`Tensor.vk()` method.

| Function | Returns / does |
|---|---|
| `is_available()` | `True` if a Vulkan device that torch-vulkan can use was found. |
| `device_count()` | `1` if a device is available, else `0` (one device for now). |
| `device_name()` | The device's name, e.g. `"Intel(R) UHD Graphics 620"`. |
| `synchronize(device=None)` | Waits until all queued GPU work has finished. Reading a value back (`.item()`, `.cpu()`) synchronises by itself. |
| `memory_stats()` | `{"allocated_bytes", "reserved_bytes", "blocks"}`: memory held by live tensors, memory held by the caching allocator in total, and the number of device memory blocks. |
| `empty_cache()` | Returns cached, unused device memory to the driver. |
| `fallback_counts()` | `{op: count}` of aten ops that ran on the CPU instead of the GPU since the process started; an op whose CPU call only raised an error appears as `"op (raised)"`. Empty in normal use. |
| `manual_seed(seed)` | Seeds the `"vk"` random generator. `torch.manual_seed` seeds it too. |
| `seed()` | Seeds the `"vk"` generator nondeterministically and returns the seed. |
| `initial_seed()` | The seed the generator started from. |
| `get_rng_state(device="vk")` / `set_rng_state(state, device="vk")` | The generator's state (seed and offset) as a `uint8` tensor, and restoring it. |

Random numbers come from a Philox generator on the GPU. They are
deterministic for a given seed, but not the same numbers as the CPU's
generator.

## Dtypes

- float32, float64, complex64 and complex128 are computed natively; float64
  and complex128 need a GPU with `shaderFloat64`, and on one without it,
  creating such a tensor raises `TypeError` (as on Apple's MPS).
- float16 and bfloat16 are computed in float32 and rounded once (matrix
  products on tensor cores where the GPU has them); complex32 likewise.
- 8- and 16-bit integers are computed in int32; int64 and the unsigned types
  are supported everywhere (int64 arithmetic is emulated with 32-bit words, so
  `shaderInt64` is not needed).
- float8 (e4m3fn, e5m2 and the fnuz variants) converts exactly to and from the
  other types; `_scaled_mm` multiplies float8 matrices.
- Per-tensor affine quantised tensors (`quantize_per_tensor`, `dequantize`,
  `int_repr` and views) are supported; per-channel quantisation is not.

## Environment variables

Set them before importing `torch_vulkan`; most are read once, at first use.
For on/off flags, any value other than empty or `0` switches the flag on.

### Behaviour

| Variable | Effect |
|---|---|
| `TORCH_VK_STRICT=1` | Raise `NotImplementedError` instead of running an op on the CPU. |
| `TORCH_VK_CACHE_DIR=dir` | Directory for the compiled-pipeline cache (default `%LOCALAPPDATA%\torch-vulkan`, else `$XDG_CACHE_HOME/torch-vulkan`, else `~/.cache/torch-vulkan`). |
| `TORCH_VK_MAX_BATCH_GROUPS=n` | Submit a batch of queued kernels once it holds `n` workgroups (`0`: no limit). The default limits batches on Intel GPUs only, where some drivers drop the results of very long submissions. |
| `TORCH_VK_SPIN_US=n` | How long a synchronize polls for the GPU before it blocks in the driver's wait, in microseconds (default `500`; `0`: block at once). Polling saves the wake-up after short work: about 19 us per synchronize on NVIDIA. |

### Choosing kernels

| Variable | Effect |
|---|---|
| `TORCH_VK_GEMM=tiled`, `float4` or `subgroup` | The matrix-multiply kernel: `float4` is the default on Arm GPUs, `subgroup` on other GPUs with subgroup size control, `tiled` elsewhere. By default float16 and bfloat16 products use cooperative matrices (tensor cores) where available; setting this variable runs them in float32 instead. |
| `TORCH_VK_SUBGROUP=8`, `16` or `32` | Subgroup size for the `subgroup` kernels (default 16 for float32, 8 for float64), within the device's range. |
| `TORCH_VK_LINALG=cpu` | Send dense linear algebra to LAPACK on the CPU: a reference for testing and comparison, not a performance option. |
| `TORCH_VK_EIGH_QL=1` | `eigh`: use tridiagonal QL instead of divide and conquer for every size (comparison and debugging). |
| `TORCH_VK_SVD_JACOBI=1` | `svd`: use one-sided Jacobi instead of bidiagonalisation for every size (comparison and debugging). |

### Testing a smaller GPU's paths on a bigger one

| Variable | Effect |
|---|---|
| `TORCH_VK_API=1.1` | Use the Vulkan 1.1 code path on a newer driver. |
| `TORCH_VK_DISABLE=a,b,...` | Hide device features: `int64`, `float64`, `int16`, `float16`, `int8`, `storage16`, `storage8`, `denorm`, `subgroups`, `coopmat`, `bfloat16`. |

### Debugging

| Variable | Effect |
|---|---|
| `TORCH_VK_VALIDATE=1` | Enable the Khronos validation layer with synchronisation and best-practices checks; every message is printed and counted. |
| `TORCH_VK_TRACE=1` | Print every kernel launch and wait for it to finish (slow; finds which kernel fails). |
| `TORCH_VK_FALLBACK_DETAIL=1` | Count fallbacks per argument signature, e.g. `"aten::eq.Tensor_out [UInt16 1d, UInt16 1d, Bool 1d]"`. |
| `TORCH_VK_FALLBACK_TRACE=1` | Print each fallback's arguments (sizes, strides, small values) and where it came from, to stderr. |

### Benchmark scripts

Read by `benchmarks/*.py`, not by the library.

| Variable | Effect |
|---|---|
| `TORCH_VK_BENCH_SMALL=1` | Small sizes only: a quick check that the scripts run (`benchmarks/integrated.py --small` sets it). |
| `TORCH_VK_BENCH_COMMIT=sha` | The commit `benchmarks/integrated.py` names on its results page, where the tree is not a git checkout. |

## `torch.compile`

The default Inductor backend has no code generator for `"vk"`;
`torch.compile(model, backend="eager")` and `backend="aot_eager"` work, without
fusion. A fusing backend is planned for 0.3.0.
