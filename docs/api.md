# API reference

torch-vulkan adds a PyTorch device, `"vk"`. Tensors and modules move to it as
to any other device (`x.to("vk")`, `model.to("vk")`, `torch.zeros(3,
device="vk")`), and ops on `"vk"` tensors run as Vulkan compute kernels. This
page lists what the package adds on top of that.

## The `torch_vulkan` module

Importing `torch_vulkan` registers the device. It also registers
`torch.vk`, PyTorch's device-module interface for the backend (the same
functions, reached as `torch.vk.synchronize()` and so on), and the
`Tensor.vk()` method. A Termux build (a release's Android wheel, or
`tools/build-termux.sh`) records the PyTorch it was compiled against, and
importing it under any other PyTorch version raises an `ImportError` (after
a `python-torch` upgrade, install the matching build or rebuild).

| Function | Returns / does |
|---|---|
| `__version__` | The installed package's version, e.g. `"0.2.0"`. |
| `is_available()` | `True` if a Vulkan device that torch-vulkan can use was found. |
| `device_count()` | `1` if a device is available, else `0` (one device for now). |
| `device_name()` | The device's name, e.g. `"Intel(R) UHD Graphics 620"`. |
| `synchronize(device=None)` | Waits until all queued GPU work has finished. Reading a value back (`.item()`, `.cpu()`) synchronises by itself. |
| `memory_stats()` | `{"allocated_bytes", "reserved_bytes", "blocks"}`: memory held by live tensors, memory held by the caching allocator in total, and the number of device memory blocks. |
| `empty_cache()` | Returns cached, unused device memory to the driver. |
| `fallback_counts()` | `{op: count}` of aten ops that ran on the CPU instead of the GPU since the process started; an op whose CPU call only raised an error appears as `"op (raised)"`, and one refused with the CPU fallback off as `"op (refused)"`. Empty in normal use. |
| `set_cpu_fallback(enabled)` | Lets ops with no Vulkan kernel for their arguments run on the CPU (`True`) or makes them raise `NotImplementedError` (`False`, the default), for the whole process. See [The CPU fallback](#the-cpu-fallback). |
| `cpu_fallback_enabled()` | Whether the CPU fallback is on. |
| `manual_seed(seed)` | Seeds the `"vk"` random generator. `torch.manual_seed` seeds it too. |
| `seed()` | Seeds the `"vk"` generator nondeterministically and returns the seed. |
| `initial_seed()` | The seed the generator started from. |
| `get_rng_state(device="vk")` / `set_rng_state(state, device="vk")` | The generator's state (seed and offset) as a `uint8` tensor, and restoring it. |
| `set_fp16_arithmetic(enabled)` / `get_fp16_arithmetic()` | Lets float16 matrix products and convolutions multiply and add in float16 instead of float32 (off by default; see [float16 arithmetic](#float16-arithmetic)). Applies to calls made after it. |

Random numbers come from a Philox generator on the GPU. They are
deterministic for a given seed, but not the same numbers as the CPU's
generator.

## Dtypes

- float32, float64, complex64 and complex128 are computed natively; float64
  and complex128 need a GPU with `shaderFloat64`, and on one without it,
  creating such a tensor raises `TypeError` (as on Apple's MPS).
- float16 and bfloat16 are computed in float32 and rounded once (matrix
  products on tensor cores where the GPU has them); complex32 likewise.
  `set_fp16_arithmetic(True)` lets float16 matrix products and
  convolutions compute in float16 instead (below).
- 8- and 16-bit integers are computed in int32; int64 and the unsigned types
  are supported everywhere (int64 arithmetic is emulated with 32-bit words, so
  `shaderInt64` is not needed).
- float8 (e4m3fn, e5m2, the fnuz variants and e8m0fnu) converts exactly to
  and from the other types, as the CPU's conversions; the ops the CPU has for
  float8 (comparisons, `isnan`/`isinf`/`isfinite`, `abs`, `mul`, `where`,
  `any`/`all`) run on the GPU through float32, and `_scaled_mm` multiplies
  float8 matrices.
- Per-tensor affine quantised tensors (`quantize_per_tensor`, `dequantize`,
  `int_repr` and views) are supported; per-channel quantisation is not.

## The CPU fallback

Every op exercised by PyTorch's own test suites runs on the GPU. An op that
vk has no kernel for (with these arguments: a dtype or layout it does not
support, say) raises `NotImplementedError` naming the op, as PyTorch's MPS
backend does:

    NotImplementedError: torch_vulkan: aten::sort.values_stable has no Vulkan kernel for these arguments, and
    the CPU fallback is off. To run it on the CPU instead (slower), set TORCH_VK_CPU_FALLBACK=1 or call
    torch_vulkan.set_cpu_fallback(True). ...

With `TORCH_VK_CPU_FALLBACK=1` in the environment, or after
`torch_vulkan.set_cpu_fallback(True)`, such an op runs on the CPU instead:
its "vk" inputs are copied to the CPU and its results back, with a warning
the first time each op does so, and each call is counted in
`fallback_counts()`. Where the CPU would only raise an error for the
arguments (a dtype check, say), that error is raised either way.

`TORCH_VK_STRICT`, the switch before 0.3.0 (when the fallback was on by
default), still works: `TORCH_VK_STRICT=1` is the default, and
`TORCH_VK_STRICT=0` turns the fallback on. `TORCH_VK_CPU_FALLBACK` wins where
both are set. `TORCH_VK_LINALG=cpu` (below) turns the fallback on too.

Two uses of the CPU are not fallbacks and need no switch: an op's CPU call
that only raises its error (above), and, on GPUs without float64,
`torch.amp.GradScaler`'s unscaling, which takes the scale's reciprocal on a
CPU copy of the one-element scale.

## Autocast

`torch.autocast("vk", dtype=torch.float16)` (or `torch.bfloat16`; float16 is
PyTorch's default for the backend) applies the policy PyTorch uses for CUDA:
matrix products, convolutions, `linear` and the recurrent cells run in the
autocast dtype; ops that lose accuracy in 16 bits (`exp`, `log`, `pow`,
norms, losses, `softmax`, `sum`, `cumsum`, ...) run in float32; ops with
several inputs (`addcmul`, `atan2`, `dot`, `index_put`, ...) in the widest of
their dtypes. `torch.vk.get_amp_supported_dtype()` lists float16, bfloat16
and float32. Combine it with `torch.amp.GradScaler("vk")` for float16
training, as on CUDA.

## float32 matrix products and `torch.set_float32_matmul_precision`

By default (`"highest"`) float32 matrix products are computed in float32.
On GPUs with bfloat16 cooperative matrices (tensor cores: NVIDIA's from
Ampere on, with a driver that has `VK_KHR_shader_bfloat16`),
`torch.set_float32_matmul_precision` trades precision for speed as it does
on CUDA, but with bfloat16 where CUDA uses TF32 (Vulkan has no TF32
cooperative matrices):

| Setting | float32 products on "vk" | Error of each product a b | L40S, 8192^3 |
|---|---|---|---|
| `"highest"` (default) | float32 | float32's | 17 TFLOP/s |
| `"high"` | each operand split into two bfloat16 parts, hi + lo; hi hi + hi lo + lo hi, three bfloat16 products summed in float32 | under 2^-14 \|a b\| (about 16 bits) | 49 TFLOP/s |
| `"medium"` | operands rounded to bfloat16, one bfloat16 product summed in float32 | about 2^-7 \|a b\| (8 bits) | 142 TFLOP/s |

- It applies to `mm`, `bmm`, `addmm`, `baddbmm`, `matmul`, `nn.Linear`,
  `einsum` and other ops made of those, forward and backward, and to the
  real and imaginary products of complex64 ones. Results are still the same
  from run to run.
- As on CUDA, it does not apply to convolutions, pooling and interpolation
  (written as products on "vk"), factorisations and solves, or the products
  inside quantised and sparse ops: those stay in float32.
- Matrix-vector products (one row or column) stay in float32: they are
  bound by memory, not arithmetic.
- Elsewhere (no bfloat16 cooperative matrices, `TORCH_VK_GEMM` set,
  `TORCH_VK_DISABLE=bfloat16` or `coopmat`) the setting changes nothing,
  as on a CUDA GPU without TF32.
- Infinities and NaN in the operands give the infinities and NaN of the
  float32 product; finite values beyond bfloat16's largest (3.39e38) are
  truncated, not rounded to infinity.
- The bfloat16 parts are written to temporary tensors first: 2 bytes per
  operand value for `"medium"`, 6 for `"high"`.

## float16 arithmetic

By default float16 results match the CPU's: products and sums in float32,
rounded once. `torch_vulkan.set_fp16_arithmetic(True)` (or
`TORCH_VK_FP16_ARITHMETIC=1` in the environment) lets float16 matrix
products (`mm`, `bmm`, `addmm`, `linear`, ...) and convolutions multiply and
add in float16, with packed half2 arithmetic, as PyTorch's
`torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction` does for
CUDA. On GPUs with twice the float16 rate this runs them faster (on GPUs
whose kernels run 16-lane subgroups: Mali's Valhall GPUs and Intel's; elsewhere
the setting changes nothing). The cost is accuracy:

- sums are float16, so their rounding error grows with the reduction length
  (the inner dimension of a product, in channels x taps of a convolution):
  for 1408 terms (a 128-channel, 11-tap convolution) the error was 0.4% of
  the result's rms (3.6% at most), against at most 0.14% with float32 sums;
- partial sums overflow to inf past 65504, even when the final result would
  fit (scale inputs or weights down where that can happen).

bfloat16 and float32 are unaffected; so are matrix-vector products (decoding
steps), which are bound by memory bandwidth.

## Environment variables

Set them before importing `torch_vulkan`; most are read once, at first use.
For on/off flags, any value other than empty or `0` switches the flag on.

### Behaviour

| Variable | Effect |
|---|---|
| `TORCH_VK_CPU_FALLBACK=1` | Run an op that has no Vulkan kernel for its arguments on the CPU (with a one-time warning, counted in `fallback_counts()`) instead of raising `NotImplementedError`, the default. `0` keeps it off. See [The CPU fallback](#the-cpu-fallback). |
| `TORCH_VK_STRICT=0` or `1` | The older switch: `0` turns the CPU fallback on, `1` (the default) leaves it off. `TORCH_VK_CPU_FALLBACK` wins where both are set. |
| `TORCH_VK_CACHE_DIR=dir` | Directory for the compiled-pipeline cache (default `%LOCALAPPDATA%\torch-vulkan`, else `$XDG_CACHE_HOME/torch-vulkan`, else `~/.cache/torch-vulkan`). |
| `TORCH_VK_MAX_BATCH_GROUPS=n` | Submit a batch of queued kernels once it holds `n` workgroups (`0`: no limit). The default, 8192 on Intel GPUs and no limit elsewhere, keeps each submission short there: under Linux, i915 cancels GPU work still unfinished 20 s after it was submitted, and Mesa's driver does not report it. |
| `TORCH_VK_MAX_IN_FLIGHT=n` | Let at most `n` submitted batches be queued or running on the GPU at once (`0`: no limit); before submitting another, wait for the oldest. The default, 2 on Intel GPUs under Linux and no limit elsewhere, keeps the work queued there well under i915's 20 s limit, which counts queue time too. On those GPUs a batch the kernel cancelled anyway (a single kernel running for 20 s) is detected from its fence and raises an error, and every later call fails as on a lost device. |
| `TORCH_VK_BARRIERS=full` or `every` | Put a barrier before every command, instead of only where a command touches memory that one since the last barrier wrote, or writes memory one since then read (the default: kernels that touch disjoint memory then run concurrently; docs/log/M37). `every`: a barrier matched to the two commands' kinds (compute to compute between kernels), as torch-vulkan 0.2 did; `full`: a full compute and transfer barrier. Debugging switches, to rule out a missed dependency when a race is suspected. |
| `TORCH_VK_SPIN_US=n` | How long a synchronize polls for the GPU before it blocks in the driver's wait, in microseconds (default `500` on discrete GPUs, `0`, blocking at once, on integrated ones and lavapipe, which share the CPU's power or cores). Polling saves the wake-up after short work: about 19 us per synchronize on NVIDIA. |

### Choosing kernels

| Variable | Effect |
|---|---|
| `TORCH_VK_GEMM=tiled`, `float4` or `subgroup` | The matrix-multiply kernel: `float4` is the default on Arm GPUs, `subgroup` on other GPUs with subgroup size control, `tiled` elsewhere. `float4` shares operands across subgroups where the GPU runs 16-lane subgroups (Mali's Valhall GPUs; `TORCH_VK_DISABLE=subgroups` turns that off), and there also runs float16 and bfloat16 products, computed in float32. By default float16 and bfloat16 products use cooperative matrices (tensor cores) where available; setting this variable runs them in float32 instead. |
| `TORCH_VK_CONV=im2col` | Run stride-1 convolutions as im2col columns and a matrix product instead of the direct convolution kernel (`conv4s.slang`, the default where the GPU runs 16-lane subgroups: Mali, Intel). For comparison. |
| `TORCH_VK_FP16_ARITHMETIC=1` | Start with float16 arithmetic allowed (`set_fp16_arithmetic(True)`; see [float16 arithmetic](#float16-arithmetic)). |
| `TORCH_VK_GEMV=vector` or `scalar` | Matrix products with one to eight vectors (decoding steps: `x @ W.t()` for a few rows of `x`, `W @ v`, `addmv`, `_weight_int8pack_mm` with up to 8 rows): `vector` reads the weights in 16-byte loads and keeps float32 sums (`gemv4.slang`), `scalar` uses `gemv.slang` and the GEMM kernels for float weights and `qmm8.slang` for int8 ones. By default `vector`, except for float32 weights on NVIDIA and AMD GPUs. `_weight_int4pack_mm` always reads its 4-bit weights in 16-byte loads. |
| `TORCH_VK_SUBGROUP=8`, `16` or `32` | Subgroup size for the `subgroup` kernels (default 16 for float32, 8 for float64), within the device's range. |
| `TORCH_VK_COMPUTE_UNITS=n` | The GPU's SM or compute-unit count, which matrix products choose their tile shape and k split by (default: the driver's count where it reports one, `0`, unknown, elsewhere: then products are never split). |
| `TORCH_VK_SPLIT_K=n` | Split every float matrix product's k into `n` pieces where it can (`1`: never), instead of choosing by the GPU's size. The pieces' sums are added in a fixed order, so results are the same from run to run either way. For testing and tuning. |
| `TORCH_VK_GEMM_TILE=l`, `m` or `s` | The cooperative-matrix kernel's tile (128 x 256, 128 x 128 or 64 x 128) for every float16 and bfloat16 product, instead of choosing by the GPU's size. For testing and tuning. |
| `TORCH_VK_GEMM_GROUP=g` | Launch the cooperative-matrix kernel's tiles in bands of `g` tile rows (`0`: row by row). For testing and tuning. |
| `TORCH_VK_LINALG=cpu` | Send dense linear algebra to LAPACK on the CPU: a reference for testing and comparison, not a performance option. Turns the CPU fallback on (unless `TORCH_VK_CPU_FALLBACK=0`). |
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
| `TORCH_VK_BARRIERS=check` | Run each kernel alone and fail with an error if it changed device memory outside what it was declared to write (the declarations that let the stream leave barriers out; docs/log/M37). Very slow: it reads all live device memory twice per kernel. |
| `TORCH_VK_READ_LOG=dir` | Log a hash of everything the host reads back from the device (`.cpu()`, `.item()`) to `dir/reads-<pid>.txt`, so that two runs can be compared bit for bit (`tools/dev/barrier_ab.py`). |
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
