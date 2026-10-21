# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M8 — element-wise completeness**: device `"vk"` is a native PyTorch
backend (C++ on PrivateUse1). Element-wise ops (float32, float64, complex64,
complex128: arithmetic, comparisons, logical ops, the transcendental and
special functions, activations, clamp, where/masked_fill, lerp/addcmul/addcdiv),
reductions (sum,
mean, prod, max/min, argmax, var/std, norms, any/all), scans (cumsum,
cumprod), matrix products (mm, bmm, matmul, addmm, baddbmm, addmv, dot,
nn.Linear), FFTs (all of `torch.fft`, any length) and dense linear algebra
(solve, inv, det, slogdet, lu_factor/lu_solve, cholesky, solve_triangular,
tril/triu) run on the GPU; every other op runs on the CPU through PyTorch's boxed
fallback. Tested on an Intel UHD 620 with the validation layer clean.

```python
import torch, torch_vulkan

x = torch.randn(1_000_000, dtype=torch.float64, device="vk")  # random fill runs on the CPU
y = (x * 2 + 1).relu()          # three Vulkan kernels, batched
print(y[:3], y.sum())           # sum is a Vulkan reduction
torch_vulkan.fallback_counts()  # {'aten::normal_': 1, ...}  (printing uses a few CPU ops: masked_select, bool &/*)
```

## Element-wise ops (M8)

Every dense element-wise aten op on float32/float64/int32/int64/bool now has a
Vulkan kernel: the remaining transcendental and special functions (`asin` ...
`atanh`, `log10`, `log1p`, `expm1`, `erf`, `erfc`, `erfinv`, `lgamma`,
`digamma`, `sinc`, `frac`), binary math (`pow`, `atan2`, `fmod`, `remainder`,
`floor_divide`, `logaddexp`, `xlogy`, `hypot`, `copysign`, `nextafter`,
`heaviside`, `fmax`/`fmin`, shifts, `gcd`/`lcm`), activations (`gelu`, `silu`,
`mish`, `softplus`, `elu`/`selu`, `logsigmoid`, `logit`, `hardtanh`,
`hardswish`, `leaky_relu`, shrinks, `threshold`), `clamp` (scalar and tensor
bounds), predicates (`isnan`, `isinf`, `signbit`, ...), logical ops, `where`,
`masked_fill`, `lerp`, `addcmul`, `addcdiv`, `nan_to_num`, `round(decimals)`.

Math precision, by dtype:

- **float32** uses the GPU's GLSL built-ins (`exp`, `log`, `sin`, `pow`,
  `atan2`, ...), so its accuracy follows Vulkan's precision rules, plus
  `shaders/_math32.slang` for functions with no built-in (`erf`, `erfinv`,
  `lgamma`, `digamma`, `log1p`, `expm1`, `sinpi`) and short wrappers where a
  built-in mishandles edge cases or is too inaccurate on Intel (`tanh`
  overflow, `sinh`/`asinh` cancellation, `acos` and `tan` accuracy, C99
  special values of `pow`/`atan2`). float32 kernels contain no float64 code,
  so they run on GPUs without `shaderFloat64` (a Rust test checks the SPIR-V).
- **float64** uses the software library in `_math64.slang` and
  `_math64_ext.slang` (SPIR-V has no 64-bit transcendentals), agreeing with
  the CPU to a few ULP.
- `fmod`/`remainder`/`floor_divide` are exact for both.

`shaders/_fmath.slang` gives both under one set of names (`m_exp`, `m_pow`,
...), so `unary.slang` and `binary.slang` are each one kernel per dtype.

Infrastructure cleanup in the same step:

- Operation codes live in `shaders/opcodes.txt`; the build generates
  `opcodes.slang` for the shaders and `opcodes.h` (`vk::op::UNARY_EXP`, ...)
  for C++, so the two cannot disagree.
- One strided launcher (`csrc/launch.cpp` + `csrc/strided.h`) packs scalar
  parameters per kernel dtype (int64 / double / float) for the three- and
  four-operand layouts; push-constant sizes and strides are 4-vectors, since
  Intel's driver misreads dynamically indexed push-constant arrays.
- Kernels are compiled with scalar block layout and denormal preservation
  (stripped at load time on devices that cannot preserve denormals); loading
  a kernel that needs float64/int64/int8 on a device without it is a clear
  error. Shader variants are rebuilt only when their inputs change.
- The C++ static library is linked whole, so every file's kernel
  registrations are kept without link-anchor functions.
- `opinfo_coverage.py` runs OpInfo samples in parallel worker processes:
  272 of 659 ops now run fully on the GPU (170 at M7).

## Dense linear algebra (M7)

- `csrc/dense.cpp`: blocked right-looking LU with partial pivoting and
  Cholesky, panels of 64 columns. Each panel / diagonal block is factored by
  LAPACK on the CPU (memory is shared: a copy plus a sync); the O(n³)
  trailing updates run on the GPU as GEMMs. Triangular solves apply inverses
  of the 64×64 diagonal blocks (computed on the CPU in float64) with GEMMs.
  Row interchanges are applied by a gather/scatter kernel
  (`shaders/rows.slang`) to the rows that move.
- Ops: `linalg.lu_factor(_ex)`, `linalg.solve(_ex)` (vector or matrix right-hand
  side), `linalg.inv(_ex)`, `linalg.det`, `linalg.slogdet`, `linalg.lu_solve`,
  `linalg.cholesky(_ex)` (lower/upper), `linalg.solve_triangular`, plus
  `tril`/`triu` kernels (`shaders/triangle.slang`, any dtype, batched).
  Pivots match LAPACK's exactly; `info` and `check_errors` behave as on the
  CPU (singular / not positive-definite matrices raise `LinAlgError`).
- The GPU path takes single float32/float64 matrices with n ≥ 128; smaller or
  batched inputs, right-side solves, QR, eigh and SVD run on the CPU.
  `TORCH_VK_LINALG=cpu` sends everything to LAPACK.

Measured (`python benchmarks/linalg.py`, UHD 620 vs 4-thread CPU with MKL):

| n = 2048 | vk float32 | CPU | vk float64 | CPU |
|---|---|---|---|---|
| `solve` (8 right-hand sides) | 494 ms | 112 ms | 636 ms | 216 ms |
| `cholesky` | 283 ms | 64 ms | 458 ms | 101 ms |
| `inv` | 587 ms | 305 ms | 1640 ms | 572 ms |

Correct, but slower than MKL on this machine: the trailing updates use the
M5 GEMM (itself slower than MKL here), and each panel synchronizes, leaving
the GPU idle while the CPU factors it. Look-ahead (factoring panel k+1 while
the GPU updates the rest) and a faster GEMM are the next steps; on an
integrated GPU like this one, `TORCH_VK_LINALG=cpu` is the faster choice.

## Complex numbers and FFT (M6)

- complex64 / complex128 tensors (stored as float2 / double2): casts to and
  from every real dtype, fills, `+ - * /` (with complex scalars and real
  `alpha`; division uses Smith's algorithm), `neg`, `abs` (overflow-safe),
  `conj` / `conj_physical`, and the views `real`, `imag`, `view_as_real`,
  `view_as_complex` (they alias: `x.real += 1` updates `x`).
- Lazy conjugate/negative bits: `_copy_from` passes through PyTorch's
  Conjugate/Negative layers and applies the bits itself (otherwise
  materializing them recursed forever). The CPU fallback now refuses view
  ops (`error_on_views`) instead of silently returning copies.
- `shaders/fft.slang` + `csrc/fft.cpp`: `_fft_c2c`, `_fft_r2c`, `_fft_c2r`,
  which every `torch.fft` function reduces to. Power-of-two lengths use
  radix-2 Stockham passes (rows up to 2048 complex64 / 1024 complex128 run
  all passes in shared memory in one dispatch); other lengths use
  Bluestein's algorithm. Twiddle and chirp tables are computed on the host in
  float64 (the GPU's float32 sin/cos would be far too inaccurate) and cached.
  r2c transforms as complex and keeps n/2+1 bins; c2r rebuilds the Hermitian
  spectrum and keeps the real part. Error vs PyTorch's CPU FFT is a few ULP x
  log2(n); a 65536-point complex128 FFT agrees to < 1e-11.

Measured (`python benchmarks/fft.py`, UHD 620 vs 4-thread CPU):

| | vk complex64 | CPU | vk complex128 | CPU |
|---|---|---|---|---|
| fft 2^20 | 33 ms | 14 ms | 49 ms | 25 ms |
| fft2 1024×1024 | 12 ms | 6 ms | 27 ms | 12 ms |
| fft 1,000,003 (prime, Bluestein) | 119 ms | 212 ms | 249 ms | 345 ms |
| rfft 2^22 | 127 ms | 62 ms | 273 ms | 104 ms |

The FFTs are memory-bound radix-2 passes plus transposes between dims; on
this integrated GPU only long prime-length transforms beat the CPU so far.
Radix-4/8 passes, a shared-memory four-step FFT for long rows, and a real
r2c (half-length complex) path are the next steps.

## Matrix products (M5)

- `shaders/gemm.slang` + `csrc/linalg.cpp`: one batched GEMM,
  `C = alpha * A @ B + beta * Cin`, used by `mm`, `bmm`, `addmm`, `baddbmm`
  and `addmv` (PyTorch composes `matmul`, `mv`, `nn.Linear`, `einsum` from
  these); `dot` is a product plus a sum reduction. Operands are addressed by
  (row, column, batch) strides, so transposes (autograd's `A.t() @ grad`),
  slices and broadcast batches need no copies. `beta = 0` ignores `Cin`
  entirely (NaNs in it do not leak through), as on the CPU.
- 64x64 output tiles per 16x16 workgroup, K staged through shared memory in
  slices of 16, a 4x4 register block per thread. Larger tiles (128x128/8x8,
  128x64, 64x128) were measured and are slower on the UHD 620.
- float32 and float64; integer matrix products run on the CPU.

Measured (`python benchmarks/matmul.py`, UHD 620 vs 4-thread CPU with MKL):

| | vk | CPU |
|---|---|---|
| float32 1024³ | 55 GFLOP/s | 193 GFLOP/s |
| float32 2048³ | 87 GFLOP/s | 140 GFLOP/s |
| float64 2048³ | 23 GFLOP/s | 69 GFLOP/s |

On this integrated GPU, matrix products are still slower than MKL on the
CPU (the kernel reaches ~20% of the GPU's float32 peak); a subgroup-based
kernel and per-GPU tuning are future work. On discrete GPUs the balance differs.

## Reductions, scans, float64 math (M4)

- `shaders/reduce.slang` + `csrc/reduce.cpp`: tree reduction over any set of
  dims (up to 4 non-mergeable kept and reduced dims each). Large reductions
  are split into chunks across workgroups and the partials reduced again, so
  float32 sums stay as accurate as the CPU's. When the reduced dims are
  strided and outputs are adjacent (column reductions), each thread owns one
  output so reads are coalesced. Integer and bool inputs accumulate in int64,
  as in PyTorch. Ops: `sum`, `mean`, `prod`, `amax`/`amin`, `max`/`min`
  (NaN-propagating), `var`/`std` (two passes, any correction),
  `linalg.vector_norm` (ord 1, 2, ±inf), `any`/`all`.
- `shaders/argreduce.slang`: (value, index) reduction for `argmax`/`argmin`
  and `max`/`min` along a dim; ties and NaNs resolve to the first index, as on
  the CPU.
- `shaders/scan.slang`: `cumsum`/`cumprod` along any dim: per-chunk workgroup
  scans, the chunk totals scanned recursively, then added back.
- `shaders/_math64.slang`: float64 `exp`, `exp2`, `log`, `log2`, `sin`, `cos`,
  `tan`, `tanh`, `sigmoid` (SPIR-V has no 64-bit transcendentals). Measured
  against PyTorch's CPU over 200k inputs each plus special values: exp/exp2 ≤ 1
  ULP, log/log2/sigmoid/tan ≤ 2 ULP, tanh ≤ 3 ULP, sin/cos ≤ 1 ULP of scale.
  sin/cos/tan reduce arguments exactly up to |x| ≈ 1.6e6.
- Kernels preserve denormals (`-denorm-mode-fp32/fp64 preserve`): the UHD 620
  otherwise flushes subnormal float64 results to zero.

Measured (`python benchmarks/elementwise.py`, 16M elements, UHD 620 vs 4-thread CPU):

| | vk float32 | CPU float32 | vk float64 | CPU float64 |
|---|---|---|---|---|
| `sum()` | 2.5 ms | 2.7 ms | 8.7 ms | 4.9 ms |
| `var(dim=1)`, 4096×4096 | 6.3 ms | 93 ms | 17 ms | 88 ms |
| `amax(dim=0)`, 4096×4096 | 4.7 ms | 6.3 ms | 13 ms | 14 ms |
| `cumsum` | 23 ms | 31 ms | 52 ms | 34 ms |

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

Precision: float32 math uses the GPU's built-ins, which Vulkan only requires
to be accurate to ~2^-11 absolute (sin/cos) or a few ULP; tests use those
bounds. float64 uses `_math64.slang` (see M4 and M8).

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
  shaders/opcodes.txt      operation codes shared by shaders and C++ (generated headers)
  shaders/_strided*.slang  shared push-constant layouts + strided index math
  shaders/_fmath.slang     float math by dtype: built-ins + _math32 (float32), _math64* (float64)
  shaders/test/            kernels for vkcore's own tests (feature test-kernels)
  tests/runtime.rs         GPU tests (batching, dependencies, reuse, copies, pool, cache)
  examples/smoke.rs        hardware check
  examples/dispatch_bench.rs  per-op overhead, synchronized vs batched
crates/pyvk/               extension module torch_vulkan._native
  csrc/backend.cpp         allocator, device guard, hooks, views, resize, copies, CPU fallback
  csrc/launch.cpp          strided element-wise launches (csrc/strided.h plans dims)
  csrc/ops.cpp             unary, binary and comparison aten kernels
  csrc/elementwise.cpp     activations, clamp, predicates, where, lerp/addcmul, compositions
  csrc/reduce.cpp          reductions, arg-reductions, scans
  csrc/linalg.cpp          matrix products (batched strided GEMM)
  csrc/fft.cpp             FFTs (Stockham + Bluestein, r2c/c2r)
  csrc/dense.cpp           blocked LU / Cholesky / triangular solves, tril/triu
  src/ffi.rs               C ABI from C++ to the Rust runtime
  src/lib.rs               Python functions (synchronize, stats, fallback counts)
  build.rs                 compiles csrc/ against the target Python's libtorch
python/torch_vulkan/       backend naming and torch.vk device module
tests/python/test_*.py     419 tests: results on "vk" vs CPU, no unexpected fallbacks
tools/opinfo_coverage.py   runs PyTorch's OpInfo samples on "vk": GPU / fallback / mismatch per op
.github/workflows/ci.yml   Linux CI on lavapipe (Mesa's CPU Vulkan) with validation
benchmarks/elementwise.py  per-op overhead, element-wise, reduction and scan throughput vs CPU
benchmarks/matmul.py       matrix multiply GFLOP/s vs CPU
benchmarks/fft.py          FFT timings vs CPU
benchmarks/linalg.py       solve / cholesky / inv timings vs CPU
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

All 15 Rust tests (8 unit tests, 7 GPU tests) and both examples run with no
validation messages.

Python package (editable install into `.venv`, CPU build of PyTorch):

```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest maturin
.\.venv\Scripts\maturin develop --release --uv   # compiles csrc/ against this venv's torch
$env:TORCH_VK_VALIDATE = "1"; .\.venv\Scripts\python -m pytest -q
```

All 419 Python tests pass with no validation messages. Op coverage:
`python tools/opinfo_coverage.py --jobs 4` (writes `tools/opinfo_coverage.json`).

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. ~~**M2 Python prototype**~~: done; superseded by M3.
3. ~~**M3 native backend**~~: done (see above). Not yet: bool binary ops
   (`&`, `*` on masks), multiple devices, pinned memory, a GPU random-number
   generator (random fills run on the CPU).
4. ~~**M4 reductions, scans, float64 math**~~: done (see above). Not yet:
   `logcumsumexp`, `cummax`, `median`/`sort`-based ops, `masked_select`,
   float64 `pow`/`atan2`, the arg-reduction thread-per-output mode, and
   Payne-Hanek reduction for huge sin/cos arguments.
5. ~~**M5 matrix products**~~: done (see above). Not yet: a faster
   subgroup/cooperative-matrix kernel, integer and half-precision GEMM,
   convolutions.
6. ~~**M6 complex numbers + FFT**~~: done (see above), with an own FFT
   rather than VkFFT (VkFFT's Vulkan backend needs the raw device/queue
   handles and runtime glslang). Not yet: complex reductions, complex
   matmul, complex transcendentals, faster FFT kernels.
7. ~~**M7 dense linear algebra**~~: done (see above). Not yet: blocked QR
   (compact-WY updates), batched GPU paths, look-ahead, eigh/SVD on the GPU.
8. ~~**M8 element-wise completeness**~~ and infrastructure cleanup: done (see
   above). Not yet: int8/uint8/int16/float16/bfloat16 kernels, complex
   transcendentals (M14), `polygamma`, Bessel and other `torch.special`
   functions.

Goal for M9-M14: every dense aten op runs on Vulkan (`TORCH_VK_STRICT=1` with
no fallbacks in the OpInfo sweep).

9. **M9 data movement and indexing**: cat, flip, roll, gather/scatter,
   index/index_put, masked_select, nonzero, arange/eye/linspace.
10. **M10 sorting, searching and statistics**: sort, topk, median, unique,
    histc, searchsorted, cummax, logcumsumexp.
11. **M11 random numbers**: Philox-based rand/randn/bernoulli/multinomial.
12. **M12 neural-network ops**: convolution, pooling, softmax, normalization,
    losses, backward kernels.
13. **M13 linear algebra on the GPU**: no CPU panels; QR, eigh, SVD.
14. **M14 complex completion and a strict sweep with zero fallbacks**.
8. **Testing throughout**: PyTorch OpInfo samples compared against CPU.
