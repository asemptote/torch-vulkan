# Kernel routes

Ops with more than one GPU route: the kernels (`crates/vkcore/shaders`) each
op chooses between, the rule that chooses (from the code in
`crates/pyvk/csrc`, which is the reference when this page and the code
disagree), the switch that forces a route, and the devices the rule was
measured on. Switches are the developer switches in
[api.md](api.md#developer-and-tuning-switches-unstable-may-change-without-notice);
"shape" means a route follows from the arguments alone and has no switch.
A new route needs a benchmark, a switch or test hook, and its thresholds in
its design note: see [CONTRIBUTING.md](../CONTRIBUTING.md#adding-a-new-kernel-route).

Device features that decide many routes: 16-lane subgroups with shuffles
(Intel's GPUs and Mali's Valhall GPUs; `TORCH_VK_DISABLE=subgroups` hides
them), cooperative matrices with 32-lane subgroups (NVIDIA; hidden by
`TORCH_VK_DISABLE=coopmat`, bfloat16 ones by `bfloat16`), and the compute-unit
count (from the driver on NVIDIA, AMD and Arm, a table of Intel's discrete
GPUs, otherwise 0, unknown; `TORCH_VK_COMPUTE_UNITS` overrides it). CI runs
lavapipe, whose subgroup width follows its vector width (8 lanes at its
default 256 bits, so the 16-lane kernels run only in a pass at 512 bits);
no CPU driver has cooperative matrices, which are tested on hardware runs.

float16 and bfloat16 ops whose kernels decline them run in float32 on the
GPU and are rounded once (`half_via_float`, `backend.cpp`); for matrix
products and convolutions that is the float32 route below.

## Matrix products

`mm`, `bmm`, `addmm`, `baddbmm`, `addmv`, `matmul`, `linear`
(`linalg.cpp`, `gemm()`), tried in this order:

| Route | Kernels | Selected when | Force with | Tuned on |
|---|---|---|---|---|
| Matrix-vector, 16-byte loads | `gemv4` | m or n at most 8, one matrix, float32/float16/bfloat16, the matrix's k contiguous; by default everywhere except float32 weights on NVIDIA and AMD; float16 and bfloat16 with 2 to 8 vectors stay on cooperative matrices where those run float32's `gemv` | `TORCH_VK_GEMV=vector` / `scalar` | Mali-G57, HD 530, lavapipe (M38); the NVIDIA and AMD exception not measured |
| Cooperative matrices | `gemmcm_f16`, `gemmcm_bf16`, tiles `l` (128 x 256), `m` (128 x 128), `s` (64 x 128) | float16 and bfloat16 where the device has cooperative matrices of the type and `TORCH_VK_GEMM` is unset; not for one row or column, k = 0, or grids beyond 65535. Tile: `l` when the compute units are unknown; with at least as many large tiles as units, `l`, or `m` if that fills its last round of units 10 points better; with fewer, `l` if they fill more than two thirds of the units, else `l` with k split to fill them (k >= 2048, up to 8 ways), else (k < 2048) `s` | `TORCH_VK_GEMM_TILE`, `TORCH_VK_SPLIT_K`, `TORCH_VK_GEMM_GROUP` (launch order), `TORCH_VK_COMPUTE_UNITS` | H100 NVL, L40S (M22, M27, M35; tiles and splits on the L40S) |
| float32 in bfloat16 parts | `bf16parts`, then `gemmcm_bf16` | float32 products with `torch.set_float32_matmul_precision("medium")` (1 part) or `"high"` (3 parts) where bfloat16 cooperative matrices run, outside `NoTF32Guard` (convolutions) | the precision setting | L40S (M41); H100 not measured |
| 16-lane register tiles, A by shuffles | `gemm4s` (`gemm4s_f16h` with float16 arithmetic) | float32, and float16/bfloat16 (computed in float32), on Arm GPUs with 16-lane subgroups; float16 with `set_fp16_arithmetic(True)` on any GPU with them (Intel too); declines when the copies it would need cost more than 1/16 of the work | `TORCH_VK_GEMM=float4`, `TORCH_VK_FP16_ARITHMETIC` | Mali-G57 (M39), HD 530 for float16 arithmetic (M47) |
| One row or column | `gemv` | m or n is 1 (after `gemv4` declined) and the grid fits | shape; `TORCH_VK_GEMV=scalar` | not stated |
| Subgroup tiles | `gemmsg_f32` (16 lanes), `gemmsg_f64` (8 lanes) | float32 and float64 on GPUs with subgroup size control, except Arm; k split (k >= 1024, up to 16 ways) when the tiles fill at most two thirds of 8 workgroups per compute unit | `TORCH_VK_GEMM=subgroup`, `TORCH_VK_SUBGROUP`, `TORCH_VK_SPLIT_K` | UHD 620 (M20); the split on the L40S (M35) |
| float4 register tiles | `gemm4` | float32 and int32 on Arm GPUs (`gemm4s` first); operands copied into its layout when that costs at most 1/16 of the work, else the plain tiles | `TORCH_VK_GEMM=float4` with `TORCH_VK_DISABLE=subgroups` | Mali-G57 (M17; 2.5x slower on the UHD 620) |
| Register tiles | `gemm` | everything else: int64, int32 off Arm, float64 without `gemmsg` | `TORCH_VK_GEMM=tiled` | — |

A split k adds its pieces' sums in a fixed order (`splitk`), so results do
not depend on the split. Complex products run as four real products. A
`bmm` whose batch is broadcast runs as one `mm`; more than 65535 matrices run
in chunks.

## Quantised products

| Op | Routes | Selected when | Force with | Tuned on |
|---|---|---|---|---|
| `_weight_int8pack_mm` | `gemv4` (`q8`); `qmm8` rows mode; `qmm8` 64 x 64 tiles | up to 8 rows: `gemv4` where `use_gemv4` (as above), else `qmm8` rows; more rows: tiles | `TORCH_VK_GEMV=scalar`; shape | Mali-G57 (M38) |
| `_weight_int4pack_mm` | `gemv4` (`q4`); dequantise to float32 and a matrix product | up to 8 rows: `gemv4` (whatever `TORCH_VK_GEMV` says) unless it declines (alignment, int16 storage); more rows: dequantise | shape | Mali-G57 (M38) |

## Convolution

`convolution`, `conv1d`/`2d`/`3d`, their transposes and backward
(`convolution.cpp`). The forward, in order:

| Route | Kernels | Selected when | Force with | Tuned on |
|---|---|---|---|---|
| Direct | `conv4s` (`conv4s_f16h` with float16 arithmetic); tile `r4n2` on Mali, else `r8n2` with float16 arithmetic, `r8n1` without | GPUs with 16-lane subgroups; on Mali only for float16 arithmetic, or float16/bfloat16 with 5 or more inner taps. Stride 1, inner dilation 1, 2, 3 or 5, at least 3 inner taps, not depthwise with at most 32 taps, 2-D/3-D outputs at least 8 wide, at least 2048 output positions over the batch, sizes within 32-bit offsets and 65535 workgroups | `TORCH_VK_CONV=direct` / `im2col`, `TORCH_VK_FP16_ARITHMETIC` | HD 530, Mali-G57 (M47; the 2048 and width rules on both) |
| float16/bfloat16 in float32 | the float32 routes below | 16-bit inputs `conv4s` did not take: computed in float32, rounded once | shape | — |
| Depthwise multiply-adds | element-wise kernels | one input channel per group and at most 32 taps: one multiply-add per tap | shape | HD 530 (M46); the 32 is not measured |
| Rows layout | im2col columns and `bmm` | one group, or one sample: W [Cout, Cin K] @ columns, bias as the product's beta | shape | Mali-G57, HD 530, lavapipe (M39) |
| Columns layout | im2col columns and `bmm` | groups > 1 and a batch: columns [G, N L, Cin K] @ W^T | shape | — |

Transposed convolutions (and the input gradient of a convolution) run by
phases, one stride-1 convolution per output phase (sub-pixel), unless no
stride exceeds 1 or a dilation of at least the stride leaves the phases as
many taps as the kernel; then by zero insertion: the input with zeros
inserted, convolved with the flipped weight. Either way the convolutions
take the forward routes above. Phases were measured on the Mali-G57,
HD 530, lavapipe and L40S (M46). The weight gradient is always im2col
columns and `bmm`. Convolution products stay in float32 at every
`float32_matmul_precision`.

## FFT

`fft_fft`, `fft_ifft`, `fft_rfft`, `fft_irfft` and the n-dimensional ones
(`fft.cpp`), per row of length n. No switch: tests reach the routes by
size (`tests/python/test_complex_fft.py`).

| Route | Kernels | Selected when | Tuned on |
|---|---|---|---|
| One radix pass | `fft_radix` | power-of-two n up to 1024 (complex64) or 512 (complex128) | UHD 620 (M21), H100, L40S (M31) |
| Shared-memory row | `fft` (`FFT_LOCAL`) | power-of-two n up to twice that | UHD 620 (M21) |
| Register passes | `fft_reg` (`e16`: 16 values per thread, complex64 up to 2^20; `e8` otherwise) | power-of-two n from 2^14 to 2^30 where the GPU has 32 KiB of shared memory (complex64 with `e8` needs 16 KiB); two passes up to 2^20, three beyond; not complex128 on integrated Intel GPUs under Windows | H100 NVL, L40S (M32); the Windows exception on the UHD 620 (driver 101.2134; code comment only) |
| Radix passes | `fft_radix` | other power-of-two n: ceil(log2 n / 7) passes of radix up to 128 | H100, L40S (M31) |
| Bluestein | chirp, a power-of-two FFT of length at least 2n - 1 | n not a power of two | — |

Real transforms of even length run as a complex transform of half the
length, with the real-to-complex split (or the complex-to-real fold) fused
into the register passes' last (or first) pass where those run, else as a
pass of its own; odd lengths run as full complex transforms. float16 and
complex32 run in float32 and complex64.

## Dense linear algebra

(`dense.cpp`.) `TORCH_VK_LINALG=cpu` sends all of it to LAPACK on the CPU,
as a reference for tests; it is not a route.

| Op | Routes | Selected when | Force with | Tuned on |
|---|---|---|---|---|
| `eigh`, `eigvalsh` | Jacobi (`factor`); tridiagonal reduction (`latrd`) then implicit QL (`tridiag`) or divide and conquer (`stedc`, leaves of at most 32 by QL) | n < 4: Jacobi; 4 <= n < 64: reduction and QL; n >= 64: reduction and divide and conquer. Jacobi again if QL does not converge | `TORCH_VK_EIGH_QL=1`: QL from n = 4 up | n < 4 on the UHD 620 (M19); 64 and 32 not stated (M26 measured the H100 NVL, L40S and UHD 620 at larger sizes) |
| `svd` | one-sided Jacobi (`factor`); bidiagonalisation, divide and conquer on the Golub-Kahan tridiagonal, then orthonormalisation (Newton-Schulz, or Householder QR when far from orthonormal) | the smaller dim < 32: Jacobi; else bidiagonalisation, Jacobi if it fails | `TORCH_VK_SVD_JACOBI=1` | 32 not stated (M28 measured the H100 NVL and L40S at n = 1024 and 4096) |

LU, QR, Cholesky, triangular solves and `eig` have one route each.

## Other ops

These choose by shape or layout only, with no switch; the project's tests
cover each route by shape.

| Op | Routes |
|---|---|
| Reductions (`reduce.cpp`) | one thread per output where there are at least 256 outputs and the reduced dim is strided with contiguous outputs, or at most 32 long (lavapipe); else workgroups per output, with the reduced dim split in chunks of at least 2048 elements over up to 1024 workgroups and further passes; 1-D `max`/`min` of 2^31 or more elements in rows of 65536 |
| `index_add`, `scatter_reduce`, `index_put` with accumulation (`indexing.cpp`) | whole rows by `segreduce` when dim 0 is contiguous and the index expanded; else per element; indices beyond 2^26 elements in slices |
| Sparse products (`sparse.cpp`) | sparse @ sparse by `spgemm` without dense dims, else densified; dense @ sparse computed transposed; compressed layouts through COO |
| Copies (`launch.cpp`), and `flip` as a copy with the flipped dims' strides negated | `copy8` for 5 to 8 dims after merging, else the strided kernels with a dispatch per outer index |
| Interpolation, grid sampling (`python/torch_vulkan/_kernels.py`) | uint8 bilinear/bicubic/antialiased resizes in fixed point; float resizes by `resample` or by products; `grid_sample` bicubic composite, nearest and bilinear by `gridsample` |

## Tests that force routes

`tests/python` forces the switchable routes in subprocesses:
`test_matmul.py` (`TORCH_VK_GEMM`, `TORCH_VK_SUBGROUP`, `TORCH_VK_SPLIT_K`,
`TORCH_VK_GEMM_TILE`, `TORCH_VK_GEMM_GROUP`, `TORCH_VK_COMPUTE_UNITS`,
`float32_matmul_precision`), `test_gemv4.py` (`TORCH_VK_GEMV`), `test_nn.py`
(`TORCH_VK_CONV`), `test_linalg_gpu.py` (`TORCH_VK_EIGH_QL`,
`TORCH_VK_SVD_JACOBI`) and `test_fp16_arithmetic.py`. A forced route still
runs only where the device can run its kernel (16-lane subgroups,
cooperative matrices), and the tests skip elsewhere. The routes chosen by
shape alone are covered by the shapes in the tests: the gemv grid limit,
the `gemm4` copies, depthwise and the two im2col layouts, phases and zero
insertion, and the FFT routes the device's shared memory allows (radix
passes beyond 2^14 only where the register passes do not fit). The Jacobi
fallbacks of `eigh` and `svd` depend on the data and are not forced.
