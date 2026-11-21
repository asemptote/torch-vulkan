# float32 products at torch.set_float32_matmul_precision (M41)

Issue #76. `torch.set_float32_matmul_precision("high")` or `("medium")`
lets PyTorch compute float32 matrix products at lower precision; CUDA then
uses TF32 tensor cores. torch-vulkan ignored the setting, so float32
products always ran on the shader cores (gemmsg.slang): 13-21 TFLOP/s on
the L40S, against 137-164 for bfloat16 on cooperative matrices (M27, M35).

## What the settings mean on "vk"

Vulkan has no TF32 cooperative matrices, so both reduced settings use
bfloat16 ones, as PyTorch's documentation of the setting allows:

- **"highest"** (the default): float32 products as before, the same bits.
- **"medium"**: A and B rounded to bfloat16 (to nearest even), one bfloat16
  product, summed in float32. Each product a b is off by at most
  (2 u + u^2) |a b|, u = 2^-8: 2^-7 + 2^-16 of its magnitude.
- **"high"**: each operand split into two bfloat16 parts, hi = bf16(x) and
  lo = bf16(x - hi) (x - hi is exact in float32), and
  a b ~ hi_a hi_b + hi_a lo_b + lo_a hi_b (Henry et al.,
  arXiv:1904.06376, which PyTorch's documentation cites for this setting).
  What is dropped is lo_a lo_b (at most 2^-16 |a b|) and the residuals
  past lo (2^-16 |x| each): under 3 (1 + 2^-8) 2^-16 |a b|, about 16 bits.

Both sum in float32 on the tensor cores, in their order, as the 16-bit
products already do. Results are the same from run to run: the parts are a
fixed function of the operands, and the products run gemmcm.slang with the
same tile and split-k choice (M35) as any bfloat16 product of their shape,
whose sums are fixed in order.

The setting is read at each product
(`at::globalContext().float32MatmulPrecision()`; if PyTorch refuses to
answer because the per-backend `fp32_precision` settings contradict the
legacy one, the product stays in float32). Where the device has no bfloat16
cooperative matrices (`TORCH_VK_DISABLE=bfloat16` or `coopmat`, any
`TORCH_VK_GEMM`), the setting changes nothing, as on a CUDA GPU without
TF32.

## Where it applies

`mm`, `bmm`, `addmm`, `baddbmm`, and through them `matmul`, `nn.Linear`,
`einsum` and the rest, forward and backward; also the four real products of
a complex64 product, as cuBLAS's complex products follow TF32 on CUDA. Not:

- **Convolutions.** PyTorch documents that the setting "does not change the
  precision of convolution operations" (cuDNN has its own switch). On "vk"
  a convolution is a product of an im2col view, so `convolution` and
  `convolution_backward` hold an `at::NoTF32Guard`, PyTorch's own scope for
  "float32 here", which the products check.
- **Factorisations and solves** (LU, QR, Cholesky, the eigensolvers, SVD,
  lstsq, triangular solves): their blocked updates are products, but on
  CUDA they run in cuSOLVER and MAGMA, which keep float32. Losing 8 bits in
  a trailing update would also make the factorisation useless. Each of
  dense.cpp's blocked routines holds a guard.
- **Pooling and interpolation** written as products (adaptive average
  pooling, the antialiased and linear upsamplings: `_kernels._apply_dims`).
  Python has no `NoTF32Guard`, so `_native.exact_float32_products(True or
  False)` enters and leaves one (a stack of them per thread, in linalg.cpp),
  wrapped in `_kernels.exact_products()`.
- **Quantised and sparse products** (`_weight_int4pack_mm`'s dequantised
  float32 product, `_scaled_mm`'s emulation, the sparse products' dense
  fallbacks): guarded, as their CUDA versions don't use TF32.
- **Matrix-vector products** (one row or column): they are bound by memory,
  so rounding the operands would buy nothing.

The math SDPA path (`at::matmul` of q and k) follows the setting, as on
CUDA's math backend.

## How: bfloat16 parts first, then a bfloat16 product

**First try: rounding in gemmcm's tile loads.** The issue suggested folding
the rounding (or the hi/lo split) into the kernel's staging, to avoid extra
passes over memory. gemmcm.slang gained float32 variants that loaded float32
pieces (two 16-byte loads per 8 values), held them in registers through the
products, and rounded or split them as they were stored to shared memory.
"high" staged hi and lo blocks of 16 values of k (the shared memory of 32)
and ran three products per fragment pair. It was correct, but slow on the
L40S:

| TFLOP/s | 1024^3 | 2048^3 | 4096^3 | 8192^3 |
|---|---|---|---|---|
| float32 | 14.6 | 18.7 | 20.7 | 16.8 |
| fused "medium" | 14.6 | 54.1 | 57.3 | 61.4 |
| fused "high" | 8.5 | 26.5 | 28.2 | 29.5 |

That is a third of the bfloat16 kernel's rate for "medium". Each block
loads twice the bytes, the conversions run on the shader cores between the
products, and the float32 registers held across them add to the 128 of
accumulators.

**What runs now.** A new kernel, `bf16parts.slang`, writes each operand's
bfloat16 parts once. The product is then gemmcm.slang's bfloat16 kernel, run
as it is:

- "medium": A' = bf16(A), B' = bf16(B), one product over k.
- "high": a single bfloat16 product over 3 k:

      A' = [lo_a | hc_a | hi_a]   (m x 3k)
      B' = [hc_b ; lo_b ; hi_b]   (3k x n)
      A' B' = lo_a hc_b + hc_a lo_b + hi_a hi_b

  The small terms come first in the sums. hc is hi where x is finite, else
  0 (below).

- k is padded to a multiple of 8 with zeros in each part, so that every row
  of the parts loads in 16-byte pieces.
- The parts are written contiguous along k or along the other dim,
  whichever the operand is contiguous along. Both the reads and the writes
  are then coalesced, and gemmcm.slang stages either layout (A_MODE and
  B_MODE 1 or 2) without copies.
  - The first version always wrote along k. `x.t() @ y` (a weight
    gradient) then ran at 0.6 TFLOP/s for "high" (1024 x 1024,
    k = 262144); it now runs at 26.
- A broadcast batch (stride 0) is split once.
- gemmcm.slang gains one spec constant, C_F32: C and Cin are float32 and C
  is stored unrounded. With SPLIT, the float32 workspace and splitk.slang's
  float32 variant finish it, as for any float32 product.

The cost is a pass over the operands (read 4 bytes, write 2 or 6 per value)
and temporary memory: 2 bytes per operand value for "medium", 6 for "high".
For square products that is a few per cent of the product's time. For thin
weight-gradient products (a small result over a long k) it is a larger
share: those read far more operand than they write result.

**Special values.**

- hi of a NaN is NaN; lo of an infinity or NaN is 0.
- A finite x whose rounding would overflow to infinity (above 3.39e38) is
  truncated instead, so finite operands give finite products, as in
  float32.
- With hi = inf and lo = 0, "high"'s hi_a lo_b would multiply an infinity by
  B's lo, which is 0 whenever b is exactly a bfloat16 value, giving NaN
  where float32 gives an infinity. The cross terms therefore use hc, which
  is 0 for infinities and NaN. The infinity reaches the result only through
  hi_a hi_b, which is NaN exactly where float32's product is (infinity
  times 0, or opposite infinities summed).

The fused version needed a workgroup-wide flag and skipped the cross terms
of any block holding an infinity; the parts make that unnecessary.

## Accuracy

Inputs from `torch.randn`, errors against the float64 product on the CPU,
L40S (driver 615.71.09):

- max rel: max |C - C64| / max |C64|;
- cw: max over entries of |C - C64| / (|A| @ |B|), the error per unit of the
  products' magnitudes, which the bounds above are stated in;
- fro: ||C - C64||_F / ||C64||_F.

| m x k x n | setting | max rel | cw | fro |
|---|---|---|---|---|
| 1024 x 1024 x 1024 | highest | 4.1e-7 | 1.1e-7 | 2.9e-7 |
| | high | 4.3e-6 | 1.1e-6 | 4.5e-6 |
| | medium | 2.1e-3 | 5.6e-4 | 2.3e-3 |
| | CUDA TF32 | 2.8e-4 | 7.3e-5 | 2.9e-4 |
| 4096 x 4096 x 4096 | highest | 3.0e-6 | 4.0e-7 | 1.1e-6 |
| | high | 7.9e-6 | 1.0e-6 | 6.9e-6 |
| | medium | 2.4e-3 | 3.2e-4 | 2.3e-3 |
| | CUDA TF32 | 3.1e-4 | 3.9e-5 | 2.9e-4 |
| 2048 x 8192 x 2048 | highest | 4.7e-6 | 4.1e-7 | 1.6e-6 |
| | high | 1.5e-5 | 1.3e-6 | 1.2e-5 |
| | medium | 2.5e-3 | 2.2e-4 | 2.3e-3 |
| | CUDA TF32 | 3.1e-4 | 2.7e-5 | 2.9e-4 |
| 256 x 65536 x 256 | highest | 1.2e-6 | 3.6e-8 | 1.1e-6 |
| | high | 3.3e-5 | 9.3e-7 | 3.4e-5 |
| | medium | 2.0e-3 | 5.7e-5 | 2.3e-3 |
| | CUDA TF32 | 2.7e-4 | 7.8e-6 | 2.9e-4 |

(CUDA: the same inputs through torch 2.14's cuBLAS on the same GPU, where
"high" and "medium" are both TF32.)

- **"high"** is within 6-30x of float32's own error: 1.3e-6 of the
  products' magnitudes at worst (about 2^-20), against its 2^-14 bound.
  Normwise it is 40-65x more accurate than TF32 up to k = 8192.
  - At k = 65536 the gap closes to 9x. There the float32 sums of 3 k
    products on the tensor cores dominate, not the dropped terms: "high"'s
    error grows with k as summation error does, and "highest", which sums
    on the shader cores, stays at 1.1e-6.
- **"medium"** keeps bfloat16's 8 bits: 2.3e-3 normwise at every size,
  about bfloat16's unit roundoff 2^-9 times the sqrt(2) of two rounded
  operands. That is 8x TF32's error (10 explicit bits against 7).

## Speed

`benchmarks/vs_cuda.py`: its float32 TF32 rows ran on CUDA only. They are
now `precision high` and `precision medium` rows at 2048^3, 4096^3 and
8192^3, and a weight gradient's shape, run on both devices. On CUDA both
settings mean TF32.

L40S (driver 615.71.09), TFLOP/s, medians of warm calls. CUDA is torch
2.14's cuBLAS on the same GPU; blanks were not measured.

| workload | highest | high | medium | CUDA float32 | CUDA TF32 |
|---|---|---|---|---|---|
| 1024^3 | 13.4 | 22.9 | 28.4 | | |
| 2048^3 | 18.7 | 42.2 | 87.0 | 43.3 | 111 |
| 3072^3 | 19.5 | 39.0 | 97.3 | | |
| 4096^3 | 20.7 | 46.0 | 117 | 41.4 | 99.2 |
| 8192^3 | 17.0 | 49.3 | 142 | 42.0 | 97.7 |
| 4096 x 4096, k = 65536 (x.t() @ y) | 16.1 | 40.3 | 110 | 42.8 | |
| 1024 x 1024, k = 8192 (x.t() @ y) | 16.0 | 27.0 | 59.1 | | |
| 1024 x 1024, k = 262144 (x.t() @ y) | 15.4 | 25.7 | 61.8 | 41.2 | 93.4 |

bfloat16 8192^3 for reference: 164 (CUDA 190).

- **"medium"** runs at 4.6-8.4x float32's rate on square products from
  2048^3 on. That is 87% of the bfloat16 product's own rate at 8192^3 and
  75% at 4096^3; the rest is the parts pass. It is faster than CUDA's TF32 from 4096^3 on (0.68x
  its time at 8192^3), and 1.3x slower at 2048^3.
- **"high"** runs at 2.0-2.9x float32's rate, about a third of bfloat16's,
  as three products must. That is 2-2.6x slower than CUDA's TF32, for
  40-65x its accuracy.
- Thin weight gradients gain less: their parts pass is long next to a
  small result (1024 x 1024 over k = 262144 reads 2 GB of operands for a
  4 MB result).
- With launch bands (`TORCH_VK_GEMM_GROUP=8`), "high" at 8192^3 ran at
  50.8 against 48.6, and nothing else changed by more than 1%. They stay
  off, as in M35.

H100 NVL: not measured. Its one GPU was held by another user's 48-hour job
for the whole of this work; see the pull request.

## Checks

- tests/python on the L40S: 1906 passed, 11 skipped. The new tests:
  - `test_float32_matmul_precision[high|medium]`: mm, bmm, addmm, baddbmm
    and linear on every layout, edge and strided view, a transposed result,
    NaN bias with beta = 0, and a weight gradient. Each result is within
    its precision's bound per product, and three runs give the same bits.
    Where there are bfloat16 cooperative matrices, the errors must also
    show the lower precision was used; elsewhere the results must be
    "highest"'s exactly.
  - `test_float32_matmul_precision_tiles`: the same on the m and s tiles,
    with banded launches and k split 2, 3 and 5 ways.
  - `test_float32_matmul_precision_special_values`: infinities, NaN, zeros
    and values above bfloat16's largest.
  - `test_float32_matmul_precision_scope`: conv2d, conv_transpose2d,
    bilinear interpolation, adaptive average pooling, cholesky, solve and
    qr give the same bits at "medium" as at "highest", while matmul's
    differ.
- test_matmul.py's precision, cooperative-matrix and split-k tests under
  the validation layer on the L40S: 18 passed, nothing reported. On
  lavapipe (no cooperative matrices) the precision tests pass (5, 6
  skipped).
- PyTorch's test_linalg.py product tests on the L40S (`-k "matmul or _mm or
  addmm or bmm or baddbmm or linear"`, default precision): 211 passed. The
  33 failures are the runner's expected ones: torch.compile (Inductor has
  no "vk" backend) and the 4-bit quantisation bound.
- `cargo test --release -p vkcore`: passed. Clippy runs in CI: the
  cluster's toolchain lacks it.
