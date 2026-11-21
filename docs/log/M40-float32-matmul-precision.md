# float32 products at torch.set_float32_matmul_precision (M40)

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
  (2 u + u^2) |a b|, u = 2^-8: under 2^-7 of its magnitude.
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
CUDA's math backend and the CPU's.

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

ACCURACY_TABLE

- "high" is within 3-30x of float32's own error: about 2^-17 to 2^-20 of
  the products' magnitudes, against the 2^-14 bound.
- "medium" loses 8-9 bits as expected (cw 2^-11 to 2^-14; 2.3e-3 normwise,
  about bfloat16's 2^-9 relative rounding times a random walk's factor of
  sqrt(2)).
- CUDA's TF32 (10 explicit bits) for comparison: CUDA_ACCURACY.

## Speed

`benchmarks/vs_cuda.py`: its float32 TF32 rows ran on CUDA only. They are
now `precision high` and `precision medium` rows at 2048^3, 4096^3 and
8192^3, and a weight gradient's shape, run on both devices. On CUDA both
settings mean TF32.

L40S, TFLOP/s:

SPEED_TABLE

H100 NVL: not measured. Its one GPU was held by another user's 48-hour job
for the whole of this work; see the pull request.

## Checks

CHECKS
