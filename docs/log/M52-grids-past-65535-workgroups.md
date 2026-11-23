# Grids past 65,535 workgroups (M52)

Vulkan guarantees 65,535 workgroups along each dimension of a dispatch
(`maxComputeWorkGroupCount`; NVIDIA allows 2^31 - 1 along x, 65,535
along y and z; lavapipe, Intel and Mali allow 65,535 along each).
csrc's kernels assume the guaranteed minimum, `kMaxGroups`. The model
benchmarks (#128, #131) found where that was a refusal: at batch 512, a
convolution of ghostnet_100 or mobilenetv3_large_100 became a matrix
product with 512 x 112 x 112 = 6.4M columns. That is 100,352 of
gemm.slang's 64-column tiles, so `gemm()` refused it, and with the CPU
fallback off by default (M45) the model failed with
`NotImplementedError`.

## The audit

Every launch whose grid grows with a tensor's size, and what it did past
the limit:

| launch | past 65,535 workgroups | now |
|---|---|---|
| `gemm()`: gemm.slang, gemmsg.slang, gemm4.slang, gemm4s.slang, gemmcm.slang (rows and columns of C in x and y, matrices in z) | refused (CPU fallback, or `NotImplementedError` without it); gemmsg's and gemm4's own checks fell through to gemm.slang, whose check refused | pieces of C (below) |
| split k (`split_k_gemm`, matrices times pieces in z) | ran unsplit | unchanged (a choice of speed, not a refusal) |
| bf16parts.slang (float32 at "high" or "medium": matrices in y) | ran the product in float32 | the matrices come in pieces of at most 65,280 |
| gemv.slang (`gemv`: rows / 8 in x) | fell through to the GEMM kernels | pieces of C |
| gemv4.slang (rows / 8 to rows / 16 in x) | from `gemm()`: fell through; from `_weight_int8pack_mm` (int8) to qmm8.slang; from `_weight_int4pack_mm` (4-bit) to a dequantised float32 product | row blocks folded into y |
| qmm8.slang, M <= 8 (`_weight_int8pack_mm`: N / 8 in x) | `RuntimeError` past 524,280 output columns | folded into y |
| qmm8.slang, M > 8 (N / 64 in x; rows already in pieces in y) | `RuntimeError` past 4,194,240 output columns | folded into z |
| qmm4.slang (`_dyn_quant_matmul_4bit`: N / 32 in x; rows already in pieces) | `RuntimeError` past 2,097,120 output features | folded into z |
| `bmm` with more than 65,535 matrices | in pieces where `out` was contiguous; refused otherwise | pieces in `gemm()`, for every product |
| conv4s.slang (`conv_direct`), dwconv.slang's weight gradient | another route (im2col and a product) | unchanged: routes, not refusals |
| factorisations (dense.cpp) | in pieces of 65,535 matrices | unchanged |
| element-wise, reductions, softmax, norms, pooling, sort, FFT | clamp to `kMaxGroups` and loop in the kernel | unchanged |

## Pieces of a product

`gemm()` (csrc/linalg.cpp) now wraps the routes, as `gemm_whole()`. A
product with more than 524,032 rows or columns of C, or more than 65,535
matrices, runs as products of pieces of C: views of A's rows, B's columns,
C and cin, at offsets (the kernels take every operand by address and
strides, so a piece needs no copy). Each piece is one dispatch, or a
split-k pair, through the same route choice as a whole product.

- 524,032 = 2,047 x 256: no kernel takes fewer than 8 rows or columns of C
  per workgroup (gemm4.slang's 8-row tiles, gemv.slang's and gemv4.slang's
  8 rows per workgroup), so a piece's grid stays within 65,535 along each
  dimension, whatever route takes it. Pieces start on multiples of 256,
  which are on every kernel's tile boundaries and keep the operands' 16-byte
  alignment for every dtype (gemm4's and gemmcm's staging need it).
- The pieces are about equal: a dimension of n is cut into ceil(n /
  524,032) pieces, multiples of 256 but the last. A thin last piece (n =
  524,033 would leave one column) could take another route, such as a
  matrix-vector product's, which 16-bit products on cooperative matrices
  do not have; so the last piece is always at least 256 long.
- The first piece decides whether the product runs on the GPU: if its route
  refuses (a dtype no kernel takes, say), nothing has been written and the
  caller falls back as before. The other pieces differ only in their
  offsets and the last ones' lengths, on which no refusal turns; that one
  then fails is an internal error.
- The 6.4M-column product above runs in 13 pieces of 494,080 columns
  (the last 493,568), each still thousands of workgroups: the 12 extra
  launches cost microseconds.

`TORCH_VK_GEMM_PIECE=n` (a developer switch) lowers the piece size, to
test pieces on every route with small products.

## Folding the grid

The matrix-vector and quantised kernels launch one workgroup per block of
output columns along x. There the extra blocks fold into the grid's y or
z (`folded_groups()` in vk.h): {min(blocks, 65,535), ceil(blocks / 65,535)}
workgroups, and the kernel takes block `gid.y * 65535 + gid.x`
(qmm8.slang's and qmm4.slang's tiled paths: `gid.z`, their y being rows).
The blocks past the last one load clamped rows and store nothing, as the
kernels' partial last blocks already did. That keeps one dispatch, and
these kernels needed no other change.

## A check at dispatch

A grid past the device's `maxComputeWorkGroupCount` is undefined
behaviour. The validation layer reports it (CI runs with it), but a run
without it got whatever the driver did. `Stream::dispatch_declared` now
refuses such a grid with an error naming the kernel, the grid and the
device's limits: a missed case is an error, not a wrong result.

## Tests

`tests/python/test_matmul.py`:

- `test_products_past_the_grid_limit`: on the default routes, products
  past the limit with k = 3 (fast on lavapipe): 9 x 4.2M and 4.2M x 9
  (float32, float16), 70,000 matrices into a strided `out=`, `baddbmm`,
  `addmv` and a vector-matrix product with 600,000 rows, and a linear layer
  with 600,000 outputs.
- `test_products_in_pieces`, with `TORCH_VK_GEMM_PIECE=512`, on every
  route (default, `TORCH_VK_GEMM=tiled` and `float4`, `TORCH_VK_GEMV=vector`
  and `scalar`, `TORCH_VK_SPLIT_K=2`; gemmcm.slang and bf16parts.slang
  where the device has cooperative matrices): float32, float64, int32,
  float16, bfloat16 and complex64 products in 2 to 3 pieces along rows,
  columns and matrices, with transposed, strided and broadcast operands,
  `addmm`, `baddbmm` and `out=`; and int8 weights with 600,000 output
  columns (gemv4.slang, or qmm8.slang's matrix-vector path).

`tests/python/test_quant.py::test_quantised_products_past_the_grid_limit`:
`_weight_int8pack_mm` with 4.2M output columns (qmm8.slang's tiles) and
`_dyn_quant_matmul_4bit` with 2.1M (bit for bit as the CPU).

On lavapipe the new tests take 140 s together, most of it the six
subprocesses of `test_products_in_pieces`. Without this change,
`test_products_past_the_grid_limit` fails on its first product.
