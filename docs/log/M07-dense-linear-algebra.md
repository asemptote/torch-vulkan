# Dense linear algebra (M7)

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
M5 GEMM (itself slower than MKL here), and each panel synchronises, leaving
the GPU idle while the CPU factors it. Look-ahead (factoring panel k+1 while
the GPU updates the rest) and a faster GEMM are the next steps; on an
integrated GPU like this one, `TORCH_VK_LINALG=cpu` is the faster choice.
