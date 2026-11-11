# Linear algebra on the GPU (M13)

No LAPACK anywhere: every size and batch of float32/float64 matrices runs on
the GPU (`csrc/dense.cpp`, `shaders/factor.slang`).

- Blocked LU (rectangular too), Cholesky and Householder QR: the column-by-
  column steps LAPACK would do (panel factorisation with partial pivoting
  and whole-row interchanges, Cholesky diagonal blocks, Householder panels
  and the compact-WY `T` factor, inverses of small triangular blocks, row
  interchanges) run in `factor.slang` with one workgroup per matrix; the
  trailing updates are batched GEMMs. Pivots, Householder signs and
  reflector storage follow LAPACK, so `lu_factor`, `geqrf`, `qr`,
  `householder_product` and `ormqr` match the CPU's outputs.
- `eigh`/`eigvalsh`: cyclic Jacobi with round-robin ordering (all disjoint
  index pairs rotated in parallel), converged to working precision.
- `svd`/`svdvals`: one-sided (Hestenes) Jacobi; left vectors of zero
  singular values and `full_matrices` are completed with Householder QR.
- `lstsq` (minimum-norm solution through the SVD, as LAPACK's gelsy/gelsd
  return), `matrix_exp` (Pade 13 with scaling and squaring; float32
  computed in float64), `cholesky_solve`, `cholesky_inverse`, `lu_unpack`,
  `linalg.lu`, right-side and batched triangular solves.
- `TORCH_VK_LINALG=cpu` still sends these ops to LAPACK.
- Eigenvectors and singular vectors are unique only up to sign: tests check
  reconstruction and orthonormality, the OpInfo sweep compares magnitudes.
