# Matrix-vector products and LAPACK's eigh (M19)

- **GEMV** (`shaders/gemv.slang`): products with one output column or row
  (addmv, mv, vec @ mat, one sample through nn.Linear, each real part of a
  complex one) left 63 of gemm.slang's 64 tile columns idle, about 1 GB/s on
  the UHD 620. gemm() now hands them to a matrix-vector kernel whose thread
  layout follows the matrix's strides (32 threads per row along a contiguous
  k, or one thread per row in 4 k-groups for a transposed matrix): 15-18
  GB/s for float32 and float64 at n = 2000 and 4096, against MKL's 22-27 on
  the CPU. examples/rnn.py (an addmv a step): 30 ms a step down to 1.7.
- **eigh** as LAPACK computes it (`shaders/tridiag.slang`, csrc/dense.cpp's
  `tridiag_eigh`), replacing Jacobi from n = 4: Householder reduction to a
  real tridiagonal T (dsytd2 / zhetd2; per column a reflector, A22 v as a
  GEMV, the -1/2 tau (x^H v) v correction, a rank-2 update of both
  triangles), implicit QL with Wilkinson's shift on T (tqli) in one thread
  per matrix with d and e in shared memory, its rotations logged and applied
  to Z^T by one thread per column (a chain's shared row kept in a register),
  and the eigenvectors Q Z through apply_q's blocked reflectors. Jacobi stays
  for n < 4 (10^4 2 x 2 matrices: 18 ms against 94) and as the fallback if
  QL does not converge. benchmarks/eigh.py, float64 eigvalsh on the UHD 620:
  n = 512 from 28.2 s (Jacobi, its strided column passes bound by cache-line
  traffic) to 0.54 s; eigh with vectors 0.97 s. Accurate to working
  precision (residual and orthogonality ~2e-16 in float64).
- Still slow beside oneMKL (23 ms for that eigvalsh): QL runs in one thread,
  and a float64 division or square root costs ~1 us there. Next: a parallel
  tridiagonal solver (bisection and inverse iteration, or divide and
  conquer) and a blocked reduction (latrd's panel, syr2k as a GEMM).
- Two traps: per-lane copies of a float64 division and square root came out
  infinite in some lanes of a partly active SIMD thread on the UHD 620
  (computing them once in thread 0 and sharing them fixed it); and a
  kernel's shared memory must stay within 16 KiB, Vulkan's guaranteed
  maxComputeSharedMemorySize (the UHD 620 allows 32 KiB; the validation
  layer reports any excess).
