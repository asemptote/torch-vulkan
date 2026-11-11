# SVD through bidiagonalisation and divide and conquer (M28)

`svd` was a one-sided Jacobi SVD (M7): accurate, but O(n^3) per sweep for
many sweeps. Against CUDA it was 4x slower at n = 1024 and 6x at n = 4096 on
the H100 NVL, and it exceeded 90 s at n = 4096 on the L40S (issue #32). From
32 columns up, it now follows LAPACK's gesdd. Jacobi stays below that, and
`TORCH_VK_SVD_JACOBI=1` forces it for comparison.

- **Bidiagonalisation (zgebd2).** For m >= n (a wide matrix is transposed,
  as before), each column k takes a column reflector H_k and a row reflector
  G_k (the row conjugated around it, as zgebd2's zlacgv). This leaves a real
  upper bidiagonal B = Q^H A P, with diagonal d and superdiagonal e.
  - Each reflector is `shaders/bidiag.slang`: zlarfg on a strided vector,
    one workgroup per matrix.
  - The updates are matrix-vector products and rank-one updates on views of A.
- **B's SVD from its Golub-Kahan tridiagonal.** This is the 2n x 2n
  tridiagonal with a zero diagonal and off-diagonal (d_1, e_1, d_2, ...,
  d_n). Its eigenvalues are +-sigma_i, and the eigenvector of sigma_i
  interleaves v_i and u_i as (v_1, u_1, v_2, u_2, ...) / sqrt 2.
  - M26's divide and conquer (QL below n = 64) solves it.
  - The n largest eigenvalues, reversed, are the singular values (clamped at
    0). Their eigenvectors' halves give V_B and U_B.
- **Re-orthonormalisation.** A half is exactly v / sqrt 2 or u / sqrt 2
  while sigma is clear of the other singular values' -sigma: mixing with its
  own -sigma's eigenvector only rescales the halves. Small singular values
  close to others' negatives mix the halves of several, by
  eps |B| / (sigma_i + sigma_j); singular values at 0 leave them in any mix.
  - Without a fix, a spectrum graded down to 1e-14 gave U and V orthogonal
    to only 1e-6.
  - The halves are therefore replaced by their polar factor, the nearest
    orthonormal matrix. Newton-Schulz iterations, X <- X (3 I - X^H X) / 2,
    compute it in two GEMMs each, and the error squares every iteration.
  - When the columns are dependent (|X^H X - I| > 1/2), Householder QR
    orthonormalises and completes them instead.
  - QR is slow: at n = 1024 on the H100 it took 504 of 607 ms, so it is
    kept to the case that needs it.
- **Back-transform.** U = Q U_B, with Q built from the column reflectors.
  With full_matrices, U_B is extended as diag(U_B, I) and Q gives the
  complement. V = P V_B, with P's reflectors (the conjugated rows) read as
  columns of A^H. Both use apply_q, so M26's GEMM-built block reflectors.

## Results

**Accuracy** (UHD 620, float64 and complex128; about 1e-7 in float32):
singular values, the reconstruction U S V^H and the orthogonality of U and V
come within about 1e-15 of the CPU for these inputs:
- random matrices;
- rank-deficient matrices (rank n / 3 and n / 4);
- zero matrices;
- a spectrum graded from 1 down to 1e-14;
- tall and wide shapes, with and without full_matrices, and batches.

**Speed**, `benchmarks/vs_cuda.py`, svd float64 (ms; vk / CUDA):

| GPU | n | Jacobi | now | CUDA | now ÷ CUDA |
|---|---|---|---|---|---|
| H100 NVL | 1024 | 401 | 105 | 97 | 1.1 |
| H100 NVL | 4096 | 13731 | 1739 | 2211 | 0.79 |
| L40S | 1024 | 715 | 227 | 804 | 0.28 |
| L40S | 4096 | over 90 s | 5120 | 12792 | 0.40 |

Where the time goes at n = 4096 on the H100:
- bidiagonalisation: 1.21 s (two matrix-vector products and two rank-one
  updates per column, ten small operations);
- the Golub-Kahan eigenproblem: 0.33 s (8192 x 8192, by divide and conquer);
- the polar factor: 0.12 s;
- the back-transform: 0.08 s.

The bidiagonalisation is the next step. Its fused kernels, as M26 did for
the tridiagonal reduction, can share a blocked form with #33's.
