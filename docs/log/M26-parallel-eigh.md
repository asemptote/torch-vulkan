# Parallel symmetric eigensolver (M26)

M19's eigh followed LAPACK's dsyev: Householder reduction to a tridiagonal T
(dsytd2 / zhetd2), implicit QL on T (tqli) with its rotations accumulated into
Z, then the back-transform Q Z. QL is sequential: one GPU thread iterated it,
and at n = 4096 it took 12.9 s of the H100's 14.6 s (CUDA's cuSOLVER: 130
ms). eigh now follows dsyevd. Issue #27.

## Divide and conquer (shaders/stedc.slang, csrc/dense.cpp stedc)

LAPACK's dstedc (dlaed0 to dlaed4), every merge of a level at once.

- **Tearing.** T is torn into L leaves (L a power of two), each of at most
  32 rows. At each tear, d_p and d_{p+1} lose |e_p|, which the merges put
  back as a rank-one term.
  - **Padding.** n is padded to L s with diagonal entries above the spectrum
    (twice the Gershgorin bound, plus 1), coupled by 0. They stay their own
    eigenpairs, sort last and are dropped, so that every merge of a level has
    the same size and a level is one batch.
- **Leaves.** tridiag.slang's QL, batched over the leaves.
- **Merges.** QM = diag(Q1, Q2), z from QM's rows h - 1 and h (scaled by
  1/sqrt(2), rho = 2|e|), then:
  - **PREP:** the halves' sorted eigenvalues are merged, then deflated on one
    thread, as dlaed2 does it:
    - an entry with rho |z_i| <= 8 eps max(|d|, |z|) is an eigenpair as it
      stands;
    - two close entries are rotated so that one of them is.
  - **ROT:** applies those rotations to QM's columns.
  - **SECULAR:** finds the k roots of 1/rho + sum z_i^2 / (d_i - lambda), a
    thread per root, using dlaed4's "middle way":
    - each step models both poles that bound the root by rational functions
      and solves the quadratic, with bisection when a step leaves the
      bracket;
    - each root is kept as an origin pole plus an offset, so that every
      d_i - lambda_j is accurate.
  - **ZHAT:** Gu and Eisenstat's recomputed z (dlaed3), which makes the
    eigenvectors orthogonal to working precision.
  - **VEC:** the rank-one problem's eigenvectors, zhat_i / (d_i - lambda_j)
    normalised. They are written straight into U at their columns' places in
    QM, with unit columns for the deflated entries. Sorting the merged
    eigenvalues permutes U's columns, and QM U is one batched GEMM.
- **Below n = 64**, QL runs directly (`TORCH_VK_EIGH_QL=1` forces it, for
  comparison).
- **A pitfall:** Slang's bare floating literals are float32. 1/sqrt(2) as a
  float32 constant made the float64 eigenvalues 5e-9 off, and every double
  constant now has the `l` suffix.

## The other two phases

With QL gone, the back-transform took 1.1 s at n = 4096 on the H100. Of that,
1.06 s was factor.slang's FACTOR_LARFT, which builds each panel's T on one
workgroup per matrix.

- **block_t_gemm.** T now comes from T^-1 = diag(1/tau) + striu(V^H V): one
  batched product and a small triangular inverse. A reflector with tau = 0
  gets a zero column for the inverse and 0 on T's diagonal. geqrf uses it
  too, and the old block_t is gone.
- **Wider panels.** apply_q uses panels of 128 reflectors; any blocking
  applies the same Q.
- **The reduction** went from four dispatches per column to two:
  - SYMV computes w = A22 v and per-workgroup partial sums of w^H v;
  - RANK2 forms w' = tau w - 1/2 |tau|^2 (w^H v) v from those sums, the same
    order in every workgroup, so dsytd2's separate UPDATE_W step is gone.
  - RANK2 is tiled as 16 rows by 256 columns: no division per element.
  - One extra RANK2 workgroup updates the next column alone and then
    reflects it, which saves REFLECT's dispatch.

## Results

**Accuracy** (UHD 620): eigenvalues within a few ulps of the CPU's, residual
|A V - V Lambda| / |A| about 1e-15 and orthogonality about 3e-15 in float64
up to n = 1024. The test matrices were:
- randn, float32, float64 and complex128;
- a near-identity matrix (clusters);
- two repeated eigenvalues;
- the tridiagonal Laplacian.

float32's orthogonality, about 2e-6, is better than QL's 7e-6.

**benchmarks/eigh.py, UHD 620** (main, then this branch):

| case | main | this branch |
|---|---|---|
| eigvalsh float64 n=1024 | 4.71 s | 0.84 s |
| eigvalsh float32 n=1024 | 1.60 s | 0.37 s |
| eigh float64 n=1024 | 6.44 s | 1.05 s |
| eigh float64 n=512 | 0.97 s | 0.20 s |
| eigh complex128 n=1024 | 12.8 s | 2.82 s |
| eigh float32 n=1024 | 2.56 s | 0.43 s |
| eigh float32 n=2048 | lost the device (Windows' 2 s GPU timeout, QL's single thread) | 3.86 s |

The laptop's remaining time is mostly the reduction, which is
bandwidth-bound there.

**benchmarks/vs_cuda.py, eigh float64** (time and ratio to CUDA):

| GPU | n | before (ratio) | now | CUDA | now / CUDA |
|---|---|---|---|---|---|
| H100 NVL | 1024 | 882 ms (73x) | 53.9 ms | 12 ms | 4.5 |
| H100 NVL | 4096 | 14.6 s (110x) | 598 ms | 130 ms | 4.6 |
| L40S | 1024 | 1532 ms (59x) | 72.1 ms | 25.8 ms | 2.8 |
| L40S | 4096 | 23.7 s (53x) | 1351 ms | 446 ms | 3.0 |

On the H100 at n = 4096, the time splits into the reduction (470 ms),
divide and conquer (90 ms) and the back-transform (50 ms).

**Next:** the reduction is still one column at a time, so a dispatch pair
per column, and it reads A22 twice per column.
- dsytrd's blocked panels would turn half of that traffic into GEMMs.
- A two-stage reduction (band, then bulge chasing), as MAGMA's and
  cuSOLVER's, would turn most of it.
