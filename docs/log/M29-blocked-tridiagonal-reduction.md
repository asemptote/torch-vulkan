# Blocked tridiagonal reduction, faster merges (M29)

After M26, eigh float64 on the H100 NVL took 3.4x CUDA's time at n = 1024
and 4.6x at n = 4096. At n = 4096, most of that time was the reduction to
tridiagonal form: two dispatches per column, and each column read A22 twice
and wrote it once. Issue #33 set the targets: at most 3x CUDA at n = 4096
and 5x at n = 1024.

## Blocked reduction (shaders/latrd.slang, csrc/dense.cpp latrd_reduce)

This follows LAPACK's zhetrd with zlatrd's panels (lower).

- **Panels.** A panel of NB = 32 columns is reduced one column at a time
  against A22 as it was at the panel's start. The panel keeps its reflectors
  V and vectors W, with A_now = A_start - V W^H - W V^H.
- **GEMM update.** After each panel, one batched GEMM updates the trailing
  matrix: A22 -= [V W] [W V]^H (zher2k on both triangles).
- **Traffic.** A step now reads A22 once. Half the reduction's arithmetic
  moves into the GEMMs.
- **Two dispatches per column.**
  - SYMV: every workgroup forms column k's reflector (zlarfg) from a
    contiguous copy of the column. The arithmetic is the same in each, so
    each gets the same v, and workgroup 0 writes it out. Then y = A22_start v,
    eight rows per workgroup. The partial sums of y^H v, V^H v and W^H v
    over the workgroup's rows go to a buffer.
  - FIN: every workgroup adds up those partial sums in one fixed order, which
    gives x1 = V^H v and x2 = W^H v. A thread per row then forms
    W_i = tau u - 1/2 |tau|^2 (u^H v) v, where u = y - V x2 - W x1.
  - u^H v needs no extra pass: it equals y^H v - x2^H x1 - x1^H x2.
  - Unless column i is the panel's last, FIN also brings column k + 1 up to
    date with the panel so far, so the next SYMV can reflect it. That column
    goes into A and into the contiguous copy.
- tridiag.slang keeps only QL and ROT. Its REFLECT, SYMV and RANK2 are gone.

## Divide and conquer

- **Secular equation.** One thread per root summed the k poles one after
  another at every iteration, so it was latency-bound and had too few
  threads for a large GPU. Now 32 lanes share a root: they sum the terms in
  shared memory, and the root's first lane takes the step. A workgroup's two
  roots iterate until both have converged. At n = 4096 on the H100, this
  phase went from 20 ms to 1.1 ms.
- **The merge GEMM.** QM is diag(Q1, Q2) unless a deflation rotation mixed
  the two halves. PREP now flags that case. Otherwise QM U is two products
  of half the size, which halves the top merge's GEMM (14.5 to 7.8 ms).
- **Scaling.** The tridiagonal is first scaled so that its largest entry is
  1, as LAPACK's dstedc does, and the eigenvalues are scaled back afterwards.
  - Before this, the deflation tolerances were relative to the merged
    matrices' size, which counts the padding entries. Those sit near 1, so a
    matrix with entries around 1e-100 had its eigenvalues deflated as
    negligible. Eigenvalues came out up to 40% wrong whenever n needed
    padding (n = 70, for example).
  - This affected svd too: it uses the same divide and conquer.

## Results

**Accuracy** (UHD 620): eigenvalues, the residual |A V - V Lambda| / |A|
and the orthogonality are all about 1e-15 in float64 and complex128, and
about 1e-6 in float32. The test inputs were:
- n from 2 to 500, and batches;
- the panel boundaries (n = 65, 97);
- zero, diagonal, identity, rank-deficient and tridiagonal matrices;
- matrices scaled by 1e-200 and 1e200.

`test_eigh_blocked_scaled` covers the boundaries and the scaling.

**Speed**, `benchmarks/vs_cuda.py`, float64 (ms; vk ÷ CUDA):

| GPU | op | n | M26 / M28 | now | CUDA | now ÷ CUDA |
|---|---|---|---|---|---|---|
| H100 NVL | eigh | 1024 | 53.9 | 42.2 | 12 | 3.5 |
| H100 NVL | eigh | 4096 | 598 | 323 | 130 | 2.5 |
| L40S | eigh | 1024 | 72.1 | 50.3 | 25.9 | 1.9 |
| L40S | eigh | 4096 | 1351 | 692 | 446 | 1.6 |
| H100 NVL | svd | 1024 | 105 | 77.2 | 99.6 | 0.77 |
| H100 NVL | svd | 4096 | 1739 | 1524 | 2245 | 0.68 |
| L40S | svd | 1024 | 227 | 152 | 824 | 0.18 |
| L40S | svd | 4096 | 5120 | 4394 | 13109 | 0.34 |

At n = 1024 the reduction is bound by dispatch latency: two dependent
dispatches per column, about 2000 in all. Going below that needs a
two-stage reduction (to a band by GEMMs, then bulge chasing), which is how
MAGMA and cuSOLVER do it.
