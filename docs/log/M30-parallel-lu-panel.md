# LU panels over many workgroups (M30)

linalg.solve took 8x CUDA's time at n = 1024 and 19x at n = 4096 on the H100
NVL (issue #34). The LU was already blocked as LAPACK's getrf: GEMMs did the
trailing updates, and these cost little (a whole 4096 x 4096 float64 GEMM
takes 15 ms). The time went into the panels. factor.slang's LU_PANEL
factored each panel of 64 columns on one workgroup per matrix, a column
after another, so one SM did all the panel work: 338 of the 353 ms at
n = 4096.

## The panel, a column per dispatch (shaders/getrf.slang)

- **Workgroups.** Each workgroup takes 32 rows of the panel, with 8 lanes
  along each row.
- **Pivot search.** Each step's pivot search is a reduction over every
  row, so it needs one dispatch per column. Each workgroup leaves its best
  candidate (|a| as izamax measures it, position, row). The next dispatch
  starts by reducing all of them, in the same order in every workgroup.
- **Rows stay in place.** Swapping rows would be a race: every workgroup
  reads the pivot row while another would be moving it. So a row's position
  (where LAPACK's interchanges would have put it) is kept in an array
  instead, and so are the workgroups' candidates. Both are double-buffered
  by the step's parity, because a step reads one set and writes the next.
- **The pivot.** At step j the pivot is the largest entry among the rows at
  positions j and beyond. A tie goes to the lowest position, as LAPACK picks
  it. The pivot moves to position j, and the row there moves to the pivot's
  old position. The pivot row is only read during its step, so no workgroup
  waits on another.
- **SWAP.** After the panel, SWAP applies its interchanges to whole rows,
  a thread per column. That leaves the matrix as getf2 leaves it, with the
  same pivots, info and factors.
- **Zero pivots.** LAPACK leaves a zero pivot's column unscaled; so does
  STEP, and info records the first one. Without pivoting, the pivots are
  j + 1, as before.

Then, as before, U12 = L11^-1 A12 and A22 -= L21 U12 by GEMM.

## Dispatch latency

The first version took 77 ms at n = 4096. The host was not the bottleneck:
it waited on the stream. A trivial dispatch costs about 4 us of GPU time on
the H100, but STEP took about 15 us, because its critical path was long:
- load the candidates;
- a 256-wide tree reduction with device-scope barriers;
- load the pivot;
- a barrier to share the multiplier;
- a second 256-wide tree.

Shortening that path brought n = 4096 to 44 ms:
- each thread loads its part of the row before the reduction, since it does
  not depend on the pivot;
- the trees reduce only the live entries: 32 rows, and the workgroups'
  candidates rounded up to a power of two;
- the barriers are workgroup barriers on shared memory, because no thread
  reads another's global writes within a dispatch;
- each lane divides by the pivot itself, so sharing the multiplier needs no
  barrier.

## Results

**Accuracy.** On the UHD 620, the HD 530 (Mesa ANV) and lavapipe, the
pivots, info and factors match the CPU's LAPACK. P L U reconstructs A to
about 1e-14 in float64 and complex128. `test_lu_panel_pivots` covers:
- 300 x 300, batched 130 x 130, tall and wide shapes;
- float32, float64, complex64 and complex128;
- random matrices, a zero column, zero matrices, small integers (ties),
  and no pivoting.

For rank-deficient matrices, the pivots after the numerical rank can differ
from the CPU's. There the remaining columns are rounding noise, and which
entry is largest depends on the order of the arithmetic.

**Speed** on the H100 NVL, float64, with a 64-column right-hand side:

| n | solve before | solve now | lu_factor now | lu_solve | CUDA solve | now ÷ CUDA |
|---|---|---|---|---|---|---|
| 1024 | 27.9 ms | 12.0 ms | 8.6 ms | 3.8 ms | 3.43 ms | 3.5 |
| 4096 | 353 ms | 60.0 ms | 44.4 ms | 16.4 ms | 18.1 ms | 3.3 |

`benchmarks/vs_cuda.py`, linalg.solve float64 (ms; vk ÷ CUDA):

| GPU | n | before | now | CUDA | now ÷ CUDA |
|---|---|---|---|---|---|
| H100 NVL | 1024 | 27.9 | 12.0 | 3.4 | 3.5 |
| H100 NVL | 4096 | 353 | 60.0 | 18.1 | 3.3 |
| L40S | 1024 | 27.1 | 14.4 | 6.73 | 2.1 |
| L40S | 4096 | 329 | 94.5 | 66.3 | 1.4 |

What remains is mostly dispatch latency: a STEP per column, plus six
dispatches per panel (START, SWAP, the triangular inverse, two products and
a copy). The triangular solves (lu_solve) are blocked the same
way, three dispatches per 64 rows, and they are the next step if solve
should go faster.

## Also

svd of an empty batch with 32 or more columns failed: the bidiagonal path
(M28) took a full amax() of an empty tensor. PyTorch's own test helpers
build such batches. (Jacobi, the path below 32 columns, dispatched an empty
grid for them, which the validation layer flags.) An empty batch now returns
its empty results without dispatching anything.
