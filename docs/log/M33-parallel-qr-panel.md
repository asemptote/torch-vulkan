# QR panels over many workgroups (M33)

`torch.linalg.qr` of a tall-skinny matrix ran at 20-40x CUDA's time (#62).
The chaotic-RNN benchmark (`benchmarks/rnn.py`) hit it first: its Lyapunov
spectrum re-orthonormalises 64 tangent vectors with a 1000 x 64 QR every 10
steps, and those QRs took 525 of the run's 580 ms on the L40S.

The QR was blocked Householder already (geqrf: 64-column panels, then block
reflectors and GEMMs for the rest). The panels went to factor.slang's
QR_PANEL, though, which factored a panel column by column on one workgroup
per matrix. For a 1000 x 64 matrix, that panel is the whole matrix. This was
LU's problem before M30.

## The panel, two dispatches a column (shaders/geqrf.slang)

Rows are split in blocks of 256, one block per workgroup.

- **DOT.** Every workgroup forms column j's reflector from the whole column,
  as zlarfg (QR_PANEL's arithmetic: norms scaled by the largest |part|, a
  column whose largest part is subnormal counting as zero). Each workgroup
  does the same arithmetic, so all of them get the same v and tau. Each
  workgroup then leaves the partial sums of v^H a[:, c] over its rows, for
  the panel's later columns c. A thread per (column, lane) reads the rows
  coalesced. Workgroup 0 records beta, the scale and tau.
- **UPD.** Every workgroup adds up those partial sums in the same order,
  then updates its rows of the later columns, a[i, c] -= v_i conj(tau)
  (v^H a[:, c]). Last, it stores v (x scaled, with beta on the diagonal) in
  column j.
- **No races.** A workgroup writes only its own rows. Column j is rewritten
  only in UPD, after every workgroup has read it in DOT.

The block reflectors and the trailing update are unchanged, and
factor.slang's QR_PANEL is gone.

## Results

**Accuracy** (HD 530 under Mesa, against the CPU; all four dtypes; tall,
wide, square, batched and rank-deficient matrices; reduced and complete
modes):
- the reconstruction |QR - A| / |A| is 1e-6 in single precision and
  1e-15 in double;
- the orthogonality |Q^H Q - I| is 4-6e-7 and 7-9e-16;
- R is exactly upper triangular, and |R| matches the CPU's;
- the linalg tests pass with the validation layer.

**Speed** (ms; vk ÷ CUDA, PyTorch 2.14.1 + CUDA 13.0):

| GPU | qr | before | now | CUDA | now ÷ CUDA |
|---|---|---|---|---|---|
| L40S | 1000 x 64 float32 | 10.6 | 1.34 | 0.26 | 5.2 |
| L40S | 1000 x 64 float64 | 13.4 | 2.85 | 0.68 | 4.2 |
| L40S | 5000 x 64 float32 | | 2.35 | 0.36 | 6.5 |
| L40S | 2048 x 2048 float64 | | 123 | 49 | 2.5 |
| H100 NVL | 1000 x 64 float32 | | 1.88 | 0.36 | 5.2 |
| H100 NVL | 1000 x 64 float64 | | 1.97 | 0.39 | 5.1 |
| H100 NVL | 2048 x 2048 float32 | | 67 | 12 | 5.6 |

The RNN's Lyapunov run (k = 64, N = 1000, 500 steps) on the H100 went from
1025 to 169 ms in float32 and from 1056 to 178 ms in float64. That is 3.3x
and 3.5x CUDA, from 21x. Its leading exponent is unchanged (0.043073).

The rest is mostly dispatch latency: two dependent dispatches per column,
128 for a 64-column panel. Fusing them is possible. The next column's
reflector would then come, redundantly, from the updated column in each
workgroup, as latrd does it (M29).
