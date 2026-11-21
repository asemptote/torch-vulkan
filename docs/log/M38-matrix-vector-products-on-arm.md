# Matrix-vector products with vector loads, and int8 and int4 weights (M38)

Issues #74 and #80. A language model's decoding step multiplies every
weight matrix by one column (or a few, for a batch of sequences or
speculative decoding): it reads each weight once, so it is bound by memory
bandwidth. On the Mali-G57 (Galaxy A13, a bus of about 13 GB/s) these
products ran far below the bus:

- float32 with one column went through `gemv.slang` (32 threads per row,
  4-byte loads): 7.6 GB/s. With two to eight columns they took the GEMM
  tiles, which leave most of a tile idle: 0.47-0.57 GB/s.
- float16 and bfloat16 had no matrix-vector path: they were widened to
  float32 (a copy of the whole weight matrix per call) first: 0.03-1.5 GB/s.
- `_weight_int8pack_mm` with up to 8 rows read its weights one byte per
  load (`qmm8.slang`'s matrix-vector path): 1.7 GB/s.
- `_weight_int4pack_mm` (the 4-bit layout of torchao and gpt-fast for
  non-CPU devices) ran on the CPU.

## What changed

- **`shaders/gemv4.slang`**: out[j][r] = alpha sum_k W[r, k] x[j][k] (+ beta
  cin[j][r]) for one to eight vectors, W's rows contiguous along k, float32
  sums rounded once. Variants by W's storage (float32, float16, bfloat16;
  int8 with a scale per row; 4-bit with a scale and a zero per row and
  group) and by vector count (1, 2, 4, 8: fewer are padded by reading the
  last vector again and not storing it).
  - Lanes along a row walk k in 16-byte chunks of W, consecutive lanes on
    consecutive chunks, so a warp's loads are contiguous; x is read in
    16-byte loads too (the host copies it, zero-padded, when it is not
    contiguous and aligned; it is small). Rows whose start is not 16-byte
    aligned (odd k, a slice) are read one element at a time, with the same
    sums.
  - Each thread keeps several rows (4 for one or two vectors, 2 for more),
    taking a chunk in steps of 4 or 8 values of k: x's values for a step are
    loaded once and used for every row. This matters on Mali, where every
    load, x's from cache too, goes through one load/store unit: the first
    version, which reloaded x per row and held a whole chunk of W for every
    row in registers, fell from 12 GB/s with two vectors to 1.4 with four.
    Rows are taken one at a time within a step, so registers hold the sums,
    a step of x and one row's step of W.
  - The lanes' partial sums are added in shared memory (Mali's subgroups are
    not used).
  - Integers become floats through the exponent of 2^23
    (`asfloat(0x4B000000 | u) - 2^23`, exact), not integer-to-float
    conversions. int8: sum q x, times the row's scale at the end. 4-bit: each
    32-value chunk lies in one group, so its sums are scale * sum (q - 8) x +
    zero * sum x, sum x shared by the thread's rows. Both differ from the
    CPU's order (it dequantises each weight first) within float32 rounding.
  - Lanes per row: 32 for one or two vectors of float or int8 weights, 8 for
    4-bit weights and for four or eight vectors (measured on the Mali; 16 is
    a warp there).
- **Where it runs.** `gemm()` hands it products with one to eight columns
  (W = A, as `W @ x`) or rows (W = B^T, as `x @ W.t()`), batches of one,
  whose matrix side has unit k stride; `_weight_int8pack_mm` with up to 8
  rows; `_weight_int4pack_mm` with up to 8 rows. `TORCH_VK_GEMV=vector` or
  `scalar` chooses; by default it is `vector` everywhere except for float32
  on NVIDIA and AMD GPUs, where `gemv.slang` already reads near the
  bandwidth (not measured here) and float16 products of two to eight
  vectors stay on the tensor cores.
- **`_convert_weight_to_int4pack` and `_weight_int4pack_mm`** on "vk", with
  the checks of the CUDA and MPS kernels. The packed tensor has the shape
  every device's meta kernel gives ([N / 8, K / (16 innerKTiles), 32,
  innerKTiles / 2], int32) but holds the input's bytes in order (rows of K /
  2 bytes, the even k in the high nibble): gemv4 reads them as they are.
  More than 8 rows dequantise W into float32 with the GPU's element-wise
  ops and take the float32 GEMM. PyTorch's `test__int4_mm` now passes on vk
  (its runner's "out of scope" entry is gone).

## Measured

4096 x 4096 weights, `x @ W.t()` (`benchmarks/gemv.py`), GB/s of weights
read (int4: with its scales and zeros). The CPU ran one thread
(Termux's OpenBLAS is not thread-safe; with the locking build and 8 threads
float32 reaches about 3.1 GB/s).

| Mali-G57 MC2 | 1 vector | 2 | 4 | 8 | before (1 vector; 2 to 8) | CPU, 1 vector |
|---|---|---|---|---|---|---|
| float32 | 12.9 | 12.6 | 11.6 | 5.0 | 7.6; 0.47 | 2.5 |
| float16 | 12.8 | 11.1 | 7.4 | 2.8 | 0.04; 0.03-0.05 (`W @ x`: 1.5; 0.25) | 0.15 |
| int8 | 7.8 | 5.2 | 1.7 | 0.57 | 1.7; 1.4, 1.0, 0.61 | 0.88 |
| int4 | 4.2 | 2.3 | 1.0 | 0.31 | (no kernel on vk) | 0.06 (`_for_cpu`) |

float32 and float16 with one or two vectors run at the bus's speed. The
issue's 0.57 GB/s (float32) and 0.26 (float16) are the GEMM tiles' and the
widening path's rates with two or more columns.

On the Intel HD 530 (dual-channel DDR4) float32 reads 19-24 GB/s for one to
eight vectors (before: 18 for one, 2.6-3.0 for two), float16 9-24 (before:
0.2-2.0), int8 3-19 (before: 3.9-5.8 for one or two), int4 2-11. On
lavapipe (1024 x 1024) every case is faster too (float32 one vector 20.7
GB/s against 8.2, float16 13.7 against 0.5).

## Not done

- Four and eight vectors of int8 or 4-bit weights stay below 2 GB/s on the
  Mali: every weight costs a decode and NC multiply-adds in float32, and
  the Mali-G57's ALUs run out first. Integer dot products (int8 x against
  int8 W, with activations quantised per row) or packed float16 arithmetic
  would raise it but change the results; both are left for #75's opt-in
  reduced-precision flag.
- 4-bit weights with one vector read 4 GB/s of the 13: the decode
  (shift, mask, exponent trick) per value limits it.
