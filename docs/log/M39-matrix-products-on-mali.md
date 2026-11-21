# Matrix products on Mali with subgroup shuffles (M39)

Issue #79. On the Mali-G57 MC2 (Galaxy A13 5G) `gemm4.slang` (M17: float4
loads, 2 x 8 outputs per thread, no shared memory) multiplied float32 at
about 31 GFLOP/s, a quarter of the 122 GFLOP/s the GPU's FMA units could do
(2 cores x 2 engines x 16 lanes x 2 flops x 955 MHz). Kokoro's vocoder spends
61% of its GPU time in convolutions, which are im2col plus this product
(#75's profile).

## What limits gemm4

Measured with Arm's Mali Offline Compiler (malioc 2026.5, part of the free
Arm Performance Studio, for the G57 r0p1 and driver r51p0) and with timing
sweeps of experimental kernels on the phone:

- **Loads, not arithmetic.** gemm4 uses all 64 registers (so half the
  threads fit) and spills 28 bytes. Each step of 4 values of k issues 10
  float4 loads (2 of A, 8 of B) for 64 multiply-adds per thread. Fitting the
  sweeps below, one 16-byte load costs the load/store unit about as long as
  12 multiply-adds take the FMA unit, and a subgroup shuffle about 1.3: gemm4
  is bound by its loads at 20-25% of peak.
- On Mali every load goes through the one load/store unit, from L1, from
  workgroup (shared) memory or from a broadcast address alike, so staging
  operands in shared memory does not reduce the count (M17's gemm.slang tiles
  reached 18-24 GFLOP/s; ncnn also leaves shared memory off on Mali).
- gemm4 is compiled in precise mode (every multiply and add separate). With
  fused multiply-adds allowed, malioc counts half the arithmetic, but on the
  phone the kernel ran 2x slower (17 GFLOP/s): the compiler's other choices
  cost more than the FMA saves. It stays precise.
- Larger register tiles fill the registers first. With A stored k-major
  (one float4 of A and one or two of B per k, as outer products), 4 x 8 per
  thread reached 51-55 GFLOP/s at 2048^3, 4 x 4 45, 8 x 4 36, 4 x 12 45, and
  8 x 8 spilled (13).

## gemm4s.slang: A shared across a subgroup

The G57 runs compute shaders in subgroups of 16 lanes with shuffles (without
subgroup size control; M38 did not use them). In `gemm4s.slang` a subgroup
computes an 8 x 64 block of C: lane l the 8 rows' columns 4l .. 4l + 3. Per
value of k a lane loads its 4 values of B (one 16-byte load) and does 32
multiply-adds; A's 8 values come by shuffles from the lanes that loaded
them, each lane loading one 16-byte piece of A per block of 8 values of k.
That is one load and 8 shuffles per 32 multiply-adds, against gemm4's 10
loads per 64. Workgroups are 2 subgroups (16 x 64 tiles); A and B row-major,
16-byte aligned (8 for 16-bit types), as gemm4's, else copied.

Sweeps at 2048^3 float32 on the A13 (GFLOP/s; each line one session, so
comparable within it; the phone throttles, by up to 15%, as it warms):

| kernel | 2048^3 | operands from cache* |
|---|---|---|
| gemm4 (2 x 8, loads of A per lane) | 33-34 | 37 |
| 8 x 4 from broadcast loads of A (no shuffles) | 34 | 57 |
| 8 x 4, A by shuffles, blocks of 8 k | 65-68 | 71-76 |
| the same, blocks of 4 / 16 / 32 k | 62 / 57-60 / 55 | 74 / - / - |
| 4 x 4 / 4 x 8 / 6 x 4 / 10 x 4 per lane | 58 / 62 / 66 / 60 | - / - / 72 / 73 |
| 12 x 4 per lane (spills) | 27-32 | 36 |
| A by quad broadcasts (4x cheaper in malioc, 4x more loads of A) | 52 | 68 |
| next block of A prefetched into registers | 67 | - |
| workgroups of 1 / 2 / 4 / 8 subgroups | 55 / 73 / 67 / 58 | - |
| shuffles in precise mode | 1 | - |

\* all rows of A and of B the same, so every load hits L1: the kernel's
ceiling, about 62% of peak. The rest is memory: the convolution shapes (A
streamed once, N = 128) lose most there.

16-bit storage: float16 and bfloat16 products had no Mali kernel (they ran
as float32 products of widened copies). gemm4s reads them with 8-byte loads
and widens to float32 in registers; sums are float32, rounded once to
nearest even, as before. bfloat16 spilled with blocks of 8 values of k (43
GFLOP/s at 2048^3), so it uses blocks of 4 (58-64).

`gemm4s.slang` is in build.rs's DEFAULT_FP_MODE list (fused multiply-adds),
and vkcore runs it where a pipeline can require 16 lanes or the device's
only subgroup size is 16 (`EXACT_SUBGROUP_KERNELS`; the HD 530 runs it with
required size 16, lavapipe cannot). `TORCH_VK_DISABLE=subgroups` now hides
subgroup operations too, which brings back gemm4 on the Mali.

When B is column-major (x @ W.t(), nn.Linear) and A is small, gemm4s
computes C^T = B^T A^T instead (C through a transposed view), copying x
rather than the weight: linear 95 x 768 -> 3072 float16 went from 8.7 to
about 24 GFLOP/s.

## Convolutions

`convolution.cpp` now computes a plain convolution with batch 1 or one group
as out[n, g] = W_g [Cout, Cin K] @ cols[n, g] [Cin K, L]: the im2col columns
are copied as rows along the output positions (for stride 1, runs of the
input), padded to a multiple of 4 (gemm4s reads B's rows in fours and
stores only the first L columns; `padded_cols` lets it read a narrowed view
of such a buffer), and the bias is the product's beta Cin (one rounding, as
y + bias). The result comes out as [N, Cout, L], without the [N L, Cout]
product's transposing copy and its separate bias pass. Same results; faster
on every GPU measured (other batches with groups keep the old layout).

Kokoro's convolutions on the A13 (float32, ms):

| | main | gemm4s | + columns as rows |
|---|---|---|---|
| 128 -> 128, k 11, 29,041 samples | 565 | 304 | 221 |
| 128 -> 128, k 11, dilation 3 | 562 | 293 | 222 |
| 128 -> 128, k 7 | 417 | 210 | 150 |
| 128 -> 128, k 3 | 198 | 131 | 79 |
| 256 -> 256, k 11, 4,840 samples | 292 | 157 | 124 |
| 256 -> 256, k 7 | 189 | 111 | 83 |
| 256 -> 256, k 3 | 90 | 63 | 42 |
| 1024 -> 1024, k 3, 242 frames | 77 | 51 | 36 |

Of the k = 11 convolution's 221 ms, the product is about 150 ms and the
column copy 72 ms (164 MB written by the generic strided copy, about 2.3
GB/s): a direct convolution reading the input in place (#75 item 1) would
remove most of it.

The HD 530 (gemmsg, unchanged) gains from the new layout too: 128 -> 128
k 11 192 -> 157 ms, k 3 85 -> 51 ms, 256 -> 256 k 3 34 -> 28 ms; batched
conv2d (8 x 64 x 56 x 56, 3 x 3) 56 -> 52 ms. lavapipe: 459 -> 418 ms, 189
-> 139 ms.

## Measured

Matrix products on the A13 (GFLOP/s, synchronised per call; "before" is
gemm4, as on main, which `TORCH_VK_DISABLE=subgroups` restores; float16 and
bfloat16 were float32 products of widened copies). The phone throttles as it
warms: repeated runs vary by up to 15%.

| | float32 before -> after | float16 | bfloat16 |
|---|---|---|---|
| 2048^3 | 29 -> 73 | 28 -> 75 | 28 -> 64 |
| 2048^3, a @ b.T | 28 -> 68 | 26 -> 69 | 26 -> 60 |
| 1024^3 | 28 -> 66 | 25 -> 69 | 25 -> 60 |
| 29,041 x 1408 x 128 (a k = 11 convolution's) | 25 -> 53 | 23 -> 66 | 23 -> 58 |
| 4840 x 2816 x 256 | 26 -> 61 | 24 -> 72 | 25 -> 60 |
| linear, 95 x 768 -> 3072 | 16 -> 28 | 9 -> 35 | 9 -> 32 |

float16 is faster than float32 at the convolution shapes: its 8-byte loads
cost the load/store unit less.

Kokoro-82M end to end on the A13 (#75's setup: one sentence, 6.05 s of
audio, best of 3 after a warm-up):

| | main | gemm4s | + convolution layout |
|---|---|---|---|
| float32 | 22.0 s (RTF 3.64) | 14.7 s | **11.6 s** (RTF 1.92) |
| float16 | 26.7 s | 19.1 s | **15.7 s** |

By op type (float32, synchronised per op, net of the per-call cost):
Conv1d 16.4 -> 7.5 s, ConvTranspose1d 2.2 -> 0.94 s, Linear 1.1 -> 0.55 s.
The LSTMs (7.1 s net when every op is synchronised, about 1.5 s in a real
run) are now the largest item. PyTorch's CPU takes 27.8 s on one thread and
21.5 s on eight; ONNX Runtime's WebGPU float16 model 12.1 s. The float32
audio differs from main's vk run by 1.7% (rms of the difference over the
signal's; it carries noise from the vocoder's random source, the same on
both, through float32 sums taken in another order); main's vk run differs
from the CPU's by 10%, the random noise being different there.

Other GPUs: the default kernels are unchanged off Arm (gemmsg on the HD 530,
gemmsg or gemm.slang on lavapipe), apart from the convolution layout above.
Through `TORCH_VK_GEMM=float4` the HD 530 runs gemm4s with 16 lanes:
float32 2048^3 81 GFLOP/s (gemmsg 95, gemm4 41), float16 104 (the default,
float32 products of widened copies, 89). lavapipe cannot run it (8-lane
subgroups, no size control) and keeps gemm4 there.

## Tests

- tests/python on the HD 530 (1894 passed) and lavapipe (1893 passed);
  test_matmul.py and test_nn.py on the HD 530 with `TORCH_VK_GEMM=float4`,
  so through gemm4s; the strict OpInfo sweep on lavapipe (689 ops, all on
  the GPU and matching) and on the HD 530 with `TORCH_VK_GEMM=float4` for
  float32, float16 and bfloat16 (668 of 669; `_batch_norm_with_update`'s
  float16 dtype mismatch is the same on main); clippy and vkcore's tests.
- On the A13: tests/python, 1402 passed and 489 skipped (no float64), apart
  from `test_split_k_gemm` (below), which failed on main there too.
- New: gemm4s at edges, K tails, transposed and strided operands, batches,
  alpha, beta and bias; float16 and bfloat16 within one rounding plus the
  float32 sums' error, and float16 overflow to inf as on the CPU; nn.Linear
  shapes (the transposed orientation); convolutions with one sample, groups
  and lengths and K not multiples of 4.
- `test_split_k_gemm` now skips where no kernel splits k (Arm GPUs): its
  tolerance, 2^-20 of the products' magnitudes, is for split sums, and
  unsplit float32 sums over k = 2048 exceed it there (gemm4 by 1.31x on the
  A13, as on main, where it failed; gemm4s by 1.09x).

## For #75 (not done here: they change results)

- **float16 arithmetic.** Valhall's FMA unit does two float16 multiply-adds
  per lane (half2) per float32 one. With half2 accumulators a lane holds
  twice the outputs in the same registers (8 x 8 instead of 8 x 4), a
  16-byte load carries 8 values of B and a shuffle 2 values of A, so loads
  and shuffles per multiply-add halve too: about 2x on float16 products and
  convolutions, at the cost of float16 sums (overflow past 65504, error
  growing with k: 1408 values in Kokoro's largest).
- **The convolution's column copy** is now about a third of a k = 11
  convolution on the Mali (72 of 221 ms). Reading the input in place (B's
  row for (c, j) is the input row c shifted by j d, for stride 1) would
  need unaligned 16-byte loads in gemm4s; that is #75's direct convolution.
- The LSTMs (#75 item 3, #78) are now the largest share of Kokoro's time.
