# Direct convolution and float16 arithmetic (M47)

Part of issue #75 (items 1, 2 and 4: a direct convolution kernel, vectorised
float16 kernels, an opt-in flag for float16 arithmetic). After M39 and M46,
convolutions were still most of Kokoro's vocoder time on the Mali-G57: an
im2col copy of the input (K times its size: 164 MB for a k = 11
convolution over 29,041 samples) and a product that streamed those columns
once per 16 rows of the weight. float16 convolutions were widened to
float32 copies first and computed in float32.

## shaders/conv4s.slang: convolution without columns

A stride-1 convolution is a product whose B rows are the input shifted by
each tap: B[(c, t), o] = x[c, o + off(t)]. `conv4s.slang` computes it on
gemm4s.slang's scheme (M39: a 16-lane subgroup computes R rows x 4 NC
columns per lane, A's values shared by shuffles) but reads B from the
zero-padded input in place:

- Spatial dims are flattened onto the padded grid: output (od, oh, ow) is
  the virtual column v = (od Hp + oh) Wp + ow, and tap (kd, kh, kw) reads
  x at v + kd dd Hp Wp + kh dh Wp + kw dw, so every tap is one shift for
  1-, 2- and 3-D alike. Columns with ow >= Wo (wrapping into the next row)
  are computed and not stored.
- The host copies the input once into a buffer padded as the kernel needs:
  rows of Wp rounded up to 4 values (2-D, 3-D), each channel's plane a
  multiple of 4, slack at the end for columns and taps read past the last
  valid one. The weight is copied into rows of the KW inner taps padded to
  a multiple of 4 (A [G, Cog, Cg KO, KW4]).
- The inner taps are taken 4 at a time. Per group, lane i < R loads row
  i's 4 weights (one piece of A; the other lanes take them by shuffles),
  and the 4 taps read the lane's columns at 0, dw, 2 dw, 3 dw: 4-value
  windows of the input pieces around them, with shifts fixed at compile
  time because the dilation is a variant of the kernel (axis D: 1, 2, 3,
  5; others use im2col). The next group starts dw pieces on, so the pieces
  the two share stay in registers: for dw = 1 a group of 4 taps costs one
  load of the input and one of A, where the product of columns loads 4 rows
  of B.
- float32, float16 and bfloat16 storage; sums are float32 with fused
  multiply-adds, the bias added, rounded once, as gemm4s's. 16-bit input
  pieces are kept as loaded and widened per tap (widening each piece as it
  was loaded ran 2x slower on the HD 530). float16 and bfloat16
  convolutions no longer go through float32 copies where conv4s runs
  (elsewhere `convolution_overrideable` widens them itself, as
  half_via_float did).

The first version unrolled up to 16 taps with the tap count and dilation as
specialisation constants and loaded each tap's window separately; on the
Mali-G57 it ran slower than im2col (k = 11: 351 ms against 216): the loads
were not shared between taps, and every channel's loop divided to find its
outer tap. Taps in groups of 4 with the dilation compiled in, and pointers
stepped per channel and outer tap, fixed both.

Where it runs: on GPUs with 16-lane subgroups and shuffles, for stride-1
convolutions with 3 or more inner taps (fewer, padded to 4, waste half the
work: M46's 2-tap transposed-convolution phases ran 1.5x slower), at least
2048 output positions (1024 channels over 242 frames ran faster as im2col
on both GPUs), Wo >= 8 in 2-D and 3-D, and not depthwise (M46's
multiply-adds do those):

- Intel (HD 530): every dtype; 8 rows x 4 columns per lane.
- Mali (Valhall): float16 and bfloat16 only; 4 rows x 8 columns per lane.
  float32 stays with im2col and gemm4s: with float32 windows the
  accumulators of 8 rows spill there, and at 4 or 6 rows it was at best
  level with gemm4s (k = 11: 206-229 ms against 216).

`TORCH_VK_CONV=im2col` restores the columns and product, `=direct` uses
conv4s wherever it runs (comparison and testing).

## float16 arithmetic (opt-in)

`torch_vulkan.set_fp16_arithmetic(True)` or `TORCH_VK_FP16_ARITHMETIC=1`
(off by default; docs/api.md) lets float16 convolutions and matrix products
multiply and add in float16: variants `conv4s_f16h` and `gemm4s_f16h`, with
half4 accumulators (packed half2 multiply-adds on Mali and Intel). gemm4s's
lanes then hold 8 columns in the registers 4 float32 columns took, and in
both a shuffle of a 32-bit word carries two values of A. B is read as half4
with no widening. alpha, beta, Cin and the bias are applied in float32 and
the result rounded once more. With the flag on, float16 products use gemm4s
wherever it runs (on Intel too, where float32 products keep gemmsg). The
flag is read per call (a C++ atomic).

Accuracy, against float64 references (HD 530): a 128 -> 128, k = 11
convolution (1408 terms per output) has an rms error of 0.4% of the
result's rms with float16 sums (max 3.6%), against a max of 0.14% with
float32 sums (one float16 rounding of the result); a 300 x 1408 x 1000
product, max 6.9%. Partial sums overflow past 65504.
tests/python/test_fp16_arithmetic.py bounds the error by (k + 2) eps times
the sum of the products' magnitudes (recursive summation's bound), checks
that the default comes back bit for bit, and that the float16 sums did run
where the kernels can.

## Measured

`benchmarks/conv.py`, ms (k = kernel taps, d = dilation, L = samples).

HD 530, conv4s against im2col (the lab machine's 8 threads are shared:
about 10% noise):

| | float32 im2col -> direct | float16 im2col -> direct | float16 arithmetic |
|---|---|---|---|
| conv1d 128 -> 128, k 11, L 29,041 | 153 -> 58 | 159 -> 90 | 29 |
| conv1d 128 -> 128, k 11, d 3 | 153 -> 90 | 159 -> 181 | 70 |
| conv1d 128 -> 128, k 7 | 100 -> 46 | 106 -> 68 | 23 |
| conv1d 128 -> 128, k 3 | 46 -> 33 | 53 -> 46 | 17 |
| conv1d 256 -> 256, k 11, L 4,840 | 90 -> 38 | 93 -> 60 | 19 |
| conv1d 256 -> 256, k 3 | 26 -> 20 | 28 -> 29 | 9.3 |
| conv2d 64 -> 64, k 3, 56 x 56, batch 8 | 64 -> 17 | 67 -> 25 | 9.7 |

float16 matrix products on the HD 530 (GFLOP/s): 1024^3 75 -> 117, 2048^3
84 -> 133 with float16 arithmetic (gemm4s_f16h, against float32 sums).

MALI_RESULTS

## Tests

- tests/python/test_nn.py `test_direct_conv` (through
  `TORCH_VK_CONV=direct`, in a subprocess): 1-3 dims, inner taps 1-16,
  dilations 1-5, groups, batches, bias, cropped inputs (M46's phases),
  float32, float16 and bfloat16, against float64 references with the
  float32 sums' bound; `test_fp16_arithmetic.py`.
- HD 530 (conv4s the default there): tests/python (1960 passed), PyTorch's
  `nn/test_convolution.py`, `test_nn.py -k onv` and `test_ops.py -k conv`
  as in M46, the strict OpInfo sweep's convolutions; the L40S (where conv4s
  does not run): tests/python; lavapipe: CI.

## Not done

- Item 3 of #75, a fused LSTM cell (and keeping the STFT in float32), and
  graph capture (#78): Kokoro's LSTMs are now the largest share of its time
  after the vocoder.
- float32 convolutions on Mali: a conv4s that keeps float32 windows and 8
  rows without spilling (shorter windows, or A from shared memory) is left
  to try.
- Convolutions with a stride (other than transposed ones) still use
  im2col.
- float16 arithmetic in the matrix-vector kernels (gemv4, bound by
  bandwidth) and in normalisations.
