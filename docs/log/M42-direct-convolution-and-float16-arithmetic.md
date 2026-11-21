# Direct convolution and float16 arithmetic (M42)

Part of issue #75 (items 1, 2 and 4: a direct convolution kernel, vectorised
float16 kernels, an opt-in flag for float16 arithmetic). After M39 and M41,
convolutions were still most of Kokoro's vocoder time on the Mali-G57: an
im2col copy of the input (K times its size: 164 MB for a k = 11
convolution over 29,041 samples) and a product that streamed those columns
once per 16 rows of the weight. float16 convolutions were widened to
float32 copies first and computed in float32.

## shaders/conv4s.slang: convolution without columns

A stride-1 convolution is a product whose B rows are the input shifted by
each tap: B[(c, t), o] = x[c, o + off(t)]. `conv4s.slang` computes it on
gemm4s.slang's scheme (M39: a 16-lane subgroup computes R rows x 64
columns, A's values shared by shuffles) but reads B from the zero-padded
input in place:

- Spatial dims are flattened onto the padded grid: output (od, oh, ow) is
  the virtual column v = (od Hp + oh) Wp + ow, and tap (kd, kh, kw) reads
  x at v + kd dd Hp Wp + kh dh Wp + kw dw, so every tap is one shift for
  1-, 2- and 3-D alike. Columns with ow >= Wo (wrapping into the next row)
  are computed and not stored; 2-D and 3-D convolutions with Wo < 8, where
  they would be most of the work, keep im2col.
- The host copies the input once into a buffer padded as the kernel needs:
  rows of Wp rounded up to 4 values (2-D, 3-D), each channel's plane a
  multiple of 4, slack at the end for columns and taps past the last
  valid one. The weight is copied into rows of the KW inner taps padded to
  a multiple of 4 (A [G, Cog, Cg KO, KW4]).
- Per channel and outer tap (kd, kh), a lane loads its pieces of A once
  and the 4-value windows of its columns for the KW inner taps from the
  16-byte loads of its row: KW and dw are specialisation constants, so the
  window shifts are known when the driver compiles the pipeline and a load
  serves every tap it overlaps (k = 11, d = 1: 4 loads for 11 taps,
  against 11 rows of columns). The tap loops run to 16 (MAXKW) with
  `if (t < KW)`, folded away once specialised.
- float32, float16 and bfloat16 storage; sums are float32 with fused
  multiply-adds and the bias, rounded once, as gemm4s's. float16 and
  bfloat16 convolutions no longer go through float32 copies where it runs
  (elsewhere `convolution_overrideable` now widens them itself, as
  half_via_float did).
- It runs wherever the GPU runs 16-lane subgroups with shuffles (Mali's
  Valhall GPUs, Intel's), for stride-1 convolutions with at most 16 inner
  taps that are not depthwise (M41's multiply-adds do those):
  convolutions with a stride keep im2col, and transposed convolutions
  reach it through M41's phases (stride 1). `TORCH_VK_CONV=im2col` restores
  the columns and product for comparison.

## float16 arithmetic (opt-in)

`torch_vulkan.set_fp16_arithmetic(True)` or `TORCH_VK_FP16_ARITHMETIC=1`
(off by default; docs/api.md) lets float16 convolutions and matrix products
multiply and add in float16: variants `conv4s_f16h` and `gemm4s_f16h`,
where half4 accumulators (packed half2 multiply-adds on Mali and Intel) let
a lane hold 8 columns in the registers 4 float32 columns took, and one
shuffle of a 32-bit word carries two values of A (two taps of a row,
two values of k). B is read as half4 with no widening. alpha, beta, Cin and
the bias are applied in float32 and the result rounded once more. With the
flag on, float16 products use gemm4s wherever it runs (on Intel too, where
float32 products keep gemmsg). The flag is read per call, a C++ atomic.

Accuracy, against float64 references (HD 530): a 128 -> 128, k = 11
convolution (1408 terms per output) has an rms error of 0.4% of the
result's rms with float16 sums (max 3.6%), against a max of 0.14% with
float32 sums (one float16 rounding of the result); a 300 x 1408 x 1000 product, max 6.9%. Partial sums overflow
past 65504. tests/python/test_fp16_arithmetic.py bounds the error by
(k + 2) eps times the sum of the products' magnitudes (recursive summation's
bound), checks the default is restored bit for bit, and that the float16
sums did run where the kernels can.

## Measured

`benchmarks/conv.py`, ms. HD 530 (8 threads of the lab machine shared with
other work: about 10% noise):

| | main (im2col + gemmsg) | direct, float32 | float16 storage | float16 arithmetic |
|---|---|---|---|---|
| conv1d 128 -> 128, k 11, 29,041 samples | 152 | 84 | 79 | 74 |
| conv1d 128 -> 128, k 11, d 3 | 153 | 104 | 90 | 78 |
| conv1d 128 -> 128, k 7 | 100 | 68 | 51 | 51 |
| conv1d 128 -> 128, k 3 | 46 | 23 | 25 | 18 |
| conv1d 256 -> 256, k 11, 4,840 samples | 90 | 55 | 52 | 46 |
| conv1d 256 -> 256, k 3 | 26 | 14 | 15 | 9.2 |
| conv1d 1024 -> 1024, k 3, 242 frames | 19 | 17 | 18 | 13 |
| conv_transpose1d 512 -> 256, k 20, s 10 (M41's phases) | 333 | 34 | 33 | 27 |
| conv_transpose1d 256 -> 128, k 12, s 6 | 347 | 38 | 37 | 27 |
| conv2d 64 -> 64, k 3, 56 x 56, batch 8 | 64 | 14 | 15 | 10 |

float16 matrix products on the HD 530 (GFLOP/s): 1024^3 75 -> 117, 2048^3
84 -> 133 with float16 arithmetic (gemm4s_f16h, against float32 sums).

Mali-G57 (Galaxy A13): to follow (tile sweep, Kokoro end to end).

## Tests

- tests/python/test_nn.py `test_direct_conv` (through `TORCH_VK_CONV`, in a
  subprocess): 1-3 dims, every inner tap count to 16, dilations 1-5,
  groups, batches, bias, K = 1, asymmetric and cropped inputs (M41's
  phases), float32, float16 and bfloat16, against float64 references with
  the float32 sums' bound; `test_fp16_arithmetic.py`.
- The HD 530 now runs every stride-1 convolution through conv4s by
  default: tests/python (1920 passed), PyTorch's `nn/test_convolution.py`,
  `test_nn.py -k onv` and `test_ops.py -k conv` as M41, the strict OpInfo
  sweep's convolutions; lavapipe (which cannot run conv4s) tests/python
  test_nn, test_matmul, test_fp16_arithmetic.

## Not done

- Item 3 of #75, a fused LSTM cell (and keeping the STFT in float32), and
  graph capture (#78): Kokoro's LSTMs are now the largest share of its time
  after the vocoder.
- Convolutions with a stride (other than transposed ones) still use
  im2col; strided convolutions are mostly the downsampling layers of image
  models.
- float16 arithmetic in the matrix-vector kernels (gemv4, bound by
  bandwidth) and in normalisations.
