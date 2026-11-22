# Depthwise convolution kernel (M48)

Part of a cleanup of ops built from many small kernel launches. M46 (#110)
ran depthwise convolutions (one input channel per group) as one
multiply-add of a strided view of the input per tap, for up to 32 taps:
9 launches for a 3 x 3 kernel, and more where the strided kernels loop on
the host over dims they cannot merge. With a channel multiplier (M outputs
per input channel) the views have five dims, so a 3 x 3 depthwise
convolution of [8, 64, 56, 56] into 128 channels took 4,106 launches.
Depthwise convolutions with more than 32 taps, and every depthwise weight
gradient, went through im2col columns and a batch of one-row products,
which the GEMM kernels handle badly.

## shaders/dwconv.slang

One kernel, two modes:

- **Forward**: one invocation per output. Its taps are summed in order
  with fused multiply-adds, then the bias is added. Taps outside the input
  are skipped, as oneDNN's depthwise convolution (the CPU's on x86) skips
  them: an infinite or NaN weight does not reach an output whose window
  meets it only in the padding (M46's padded input made those NaN). Negative
  padding, which the transposed convolution's phases use, crops. Any number
  of taps, strides, dilations and multipliers, 1 to 3 spatial dims (as D,
  H, W with size-1 dims for fewer).
- **Weight gradient**: one workgroup per (tap, output channel). Its 64
  lanes sum gy x over the batch and the output positions, then a tree adds
  the lanes' sums.

Types: float32, float64, int32 and int64 (wrapping, as the CPU's), and
float16 and bfloat16 storage summed in float32 and rounded once (the
forward no longer widens 16-bit inputs to float32 copies first). The input
gradient of a depthwise convolution is a depthwise transposed convolution,
which reaches the same forward kernel (by phases where strided, M46).

Where it runs: every depthwise convolution whose element offsets fit in
32 bits; past that, the grouped products. `TORCH_VK_DWCONV=0` forces the
grouped products (for comparison; `test_conv_depthwise_grouped_products`
runs some cases that way).

## Measured

`benchmarks/conv.py --filter depthwise` (float32, ms), forward and
`--backward` (forward and both gradients):

HD 530 (`benchmarks/launches.py`: [8, 64, 56, 56], float32, launches and ms;
the lab machine is shared, about 10% noise):

| | main (M46) | dwconv |
|---|---|---|
| k3, forward | 67 launches, 14.9 | 1, 5.3 |
| k3, forward and backward | 146, 86.6 | 5, 16.4 |
| k3 stride 2, forward and backward | 1,628, 33.6 | 10, 11.5 |
| k7 (49 taps; the products before), forward and backward | 30, 230 | 5, 71 |
| 64 -> 128 channels (multiplier 2), forward | 4,106, 27.4 | 1, 8.1 |
| 64 -> 128, forward and backward | 4,123, 223 | 9, 39 |

The weight gradient alone on the HD 530 (ms): k3 56.6 -> 7.3, k3 stride 2
15.7 -> 2.0, k7 194 -> 39, multiplier 2 177 -> 14.

Mali-G57 (ms; M46's multiply-adds per tap emulated from Python with the same
strided views, the products by `TORCH_VK_DWCONV=0`):

| | per tap | products | dwconv | weight gradient: columns -> dwconv |
|---|---|---|---|---|
| k3 64 ch 56 x 56 b8 | 39.3 | 169 | 11.6 | 90.4 -> 16.9 |
| k3 stride 2 | 14.1 | 47.3 | 3.2 | 29.1 -> 5.1 |
| k3 64 -> 128 | 95.8 | 235 | 22.4 | 650 -> 33.5 |
| k7 96 ch | 272 | 599 | 56.8 | 443 -> 129 |
| conv_transpose1d depthwise k3 s2, 640 ch, 121 frames (Kokoro) | | 10.6 | 2.1 | |

lavapipe (`benchmarks/conv.py --filter depthwise`, ms, main -> this change;
forward, then `--backward`):

| | forward | forward and backward |
|---|---|---|
| conv2d depthwise 64 k3 56 x 56 b8 | 118 -> 47 | 445 -> 139 |
| k3 stride 2 | 42 -> 12 | 207 -> 89 |
| 64 -> 128 k3 | 288 -> 94 | 2024 -> 1148 |
| 96 k7 | 2135 -> 181 | 5557 -> 711 |
| conv1d depthwise 256 k31, 1000 samples b4 | 226 -> 53 | 703 -> 213 |
| conv_transpose1d depthwise k3 s2 (Kokoro) | 4.7 -> 5.0 | 11.0 -> 9.9 |

With a multiplier, the input gradient (a transposed convolution with 2
input channels per group) still takes the grouped products, most of the
1148 ms on lavapipe.

L40S

## Tests

tests/python/test_nn.py: `test_conv_depthwise` (1-3 dims, multipliers,
strides, dilations, 31 and 35 taps, bias; forward and backward against the
CPU; the forward in one launch), `test_conv_depthwise_dtypes` (float16 and
bfloat16 within one rounding of the exact result, integers exactly,
float64), `test_conv_depthwise_layouts_and_nonfinite` (channels-last and
strided inputs; infinite and NaN weights at the padding, where the CPU's
convolution is oneDNN's), `test_conv_depthwise_grouped_products` (the
other route).

## Not done

- Each forward invocation computes one output and reloads its neighbours'
  inputs from cache; a 7 x 7 kernel runs at about 5 GMAC/s on the HD 530.
  A strip of outputs per invocation sharing the loaded rows would be
  faster, if depthwise convolutions with large kernels matter somewhere.
