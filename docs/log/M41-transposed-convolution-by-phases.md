# Transposed convolution by phases (M41)

Issue #77. A transposed convolution (`conv_transpose1d/2d/3d`, and the input
gradient of every strided convolution) ran as an ordinary convolution of the
input with stride - 1 zeros inserted between its elements. That spends
almost all of its multiply-adds on the zeros: Kokoro's vocoder upsamples
with strides 10 and 6, so 90% and 83% of its two upsampling convolutions'
work was on zeros (#75's profile: 2.2 s of 21.8 s on the Mali-G57 then,
0.94 s of 11.6 s after M39).

## Sub-pixel convolution

Along a dim with stride s > 1, input i and tap k reach the uncropped output
position o' = o + p = i s + k d (padding p, dilation d). Write o' = q s + r
with 0 <= r < s: the outputs of phase r take the taps with k d = r + t s, from
input i = q - t. So phase r is an ordinary stride-1 convolution with
T = floor(d (K - 1) / s) + 1 taps (k's that land on t, zeros where none
does), and every phase reads the same input positions q - t. The s phases
(s1 s2 ... in several dims) are therefore one convolution with s Co output
channels:

- the weight [Ci, Co/G, K] is dilated (zeros between taps when d > 1),
  padded to T s taps, viewed as [T, s], reversed along t and rearranged to
  [G Co/G s, Ci/G, T] (a few small copies per call);
- the input is padded by T - 1 - q0 before (cropped where negative: large
  padding) and q0 + Q - S after, where [q0, q0 + Q) is the range of q whose
  phases cover the output o' in [p, p + L_out);
- the result [N, Co s, Q] is interleaved into [N, Co, Q s] (one copy) and
  cropped to the output (a second copy where the crop is not empty).

The work is T per output instead of K: s times less for d = 1 and K a
multiple of s (Kokoro: K = 20, s = 10 and K = 12, s = 6, T = 2), about K / T
otherwise. Where the phases would not save work (no stride, or a dilation
large enough that T >= K: d >= s) the zero-insertion path stays. Dims with
stride 1 stay ordinary (reversed kernel, dilation d) inside the phase
convolution, so mixed strides work. Results are the same sums in another
order: float32 rounding differences only.

Two things on the way:

- `aten::flip` of the weight cost 105 ms for a 256 x 128 x 4 x 4 weight on
  the HD 530: it gathers through a table of int64 offsets, launched per
  outer index for these many-dim views. Convolution weights are now
  reversed by one copy per tap (`reversed()` in convolution.cpp; a few
  tens of microseconds). The zero-insertion path used flip too.
- Depthwise convolutions (one input channel per group) ran as a batch of
  one-row products, one per group, which the GEMM kernels handle badly
  (640 products of [2 x 2] x [2 x 122] took 1.9 ms on the HD 530). They now
  run as one multiply-add per tap of the input's strided view into the
  result (at most 32 taps), which also serves the phases of Kokoro's
  depthwise stride-2 upsampling.
- The weight gradient of a 1 x 1 convolution failed (`view` of the
  columns' non-contiguous reshape); PyTorch's `test_Conv2d_size_1_kernel`
  and `test_ConvTranspose{2,3}d_size_1_kernel` now pass.

## Measured

`benchmarks/conv.py` (float32, ms; the 2-D cases are an image decoder's
stride-2 upsampling). "Before" is main on the HD 530, and elsewhere the
zero-insertion path of this change's build (its weights reversed without
`flip`, depthwise products by multiply-adds), so main was slower still there;
on the L40S, the zero-insertion path's ops run from Python (zeros inserted,
then the ordinary convolution with the flipped weight; on the HD 530 this
gives main's times within 10%):

| | Mali-G57 before -> after | HD 530 | lavapipe | L40S |
|---|---|---|---|---|
| conv_transpose1d 512 -> 256, k 20, s 10, 484 frames | 509 -> 100 | 333 -> 38 | 1506 -> 214 | 1.79 -> 0.24 |
| conv_transpose1d 256 -> 128, k 12, s 6, 4840 frames | 522 -> 91 | 347 -> 57 | 1791 -> 432 | 2.31 -> 0.33 |
| conv_transpose1d depthwise 640, k 3, s 2, 121 frames | 1.9 -> 1.9 | 1.3 -> 0.9 | 7.7 -> 10.8 | - -> 0.06 |
| conv_transpose2d 256 -> 128, k 4, s 2, 64 x 64 | 474 -> 104 | 322 -> 71 | 1304 -> 387 | 1.64 -> 0.37 |
| conv_transpose2d 128 -> 64, k 3, s 2, batch 4, 32 x 32 | 133 -> 43 | 85 -> 26 | 489 -> 125 | 0.50 -> 0.16 |

float16 runs at about the same speed (widened to float32, as before).
lavapipe's depthwise case is slower: its per-dispatch cost outweighs the
small work there.

Kokoro-82M end to end on the A13 (#75's setup, best of 3): float32 11.6 s
(M39) -> 10.7 s (RTF 1.77).

## Tests

- tests/python: transposed convolutions by phases in 1-3 dims with K
  multiples of the stride and not, padding past T - 1 (cropped input),
  dilation coprime to the stride, mixed strided and stride-1 dims, groups,
  depthwise, output_padding, with and without bias, forward and backward;
  float16, bfloat16 and int32; depthwise convolutions in 1-D and 3-D.
- PyTorch's `nn/test_convolution.py` on the HD 530: no failures but the
  `test_conv_backend_*` ones (which check that the CPU's backend enum was
  chosen, as on main) and `test_slow_conv_dilated{2,3}d_unbatched` (direct
  calls to `slow_conv_dilated2d`, which this change does not touch);
  `test_nn.py -k onv` (90 passed); `test_ops.py -k conv` (909 passed; the 6
  `test_dtypes_*conv*` failures are the runner's expected ones: vk also
  takes integer convolutions).
