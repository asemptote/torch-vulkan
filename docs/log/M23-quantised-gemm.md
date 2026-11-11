# Quantised matrix products (M23)

The three quantised GEMM-shaped ops ran on the CPU; they now run on the GPU,
computed as the CPU's kernels compute them (csrc/quant.cpp, with the CPU's
argument checks and messages).

- **_weight_int8pack_mm** (`shaders/qmm8.slang`): float32, float16 or
  bfloat16 A against int8 weights [N, K] with per-channel scales. Weights are
  dequantised as float(b) * scale while staged in shared memory (no float
  copy of B), products summed in float32, the result rounded once, as the
  CPU's x86 kernels do (in another order: within float32 rounding). Rows of
  A up to 8 (a decoding step) take a matrix-vector path: 8 output columns
  per workgroup, 32 threads each along k. UHD 620, 4096 x 4096 weights:
  1 row 5.0 ms (CPU 13.3), 8 rows 14 ms (105), 32 rows 29 ms (422); the
  64 x 64 tiles alone took 52 ms for one row.
- **_dyn_quant_pack_4bit_weight / _dyn_quant_matmul_4bit**
  (`shaders/qmm4.slang`): bit for bit as the CPU's reference kernels
  (int4mm_kernel.cpp, used on every CPU without KleidiAI). The packed weights
  keep the CPU's layout (a float32 vector of the 4-bit weights' bytes, the
  scales, the bias), so either device's packing works with either's matmul.
  One workgroup per input row finds its range and computes the int8 scale,
  its reciprocal and the zero point on one thread, then quantises the row
  (std::round's halfway-away rounding, lrintf's to-even); the product sums
  (q - zp) w exactly in int32 per block of k, then scales per block in
  float32, in the CPU's order, and bfloat16 outputs are truncated as the
  CPU's bf16 kernel truncates. Two details only showed against the CPU:
  - the zero point travels as int bits, not through a float (a small int
    read as a float is a denormal, which the GPU flushed);
  - the scale's divisions must be correctly rounded. _fmath.slang's m_div
    corrects the quotient with an FMA, but the UHD 620's driver does not fuse
    FMAs: 14% of torch.div's float32 quotients there are 1 ulp off the CPU's
    (an open issue for the elementwise division too). qmm4's div_exact gets
    the exact residual from Dekker's product instead and picks the nearest
    of the quotient's neighbours.
  32 x 4096 x 4096 with groups of 32: 60 ms (CPU 1.09 s).
- **_scaled_mm** (and .out): the CPU's emulated path on vk tensors (float8
  upcast to float32 exactly, tensor-wise or row-wise scales, float32 product,
  bias, 1 / scale_result for float8 outputs, cast), with the CPU's scaling
  checks (get_joint_scaling's "Invalid scaling configuration" message). Equal
  to the CPU's results in the tests (float8 products are exact in float32).
- Tests: tests/python/test_quant.py (exact comparisons for the 4-bit and
  float8 ops, errors, no fallbacks); PyTorch's test_linalg `_int8_mm` /
  `_dyn_quant` (eager ones pass; torch.compile ones are Inductor's), test_ops
  and test_decomp `_scaled_mm`. One test__dyn_quant_matmul_4bit case is
  EXPECTED: its 0.06 error bound holds for the CPU's random draws only (on
  vk's Philox draws the CPU's own kernels miss it by one element).
