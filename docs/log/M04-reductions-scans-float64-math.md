# Reductions, scans, float64 math (M4)

- `shaders/reduce.slang` + `csrc/reduce.cpp`: tree reduction over any set of
  dims (up to 4 non-mergeable kept and reduced dims each). Large reductions
  are split into chunks across workgroups and the partials reduced again, so
  float32 sums stay as accurate as the CPU's. When the reduced dims are
  strided and outputs are adjacent (column reductions), each thread owns one
  output so reads are coalesced. Integer and bool inputs accumulate in int64,
  as in PyTorch. Ops: `sum`, `mean`, `prod`, `amax`/`amin`, `max`/`min`
  (NaN-propagating), `var`/`std` (two passes, any correction),
  `linalg.vector_norm` (ord 1, 2, ±inf), `any`/`all`.
- `shaders/argreduce.slang`: (value, index) reduction for `argmax`/`argmin`
  and `max`/`min` along a dim; ties and NaNs resolve to the first index, as on
  the CPU.
- `shaders/scan.slang`: `cumsum`/`cumprod` along any dim: per-chunk workgroup
  scans, the chunk totals scanned recursively, then added back.
- `shaders/_math64.slang`: float64 `exp`, `exp2`, `log`, `log2`, `sin`, `cos`,
  `tan`, `tanh`, `sigmoid` (SPIR-V has no 64-bit transcendentals). Measured
  against PyTorch's CPU over 200k inputs each plus special values: exp/exp2 ≤ 1
  ULP, log/log2/sigmoid/tan ≤ 2 ULP, tanh ≤ 3 ULP, sin/cos ≤ 1 ULP of scale.
  sin/cos/tan reduce arguments exactly up to |x| ≈ 1.6e6.
- Kernels preserve denormals (`-denorm-mode-fp32/fp64 preserve`): the UHD 620
  otherwise flushes subnormal float64 results to zero.

Measured (`python benchmarks/elementwise.py`, 16M elements, UHD 620 vs 4-thread CPU):

| | vk float32 | CPU float32 | vk float64 | CPU float64 |
|---|---|---|---|---|
| `sum()` | 2.5 ms | 2.7 ms | 8.7 ms | 4.9 ms |
| `var(dim=1)`, 4096×4096 | 6.3 ms | 93 ms | 17 ms | 88 ms |
| `amax(dim=0)`, 4096×4096 | 4.7 ms | 6.3 ms | 13 ms | 14 ms |
| `cumsum` | 23 ms | 31 ms | 52 ms | 34 ms |
