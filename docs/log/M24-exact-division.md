# Correctly rounded float32 division without FMAs (M24)

Vulkan allows OpFDiv an error of 2.5 ulp, so `_fmath.slang`'s `m_div` has
corrected the GPU's quotient with its exact remainder, `fma(-q, b, a)`. That
only works where the driver fuses the FMA. The UHD 620's Windows driver does
not, and about 12% of float32 `x / y` results and 11% of `1 / x` results on
vk were 1 ulp off the CPU's (issue #19, found while matching the CPU's 4-bit
quantised matmul bit for bit: M23's `div_exact`). `torch.reciprocal` divided
with a bare `1 / x`.

- **m_div32** (float32's `m_div`, also used by reciprocal now). Zeros,
  infinities and NaNs keep the GPU's division, which is exact there; they are
  detected on the bits, since a device that flushes subnormals may compare
  them equal to 0.
  - **Common case** (a, b and the quotient within 2^±60): the quotient is
    corrected once with the residual a - q b, computed exactly from Dekker's
    product with Veltkamp's split (plain multiplications and additions, which
    `-fp-mode precise` keeps unfused). It is kept when the exact residual is
    clearly under half an ulp of q times b.
  - **Otherwise** (and in the rare near-halfway case): the significands are
    divided in [1, 2), so no overflow and no subnormal intermediate is
    possible. Their quotient is chosen among its neighbours by exact
    residuals. Exponent, overflow and subnormal results are assembled on the
    bits, with subnormal ties broken by the residual's sign. So subnormal
    inputs and results come out as the CPU's even on devices that flush
    subnormals in arithmetic.
- **Agreement**, UHD 620, against the CPU bit for bit (NaN with NaN), 2^21
  pairs each:

  | Inputs | x / y mismatches, before → after | 1 / x mismatches, before → after |
  |---|---|---|
  | randn | 250,327 → 0 | 240,433 → 0 |
  | uniform bit patterns (subnormals included) | 187,057 → 0 | 228,289 → 0 |
  | huge / tiny | 205,728 → 0 | 241,000 → 0 |
  | tiny / huge | 11,233 → 0 | 240,954 → 0 |
  | all pairs of zeros, infinities, NaN, extremes | 1 → 0 | 18 → 0 |

- **Cost**, UHD 620, 16M float32 values (benchmarks/elementwise.py, now with
  division lines): `x / y` 10.5-12.2 → 15.5 ms, `1 / x` 14.6-15.0 → 20 ms
  (the CPU: 12 and 17 ms). This integrated GPU is compute-bound here; on
  discrete GPUs division is memory-bound and the extra arithmetic should
  disappear. The checks without the fast path cost twice as much (21-27 ms).
  float64 keeps the FMA correction (no device here showed float64 misrounding).
