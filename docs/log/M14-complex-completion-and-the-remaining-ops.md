# Complex completion and the remaining ops (M14)

Coverage is measured, not listed: `tools/opinfo_coverage.py` runs PyTorch's
OpInfo samples (float32/float64/int64/bool/complex64) and reports per op
whether it ran without CPU fallbacks and matched the CPU (commit messages
record the numbers at each milestone).

- Complex: transcendental functions (`shaders/_cmath.slang`: exp/log/sqrt,
  trig and hyperbolic functions and their inverses, log1p/expm1, pow; the
  usual principal branches), matrix products (4 real GEMMs on the real and
  imaginary parts), reductions (sum/mean through the real view; prod and
  cumprod as log-depth trees of complex products; var/std/norms through
  |z|), accumulation (index_add, scatter_add, index_put/put with accumulate
  on the parts), logaddexp/logcumsumexp, lerp/addcmul/addcdiv, masked_fill,
  complex32 conversions.
- Complex linear algebra: `factor.slang` has complex variants following
  LAPACK's z* routines (izamax pivots, real Cholesky diagonals, zlarfg
  reflectors, Jacobi rotations that first remove the phase of the pivot
  entry), so LU, Cholesky, QR, eigh, SVD, lstsq and their relatives accept
  complex matrices with LAPACK's outputs.
- General `eig`/`eigvals` (`shaders/eig.slang`): Householder reduction to
  Hessenberg form, single-shift QR with Givens rotations to the complex Schur
  form, eigenvectors by back substitution, normalised as LAPACK's geev
  (unit norm, largest component real). Eigenvalues come in no particular
  order (as documented for `torch.linalg.eig`).
- `ldl_factor`/`ldl_solve`: Bunch-Kaufman as LAPACK's sytf2/hetf2 and
  sytrs/hetrs (same pivots and factors as the CPU).
- Special functions (`shaders/special.slang`, `_special.slang`): a
  mechanical port of PyTorch's own CPU implementations
  (`tools/port_special.py` regenerates it from the installed torch), in five
  modules (a single float64 module is too large for some drivers).
- float32 `sin`/`cos`/`tan` use Cody-Waite argument reduction and
  polynomials: the GPU built-ins are only accurate to ~1e-5 for moderate
  arguments, enough to break the Bessel functions' agreement with the CPU.
  It costs ~10-25% on sin/cos-heavy kernels (~2 ms per 16M elements).
- Integer and bool variants (rounding ops, fmax/fmin, heaviside, hardtanh,
  threshold, clamp on bools, max/min over bools, integer convolution and
  average pooling, int32 indices for gather/scatter) and composite kernels
  for `hash_tensor`, `frexp`, `histogram`/`histogramdd`, `segment_reduce`,
  CTC loss (forward and backward), `matrix_sqrth`, `polar`, `addbmm`.
