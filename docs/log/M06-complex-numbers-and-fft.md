# Complex numbers and FFT (M6)

- complex64 / complex128 tensors (stored as float2 / double2): casts to and
  from every real dtype, fills, `+ - * /` (with complex scalars and real
  `alpha`; division uses Smith's algorithm), `neg`, `abs` (overflow-safe),
  `conj` / `conj_physical`, and the views `real`, `imag`, `view_as_real`,
  `view_as_complex` (they alias: `x.real += 1` updates `x`).
- Lazy conjugate/negative bits: `_copy_from` passes through PyTorch's
  Conjugate/Negative layers and applies the bits itself (otherwise
  materialising them recursed forever). The CPU fallback now refuses view
  ops (`error_on_views`) instead of silently returning copies.
- `shaders/fft.slang` + `csrc/fft.cpp`: `_fft_c2c`, `_fft_r2c`, `_fft_c2r`,
  which every `torch.fft` function reduces to. Power-of-two lengths use
  radix-2 Stockham passes (rows up to 2048 complex64 / 1024 complex128 run
  all passes in shared memory in one dispatch); other lengths use
  Bluestein's algorithm. Twiddle and chirp tables are computed on the host in
  float64 (the GPU's float32 sin/cos would be far too inaccurate) and cached.
  r2c transforms as complex and keeps n/2+1 bins; c2r rebuilds the Hermitian
  spectrum and keeps the real part. Error vs PyTorch's CPU FFT is a few ULP x
  log2(n); a 65536-point complex128 FFT agrees to < 1e-11.

Measured (`python benchmarks/fft.py`, UHD 620 vs 4-thread CPU):

| | vk complex64 | CPU | vk complex128 | CPU |
|---|---|---|---|---|
| fft 2^20 | 33 ms | 14 ms | 49 ms | 25 ms |
| fft2 1024×1024 | 12 ms | 6 ms | 27 ms | 12 ms |
| fft 1,000,003 (prime, Bluestein) | 119 ms | 212 ms | 249 ms | 345 ms |
| rfft 2^22 | 127 ms | 62 ms | 273 ms | 104 ms |

The FFTs are memory-bound radix-2 passes plus transposes between dims; on
this integrated GPU only long prime-length transforms beat the CPU so far.
Radix-4/8 passes, a shared-memory four-step FFT for long rows, and a real
r2c (half-length complex) path are the next steps (done in M21).
