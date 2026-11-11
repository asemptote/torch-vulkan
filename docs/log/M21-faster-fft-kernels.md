# Faster FFT kernels (M21)

- **fft_radix.slang**: rows longer than fft.slang's LOCAL_MAX took log2(n)
  radix-2 passes over global memory (20 for 2^20 points); they now take
  ceil(log2(n) / 7) Stockham passes of radix r <= 128 (the bits of n split
  evenly). A workgroup holds 1024 complex64 or 512 complex128 values (8 KiB
  of shared memory): g = 1024 / r groups, multiplied by their twiddles while
  loading, then the r-point DFTs as radix-4 stages in place (one butterfly
  per thread, a last radix-2 stage for odd log2(r)), then stored
  contiguously. Indices are shifts and masks (integer division roughly
  doubled the kernel's cost on the UHD 620). Twiddles still come from the
  float64 host table.
- **Short rows**: rows up to that budget take one radix-n pass with several
  rows per workgroup. fft.slang's FFT_LOCAL ran one row per 256-thread
  workgroup, mostly idle for short rows (2^20 complex64 values as rows of 2:
  95 -> 2.3 ms; rows of 64: 15 -> 3.8 ms). FFT_LOCAL keeps rows between the
  budget and twice it, where it beats two radix passes.
- **Real transforms at half length** (fft.slang FFT_R2C and FFT_C2R): an
  even-length real row read as complex pairs x[2j] + i x[2j + 1] takes a
  half-length FFT plus one O(n) pass that separates the even and odd
  samples' spectra, instead of a full-length complex FFT. irfft ignores the
  imaginary parts of X[0] and X[n/2] as the CPU does. Multi-dim rfftn and
  irfftn take the last dim's real transform first, as the CPU does.
- **Output layouts**: a transform along a non-last dim made its transposed
  result contiguous, then copied it back to the CPU's (MKL's) layout, which
  is the transposed one: three transposing copies where one is needed.
  Results are now copied only when their layout differs from the CPU's
  (`fft(x, dim=0)` on 1024 x 1024 complex64: 22 -> 6 ms).
- UHD 620, benchmarks/fft.py (ms, main -> M21, complex64 / complex128; the
  CPU's 4 threads in brackets): fft 2^20 32.9 -> 12.4 (6) / 47.8 -> 17.4
  (14); 64 x 2^14 12.6 -> 6.8 (2) / 25.1 -> 8.8 (4); Bluestein 1000003
  117 -> 41 (100-125) / 255 -> 75 (170); rfft 2^22 131 -> 19 (23-28) /
  288 -> 45 (51); irfft 2^20 24.4 -> 4.2 (6.5) / 56.4 -> 9.5 (13). fft2
  1024^2 12.6 -> 12.4 (2.7) / 42.7 -> 39-42 (6.7): the benchmark's five
  back-to-back calls include allocations; synchronized calls take 23 ms
  (rows 8, columns 12, mostly the transposing copy). Errors against NumPy's
  float64 FFT are equal or lower (complex128 2^20: 7.8e-16 -> 5.9e-16).
- Next: column transforms without the transposing copy (strided loads of
  several adjacent columns per workgroup), and a faster strided copy.
