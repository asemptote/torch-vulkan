# Intel(R) UHD Graphics 620: torch-vulkan against the CPU

- GPU: Intel(R) UHD Graphics 620, Vulkan driver Intel Corporation, Intel driver, version 101.2134
- CPU baseline: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz, 4 threads, PyTorch 2.14.1+cpu
- OS: Windows 11
- torch-vulkan commit 96637d5, 2026-11-11 (fft: commit 93d188a, 2026-11-11, from its
  release dry-run's wheel)
- `python benchmarks/integrated.py uhd620-windows matmul fft division`; fft:
  `python benchmarks/fft.py`, each time the median over 7 blocks of 5 calls

## matmul

`matmul.py`:

```
Intel(R) UHD Graphics 620 (TORCH_VK_GEMM=default) vs CPU (4 threads), torch 2.14.1+cpu
  float32    256^3   vk    31.3   vk a@b.T    16.8   cpu   119.0 GFLOP/s
  float32   1024^3   vk    66.6   vk a@b.T    84.4   cpu   148.2 GFLOP/s
  float32   2048^3   vk   105.8   vk a@b.T   111.3   cpu   111.1 GFLOP/s
  float64    256^3   vk    17.6   vk a@b.T    17.2   cpu    37.5 GFLOP/s
  float64   1024^3   vk    22.2   vk a@b.T    22.4   cpu    53.0 GFLOP/s
  float64   2048^3   vk    26.9   vk a@b.T    26.2   cpu    58.1 GFLOP/s
  float16    256^3   vk    27.4   vk a@b.T    25.7   cpu     0.6 GFLOP/s
  float16   1024^3   vk    66.7   vk a@b.T    73.3   cpu     0.3 GFLOP/s
  float16   2048^3   vk   116.9   vk a@b.T    92.5   cpu     0.2 GFLOP/s
  CPU fallbacks: none
```


## fft

`fft.py`:

```
Intel(R) UHD Graphics 620 vs CPU (4 threads), torch 2.14.1+cpu
  fft 2^20                 complex64  vk     12.2 ms   cpu     12.7 ms
  fft 64 x 2^14            complex64  vk      5.0 ms   cpu      3.8 ms
  fft 1000003 (Bluestein)  complex64  vk     39.2 ms   cpu    221.3 ms
  fft2 1024x1024           complex64  vk     14.6 ms   cpu      3.5 ms
  rfft 2^22                complex64  vk     17.0 ms   cpu     36.3 ms
  irfft 2^20               complex64  vk      4.8 ms   cpu      8.9 ms
  fft 2^20                 complex128 vk     14.2 ms   cpu     15.5 ms
  fft 64 x 2^14            complex128 vk      9.5 ms   cpu      4.5 ms
  fft 1000003 (Bluestein)  complex128 vk     69.5 ms   cpu    164.4 ms
  fft2 1024x1024           complex128 vk     26.8 ms   cpu      7.6 ms
  rfft 2^22                complex128 vk     36.7 ms   cpu     72.0 ms
  irfft 2^20               complex128 vk     10.9 ms   cpu     19.8 ms
  CPU fallbacks: none
```

This laptop's CPU times vary up to 2x from run to run (a 15 W part, with turbo and
thermal limits); the GPU times mostly within 15% (Bluestein complex64: 27 to 39 ms).


## division

`elementwise.py division`:

```
Intel(R) UHD Graphics 620 vs CPU (4 threads), torch 2.14.1+cpu
  x / y                  16M float32  vk    23.4 ms   cpu    18.6 ms
  1 / x                  16M float32  vk    23.1 ms   cpu    18.6 ms
  x / y                  16M float64  vk    63.2 ms   cpu    26.7 ms
  1 / x                  16M float64  vk    74.8 ms   cpu    39.3 ms
  CPU fallbacks during benchmark: none
```
