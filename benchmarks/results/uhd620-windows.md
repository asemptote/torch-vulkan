# Intel(R) UHD Graphics 620: torch-vulkan against the CPU

- GPU: Intel(R) UHD Graphics 620, Vulkan driver Intel Corporation, Intel driver, version 101.2134
- CPU baseline: Intel(R) Core(TM) i5-8350U CPU @ 1.70GHz, 4 threads, PyTorch 2.14.1+cpu
- OS: Windows 11
- torch-vulkan commit 96637d5, 2026-11-11
- `python benchmarks/integrated.py uhd620-windows matmul fft division`

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
  fft 2^20                 complex64  vk     17.7 ms   cpu      7.2 ms
  fft 64 x 2^14            complex64  vk      5.4 ms   cpu      2.4 ms
  fft 1000003 (Bluestein)  complex64  vk     47.9 ms   cpu    120.9 ms
  fft2 1024x1024           complex64  vk     23.8 ms   cpu      3.1 ms
  rfft 2^22                complex64  vk     22.2 ms   cpu     34.4 ms
  irfft 2^20               complex64  vk      6.5 ms   cpu      8.1 ms
  fft 2^20                 complex128 vk     31.1 ms   cpu     15.9 ms
  fft 64 x 2^14            complex128 vk     13.7 ms   cpu      4.7 ms
  fft 1000003 (Bluestein)  complex128 vk     89.0 ms   cpu    238.7 ms
  fft2 1024x1024           complex128 vk     29.1 ms   cpu      8.4 ms
  rfft 2^22                complex128 vk     58.8 ms   cpu     69.5 ms
  irfft 2^20               complex128 vk     11.0 ms   cpu     18.8 ms
  CPU fallbacks: none
```


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
