# Intel(R) HD Graphics 530 (SKL GT2): torch-vulkan against the CPU

- GPU: Intel(R) HD Graphics 530 (SKL GT2), Vulkan driver Intel open-source Mesa driver, Mesa 26.0.8-1ubuntu0.3, version 26.0.8
- CPU baseline: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz, 4 threads, PyTorch 2.14.1+cpu
- OS: Linux 7.0.0-43-generic
- torch-vulkan commit cccade8, 2026-11-11
- `python benchmarks/integrated.py hd530-mesa matmul linalg eigh elementwise`

## matmul

`matmul.py`:

```
Intel(R) HD Graphics 530 (SKL GT2) (TORCH_VK_GEMM=default) vs CPU (4 threads), torch 2.14.1+cpu
  float32    256^3   vk    20.2   vk a@b.T    25.6   cpu   294.0 GFLOP/s
  float32   1024^3   vk    97.4   vk a@b.T    88.9   cpu   235.0 GFLOP/s
  float32   2048^3   vk    95.5   vk a@b.T    90.2   cpu   367.6 GFLOP/s
  float64    256^3   vk    23.2   vk a@b.T    23.2   cpu   166.3 GFLOP/s
  float64   1024^3   vk    26.4   vk a@b.T    26.0   cpu   171.2 GFLOP/s
  float64   2048^3   vk    26.4   vk a@b.T    26.1   cpu   174.2 GFLOP/s
  float16    256^3   vk    37.5   vk a@b.T    37.4   cpu     1.6 GFLOP/s
  float16   1024^3   vk    83.3   vk a@b.T    65.7   cpu     1.1 GFLOP/s
  float16   2048^3   vk    88.9   vk a@b.T    73.1   cpu     0.3 GFLOP/s
  CPU fallbacks: none
```


## linalg

`linalg.py`:

```
Intel(R) HD Graphics 530 (SKL GT2) vs CPU (4 threads), torch 2.14.1+cpu
  solve       512 float32  vk     31.4 ms   cpu      3.1 ms
  cholesky    512 float32  vk     10.3 ms   cpu      1.3 ms
  inv         512 float32  vk     35.3 ms   cpu      6.5 ms
  solve      2048 float32  vk    314.0 ms   cpu     42.9 ms
  cholesky   2048 float32  vk    151.6 ms   cpu     26.3 ms
  inv        2048 float32  vk    568.5 ms   cpu    100.1 ms
  solve       512 float64  vk     42.4 ms   cpu      5.8 ms
  cholesky    512 float64  vk     15.2 ms   cpu      5.2 ms
  inv         512 float64  vk     56.7 ms   cpu     11.9 ms
  solve      2048 float64  vk    588.5 ms   cpu     71.8 ms
  cholesky   2048 float64  vk    325.8 ms   cpu     43.5 ms
  inv        2048 float64  vk   1402.6 ms   cpu    177.9 ms
  CPU fallbacks: none
```


## eigh

`eigh.py`:

```
Intel(R) HD Graphics 530 (SKL GT2), torch 2.14.1+cpu, TORCH_VK_LINALG=gpu
  diagonal float64 n= 128                 11.5 ms  (dispatches only)
  diagonal float64 n= 256                 26.8 ms  (dispatches only)
  diagonal float64 n= 512                 90.2 ms  (dispatches only)
  random   float32 n= 128                 18.4 ms
  random   float32 n= 256                 25.0 ms
  random   float32 n= 512                 74.2 ms
  random   float32 n=1024                285.6 ms
  random   float64 n= 128                 22.8 ms
  random   float64 n= 256                 40.0 ms
  random   float64 n= 512                117.4 ms
  random   float64 n=1024                504.7 ms
  eigh     float32    n= 512              92.6 ms  (with eigenvectors)
  eigh     float32    n=1024             350.7 ms  (with eigenvectors)
  eigh     float32    n=2048            2029.0 ms  (with eigenvectors)
  eigh     float64    n= 512             151.6 ms  (with eigenvectors)
  eigh     float64    n=1024             649.0 ms  (with eigenvectors)
  eigh     float64    n=2048            3909.1 ms  (with eigenvectors)
  eigh     complex128 n= 512             274.9 ms  (with eigenvectors)
  eigh     complex128 n=1024            1427.2 ms  (with eigenvectors)
  random   float64 n= 256, batch of 4       94.1 ms
```

`eigh.py TORCH_VK_LINALG=cpu`:

```
Intel(R) HD Graphics 530 (SKL GT2), torch 2.14.1+cpu, TORCH_VK_LINALG=cpu
  diagonal float64 n= 128                  0.4 ms  (dispatches only)
  diagonal float64 n= 256                  1.0 ms  (dispatches only)
  diagonal float64 n= 512                  2.9 ms  (dispatches only)
  random   float32 n= 128                  0.8 ms
  random   float32 n= 256                  2.3 ms
  random   float32 n= 512                  7.2 ms
  random   float32 n=1024                 29.2 ms
  random   float64 n= 128                  0.9 ms
  random   float64 n= 256                  2.8 ms
  random   float64 n= 512                  9.8 ms
  random   float64 n=1024                 48.0 ms
  eigh     float32    n= 512              10.9 ms  (with eigenvectors)
  eigh     float32    n=1024              47.8 ms  (with eigenvectors)
  eigh     float32    n=2048             361.6 ms  (with eigenvectors)
  eigh     float64    n= 512              13.8 ms  (with eigenvectors)
  eigh     float64    n=1024              78.5 ms  (with eigenvectors)
  eigh     float64    n=2048             720.4 ms  (with eigenvectors)
  eigh     complex128 n= 512              28.9 ms  (with eigenvectors)
  eigh     complex128 n=1024             195.4 ms  (with eigenvectors)
  random   float64 n= 256, batch of 4       10.4 ms
```


## elementwise

`elementwise.py`:

```
Intel(R) HD Graphics 530 (SKL GT2) vs CPU (4 threads), torch 2.14.1+cpu
  per-op overhead, 1024 floats:     vk    12.4 us   cpu     3.2 us
  relu(x*2+1), 16M float32        vk    40.0 ms   cpu    42.8 ms   (10.1 GB/s on vk)
  relu(x*2+1), 16M float64        vk    41.7 ms   cpu    85.8 ms   (19.3 GB/s on vk)
  x / y                  16M float32  vk    17.9 ms   cpu    16.0 ms
  1 / x                  16M float32  vk    28.4 ms   cpu    34.8 ms
  x / y                  16M float64  vk    54.6 ms   cpu    30.8 ms
  1 / x                  16M float64  vk    65.6 ms   cpu    56.7 ms
  sum()                  16M float32  vk     4.4 ms   cpu     2.3 ms
  var(dim=1) 4096x4096   16M float32  vk    11.0 ms   cpu    31.0 ms
  amax(dim=0) 4096x4096  16M float32  vk     5.5 ms   cpu     5.4 ms
  cumsum                 16M float32  vk    33.7 ms   cpu    52.2 ms
  sum()                  16M float64  vk     6.3 ms   cpu     5.3 ms
  var(dim=1) 4096x4096   16M float64  vk    15.5 ms   cpu    26.4 ms
  amax(dim=0) 4096x4096  16M float64  vk     8.0 ms   cpu    14.0 ms
  cumsum                 16M float64  vk    46.0 ms   cpu    85.9 ms
  CPU fallbacks during benchmark: none
```
