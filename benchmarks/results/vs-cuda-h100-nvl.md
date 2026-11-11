vk: NVIDIA H100 NVL; CUDA: NVIDIA H100 NVL; torch 2.14.1+cpu, 2.14.1+cu130; NVIDIA driver 615.71.09 (Vulkan and CUDA 13.0)

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster).

| workload | unit | vk | CUDA | vk time / CUDA time |
|---|---|---|---|---|
| matmul float16 2048^3 | TFLOP/s | 104 | 370 | 3.6 |
| matmul float16 8192^3 | TFLOP/s | 159 | 396 | 2.5 |
| matmul bfloat16 2048^3 | TFLOP/s | 13.5 | 380 | 28 |
| matmul bfloat16 8192^3 | TFLOP/s | 17.2 | 426 | 25 |
| matmul float32 2048^3 | TFLOP/s | 17.7 | 40.5 | 2.3 |
| matmul float32 8192^3 | TFLOP/s | 17.4 | 40.1 | 2.3 |
| matmul float64 2048^3 | TFLOP/s | 7.79 | 36.4 | 4.7 |
| matmul float64 8192^3 | TFLOP/s | 6.58 | 37.6 | 5.7 |
| matmul float32 2048^3, TF32 on (CUDA) | TFLOP/s |  | 192 |  |
| matmul float32 8192^3, TF32 on (CUDA) | TFLOP/s |  | 227 |  |
| fft 2^20 complex64 | ms | 0.0985 | 0.0286 | 3.4 |
| rfft 2^20 float32 | ms | 0.0922 | 0.0284 | 3.2 |
| fft 2^20 complex128 | ms | 0.137 | 0.0471 | 2.9 |
| rfft 2^20 float64 | ms | 0.106 | 0.0402 | 2.6 |
| fft 2^24 complex64 | ms | 1.12 | 0.321 | 3.5 |
| rfft 2^24 float32 | ms | 0.79 | 0.207 | 3.8 |
| fft 2^24 complex128 | ms | 1.8 | 0.627 | 2.9 |
| rfft 2^24 float64 | ms | 1.16 | 0.41 | 2.8 |
| linalg.solve float64 n=1024 | ms | 27.9 | 3.43 | 8.1 |
| eigh float64 n=1024 | ms | 882 | 12 | 73 |
| svd float64 n=1024 | ms | 401 | 100 | 4 |
| linalg.solve float64 n=4096 | ms | 353 | 18.1 | 19 |
| eigh float64 n=4096 | ms | 14615 | 131 | 1.1e+02 |
| svd float64 n=4096 | ms | 13731 | 2271 | 6 |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 2477 | 3305 | 1.3 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 4.96 | 5.48 | 0.91 |
