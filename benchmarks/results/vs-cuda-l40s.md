vk: NVIDIA L40S; CUDA: NVIDIA L40S; torch 2.14.1+cpu, 2.14.1+cu130; NVIDIA driver 615.71.09 (Vulkan and CUDA 13.0)

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster).

| workload | unit | vk | CUDA | vk time / CUDA time |
|---|---|---|---|---|
| matmul float16 2048^3 | TFLOP/s | 118 | 212 | 1.8 |
| matmul float16 8192^3 | TFLOP/s | 161 | 180 | 1.1 |
| matmul bfloat16 2048^3 | TFLOP/s | 16.7 | 224 | 13 |
| matmul bfloat16 8192^3 | TFLOP/s | 19.3 | 194 | 10 |
| matmul float32 2048^3 | TFLOP/s | 18.7 | 43.3 | 2.3 |
| matmul float32 8192^3 | TFLOP/s | 17.1 | 42.2 | 2.5 |
| matmul float64 2048^3 | TFLOP/s | 1.33 | 1.13 | 0.85 |
| matmul float64 8192^3 | TFLOP/s | 1.39 | 1.25 | 0.9 |
| matmul float32 2048^3, TF32 on (CUDA) | TFLOP/s |  | 111 |  |
| matmul float32 8192^3, TF32 on (CUDA) | TFLOP/s |  | 96.8 |  |
| fft 2^20 complex64 | ms | 0.0919 | 0.0258 | 3.6 |
| rfft 2^20 float32 | ms | 0.0711 | 0.0248 | 2.9 |
| fft 2^20 complex128 | ms | 0.253 | 0.202 | 1.3 |
| rfft 2^20 float64 | ms | 0.167 | 0.16 | 1 |
| fft 2^24 complex64 | ms | 2.21 | 1.49 | 1.5 |
| rfft 2^24 float32 | ms | 1.28 | 0.781 | 1.6 |
| fft 2^24 complex128 | ms | 4.86 | 2.94 | 1.7 |
| rfft 2^24 float64 | ms | 2.88 | 1.88 | 1.5 |
| linalg.solve float64 n=1024 | ms | 27.1 | 6.73 | 4 |
| eigh float64 n=1024 | ms | 1532 | 25.9 | 59 |
| svd float64 n=1024 | ms | 715 | 816 | 0.88 |
| linalg.solve float64 n=4096 | ms | 329 | 66.3 | 5 |
| eigh float64 n=4096 | ms | 23650 | 446 | 53 |
| svd float64 n=4096 | ms | over 90.0 s | 13086 |  |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 634 | 634 | 1 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 4.38 | 4.57 | 0.96 |
