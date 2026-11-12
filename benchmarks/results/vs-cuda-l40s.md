vk: NVIDIA L40S; CUDA: NVIDIA L40S; torch 2.14.1+cpu, 2.14.1+cu130

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster).

| workload | unit | vk | CUDA | vk time / CUDA time |
|---|---|---|---|---|
| matmul float16 2048^3 | TFLOP/s | 131 | 212 | 1.6 |
| matmul float16 8192^3 | TFLOP/s | 159 | 180 | 1.1 |
| matmul bfloat16 2048^3 | TFLOP/s | 139 | 223 | 1.6 |
| matmul bfloat16 8192^3 | TFLOP/s | 165 | 194 | 1.2 |
| matmul float32 2048^3 | TFLOP/s | 18.7 | 43.4 | 2.3 |
| matmul float32 8192^3 | TFLOP/s | 16.9 | 42.4 | 2.5 |
| matmul float64 2048^3 | TFLOP/s | 1.33 | 1.13 | 0.85 |
| matmul float64 8192^3 | TFLOP/s | 1.39 | 1.25 | 0.9 |
| matmul float32 2048^3, TF32 on (CUDA) | TFLOP/s |  | 111 |  |
| matmul float32 8192^3, TF32 on (CUDA) | TFLOP/s |  | 96.8 |  |
| fft 2^20 complex64 | ms | 0.0539 | 0.0258 | 2.1 |
| rfft 2^20 float32 | ms | 0.0439 | 0.0247 | 1.8 |
| fft 2^20 complex128 | ms | 0.247 | 0.202 | 1.2 |
| rfft 2^20 float64 | ms | 0.154 | 0.159 | 0.97 |
| fft 2^24 complex64 | ms | 1.83 | 1.49 | 1.2 |
| rfft 2^24 float32 | ms | 0.955 | 0.776 | 1.2 |
| fft 2^24 complex128 | ms | 4.19 | 2.93 | 1.4 |
| rfft 2^24 float64 | ms | 2.54 | 1.88 | 1.4 |
| linalg.solve float64 n=1024 | ms | 13.8 | 6.72 | 2.1 |
| eigh float64 n=1024 | ms | 48.8 | 25.9 | 1.9 |
| svd float64 n=1024 | ms | 148 | 798 | 0.18 |
| linalg.solve float64 n=4096 | ms | 91.6 | 66.3 | 1.4 |
| eigh float64 n=4096 | ms | 685 | 447 | 1.5 |
| svd float64 n=4096 | ms | 4374 | 13076 | 0.33 |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 635 | 633 | 1 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 3.98 | 4.61 | 0.86 |
