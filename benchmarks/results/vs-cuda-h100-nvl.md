vk: NVIDIA H100 NVL; CUDA: NVIDIA H100 NVL; torch 2.14.1+cpu, 2.14.1+cu130

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster).

| workload | unit | vk | CUDA | vk time / CUDA time |
|---|---|---|---|---|
| matmul float16 2048^3 | TFLOP/s | 118 | 366 | 3.1 |
| matmul float16 8192^3 | TFLOP/s | 155 | 381 | 2.5 |
| matmul bfloat16 2048^3 | TFLOP/s | 127 | 381 | 3 |
| matmul bfloat16 8192^3 | TFLOP/s | 161 | 426 | 2.6 |
| matmul float32 2048^3 | TFLOP/s | 17.6 | 40.5 | 2.3 |
| matmul float32 8192^3 | TFLOP/s | 17.4 | 40 | 2.3 |
| matmul float64 2048^3 | TFLOP/s | 7.64 | 36.4 | 4.8 |
| matmul float64 8192^3 | TFLOP/s | 6.69 | 37.9 | 5.7 |
| matmul float32 2048^3, TF32 on (CUDA) | TFLOP/s |  | 191 |  |
| matmul float32 8192^3, TF32 on (CUDA) | TFLOP/s |  | 231 |  |
| fft 2^20 complex64 | ms | 0.0662 | 0.0289 | 2.3 |
| rfft 2^20 float32 | ms | 0.0616 | 0.0291 | 2.1 |
| fft 2^20 complex128 | ms | 0.104 | 0.0471 | 2.2 |
| rfft 2^20 float64 | ms | 0.0755 | 0.0403 | 1.9 |
| fft 2^24 complex64 | ms | 0.769 | 0.323 | 2.4 |
| rfft 2^24 float32 | ms | 0.543 | 0.208 | 2.6 |
| fft 2^24 complex128 | ms | 1.32 | 0.628 | 2.1 |
| rfft 2^24 float64 | ms | 0.834 | 0.411 | 2 |
| linalg.solve float64 n=1024 | ms | 11.6 | 3.4 | 3.4 |
| eigh float64 n=1024 | ms | 41.2 | 12 | 3.4 |
| svd float64 n=1024 | ms | 75.9 | 96 | 0.79 |
| linalg.solve float64 n=4096 | ms | 58.1 | 18.1 | 3.2 |
| eigh float64 n=4096 | ms | 318 | 131 | 2.4 |
| svd float64 n=4096 | ms | 1518 | 2254 | 0.67 |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 2489 | 3313 | 1.3 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 4.88 | 5.45 | 0.9 |
