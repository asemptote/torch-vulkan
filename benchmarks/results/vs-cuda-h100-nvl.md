vk: NVIDIA H100 NVL; CUDA: NVIDIA H100 NVL; torch 2.14.1+cpu, 2.14.1+cu130; NVIDIA driver 615.71.09 (Vulkan and CUDA 13.0)

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster). FFT rows from 96637d5, the others from db43372 (the same non-FFT code).

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
| fft 2^20 complex64 | ms | 0.0618 | 0.0286 | 2.2 |
| rfft 2^20 float32 | ms | 0.0506 | 0.0285 | 1.8 |
| fft 2^20 complex128 | ms | 0.101 | 0.0471 | 2.1 |
| rfft 2^20 float64 | ms | 0.062 | 0.0402 | 1.5 |
| fft 2^24 complex64 | ms | 0.539 | 0.32 | 1.7 |
| rfft 2^24 float32 | ms | 0.322 | 0.208 | 1.5 |
| fft 2^24 complex128 | ms | 1.01 | 0.626 | 1.6 |
| rfft 2^24 float64 | ms | 0.619 | 0.4 | 1.5 |
| linalg.solve float64 n=1024 | ms | 11.6 | 3.4 | 3.4 |
| eigh float64 n=1024 | ms | 41.2 | 12 | 3.4 |
| svd float64 n=1024 | ms | 75.9 | 96 | 0.79 |
| linalg.solve float64 n=4096 | ms | 58.1 | 18.1 | 3.2 |
| eigh float64 n=4096 | ms | 318 | 131 | 2.4 |
| svd float64 n=4096 | ms | 1518 | 2254 | 0.67 |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 2489 | 3313 | 1.3 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 4.88 | 5.45 | 0.9 |
| rnn trajectory float32 N=1000, 1000 steps | ms | 21 | 28.8 | 0.73 |
| rnn trajectory float32 N=4000, 1000 steps | ms | 36.5 | 34.6 | 1.1 |
| rnn batch of 256 float32 N=4000, 200 steps | ms | 132 | 48.2 | 2.7 |
| rnn Lyapunov exponents k=64 float32 N=1000, 500 steps | ms | 169 | 49.5 | 3.4 |
| rnn trajectory float64 N=1000, 1000 steps | ms | 21.4 | 28.9 | 0.74 |
| rnn trajectory float64 N=4000, 1000 steps | ms | 52.9 | 54.3 | 0.97 |
| rnn batch of 256 float64 N=4000, 200 steps | ms | 313 | 51.5 | 6.1 |
| rnn Lyapunov exponents k=64 float64 N=1000, 500 steps | ms | 177 | 51.1 | 3.5 |

The rnn rows (benchmarks/rnn.py, a chaotic rate RNN after Sompolinsky, Crisanti and Sommers, whole runs) are from main d1c2e24 with this file; the two Lyapunov rows are from #65 (torch.linalg.qr's panels over many workgroups, #62), which took them from 21x CUDA on the H100 NVL and 14x / 5.2x on the L40S. The Lyapunov exponents agree between the two devices: max |difference| over the 64 leading exponents 1.4e-16 in float64 and 1.7e-7 in float32.
