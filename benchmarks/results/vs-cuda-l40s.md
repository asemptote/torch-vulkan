vk: NVIDIA L40S; CUDA: NVIDIA L40S; torch 2.14.1+cpu, 2.14.1+cu130; NVIDIA driver 615.71.09 (Vulkan and CUDA 13.0)

Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times slower vk is; under 1: faster). FFT rows from 96637d5, the others from db43372 (the same non-FFT code).

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
| fft 2^20 complex64 | ms | 0.0482 | 0.0258 | 1.9 |
| rfft 2^20 float32 | ms | 0.0428 | 0.0246 | 1.7 |
| fft 2^20 complex128 | ms | 0.209 | 0.202 | 1 |
| rfft 2^20 float64 | ms | 0.135 | 0.16 | 0.84 |
| fft 2^24 complex64 | ms | 1.38 | 1.48 | 0.93 |
| rfft 2^24 float32 | ms | 0.645 | 0.771 | 0.84 |
| fft 2^24 complex128 | ms | 3.18 | 2.94 | 1.1 |
| rfft 2^24 float64 | ms | 1.9 | 1.88 | 1 |
| linalg.solve float64 n=1024 | ms | 13.8 | 6.72 | 2.1 |
| eigh float64 n=1024 | ms | 48.8 | 25.9 | 1.9 |
| svd float64 n=1024 | ms | 148 | 798 | 0.18 |
| linalg.solve float64 n=4096 | ms | 91.6 | 66.3 | 1.4 |
| eigh float64 n=4096 | ms | 685 | 447 | 1.5 |
| svd float64 n=4096 | ms | 4374 | 13076 | 0.33 |
| elementwise chain 2^26 float32 (x * y + z).relu() | GB/s | 635 | 633 | 1 |
| per-op overhead: 1000 additions of 1024 floats | us/op | 3.98 | 4.61 | 0.86 |
| rnn trajectory float32 N=1000, 1000 steps | ms | 18.5 | 24.5 | 0.76 |
| rnn trajectory float32 N=4000, 1000 steps | ms | 21.5 | 24.9 | 0.86 |
| rnn batch of 256 float32 N=4000, 200 steps | ms | 105 | 50.8 | 2.1 |
| rnn Lyapunov exponents k=64 float32 N=1000, 500 steps | ms | 122 | 40.5 | 3 |
| rnn trajectory float64 N=1000, 1000 steps | ms | 18.9 | 24.5 | 0.77 |
| rnn trajectory float64 N=4000, 1000 steps | ms | 184 | 194 | 0.95 |
| rnn batch of 256 float64 N=4000, 200 steps | ms | 1388 | 1507 | 0.92 |
| rnn Lyapunov exponents k=64 float64 N=1000, 500 steps | ms | 395 | 176 | 2.2 |

The rnn rows (benchmarks/rnn.py, a chaotic rate RNN after Sompolinsky, Crisanti and Sommers, whole runs) are from main d1c2e24 with this file; the two Lyapunov rows are from #65 (torch.linalg.qr's panels over many workgroups, #62), which took them from 21x CUDA on the H100 NVL and 14x / 5.2x on the L40S. The Lyapunov exponents agree between the two devices: max |difference| over the 64 leading exponents 1.7e-16 in float64 and 1.3e-7 in float32.
