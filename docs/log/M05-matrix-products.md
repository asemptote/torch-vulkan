# Matrix products (M5)

- `shaders/gemm.slang` + `csrc/linalg.cpp`: one batched GEMM,
  `C = alpha * A @ B + beta * Cin`, used by `mm`, `bmm`, `addmm`, `baddbmm`
  and `addmv` (PyTorch composes `matmul`, `mv`, `nn.Linear`, `einsum` from
  these); `dot` is a product plus a sum reduction. Operands are addressed by
  (row, column, batch) strides, so transposes (autograd's `A.t() @ grad`),
  slices and broadcast batches need no copies. `beta = 0` ignores `Cin`
  entirely (NaNs in it do not leak through), as on the CPU.
- 64x64 output tiles per 16x16 workgroup, K staged through shared memory in
  slices of 16, a 4x4 register block per thread. Larger tiles (128x128/8x8,
  128x64, 64x128) were measured and are slower on the UHD 620.
- float32 and float64; integer matrix products run on the CPU.

Measured (`python benchmarks/matmul.py`, UHD 620 vs 4-thread CPU with MKL):

| | vk | CPU |
|---|---|---|
| float32 1024³ | 55 GFLOP/s | 193 GFLOP/s |
| float32 2048³ | 87 GFLOP/s | 140 GFLOP/s |
| float64 2048³ | 23 GFLOP/s | 69 GFLOP/s |

On this integrated GPU, matrix products are still slower than MKL on the
CPU (the kernel reaches ~20% of the GPU's float32 peak); a subgroup-based
kernel and per-GPU tuning are future work. On discrete GPUs the balance differs.
