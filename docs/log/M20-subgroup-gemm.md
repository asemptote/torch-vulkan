# Subgroup GEMM (M20)

- **gemmsg.slang**: float32 and float64 products keep A in registers and
  share it across a subgroup by broadcasts. A workgroup of 128 threads in
  subgroups of S lanes computes a (128 / S) 8 x 4S tile of C: B's S x 4S
  block goes to shared memory once per k-block, each lane loads
  A[row, k0 + lane] for its subgroup's 8 rows, and per kk a lane reads 4
  values of B and takes A[row, k0 + kk] from lane kk (WaveReadLaneAt with a
  constant lane once unrolled): 32 multiply-adds per 4 shared-memory loads,
  against gemm.slang's 16 per 8.
- **Subgroup size control** (vkcore): Intel's driver reports 32 lanes but
  may run a kernel 16 wide, and broadcasts across the two halves then read
  garbage. vkcore enables VK_EXT_subgroup_size_control (core in 1.3) with
  full subgroups and creates SUBGROUP_KERNELS' pipelines with a required
  size: 16 lanes for float32, 8 for float64 (on Intel's SIMD8 a lane has
  twice SIMD16's registers, so 8 x 4 doubles do not spill; at 16 lanes
  float64 ran at 5 GFLOP/s). TORCH_VK_SUBGROUP overrides the preference;
  TORCH_VK_DISABLE=subgroups hides the feature. Arm GPUs keep gemm4.slang
  and integers gemm.slang.
- UHD 620, benchmarks/matmul.py, alternating runs (GFLOP/s, tiles -> gemmsg):
  float32 2048^3 a @ b 87-90 -> 147-148 (MKL on the CPU: 98-120), a @ b.T
  76-81 -> 117-121, 1024^3 48-50 -> 68; float64 2048^3 22-23 -> 29.5.
- Tried and dropped: B read from global memory instead of shared (bound by
  B loads, 85 GFLOP/s), gemm.slang with 128 x 128 tiles and 8 x 8
  accumulators at 8 lanes (float64 spilled: 7 GFLOP/s), 256-thread
  workgroups and two k-blocks per barrier (slower). Laptop timings drift
  with temperature after a build: compare kernels in alternating runs.
- Next: a cooperative-matrix path (VK_KHR_cooperative_matrix) where the
  device has it (the UHD 620 has not; NVIDIA GPUs do): float16 in M22.
