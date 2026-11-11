# Cooperative-matrix GEMM (M22)

- **gemmcm.slang**: float16 products on cooperative matrices
  (VK_KHR_cooperative_matrix, tensor cores on NVIDIA GPUs): 16 x 16 x 16
  fragments of float16 multiplied into float32 accumulators, each result
  scaled, added to Cin and rounded to float16 once (to nearest even), as
  16-bit products are everywhere else. Float32 products stay exact float32
  (no TF32): cooperative matrices only take the 16-bit inputs, so float32
  and float64 keep gemmsg.slang.
- A workgroup of 8 subgroups of 32 lanes computes a 128 x 256 tile of C,
  each subgroup 4 x 4 fragments. Per block of 32 values of k, A's and B's
  blocks are staged in shared memory as float16 in 16-byte pieces: one
  16-byte load where the operand is contiguous along either of its dims
  (spec constants A_MODE and B_MODE; csrc/linalg.cpp checks strides and
  alignment), else value by value with zeros past the edges, so any shape,
  stride or transposition works. B contiguous along k (x @ W.t()) is staged
  [n][k] and loaded as column-major fragments, with no transposing copy. The
  next block is loaded into registers while the subgroups multiply the
  current one.
- vkcore: Capabilities::cooperative_matrix_f16 when the device lists the
  16 x 16 x 16 float16 -> float32 subgroup shape and has subgroup size
  control with 32 lanes, float16 arithmetic and storage, the Vulkan memory
  model with device scope and 38 KiB of shared memory; it then enables the
  extension and the feature. The pipeline requires 32 lanes (kernels.rs
  SUBGROUP_KERNELS). TORCH_VK_DISABLE=coopmat hides it; any TORCH_VK_GEMM
  choice turns it off. Matrix-vector products (m or n = 1) keep the float32
  path.
- Tensor cores sum in another order than the CPU, with float32-like error:
  results match the CPU's float32 product rounded to float16 within one
  float16 ulp plus 2^-20 times the sum of the products' magnitudes (one
  cancelling dot product of 200 terms differed by 4 float16 ulps of its
  small result, 1.5e-8 of its terms' magnitudes).
- Tile shapes (TFLOP/s at 2048^3 to 8192^3, H100 NVL / L40S): 128 x 256
  145-149 / 147-157; 256 x 128 140-146 / 143-150; 128 x 128 112-118 /
  123-138 (with 64 values of k per block 123-126 on the H100); 64 x 128
  100-109 on the H100. Float16 products ran before in float32 (gemmsg.slang:
  16-18 TFLOP/s on the H100). Both GPUs reach about the same rate, far
  below the H100's float16 tensor peak: presumably Vulkan's cooperative
  matrices do not reach Hopper's warpgroup MMA instructions.
- Checked on the H100: tests/python (the two failures there,
  test_hash_tensor_and_frexp and
  test_fallback_with_several_outputs_writes_them_back, also fail with
  TORCH_VK_DISABLE=coopmat), test_matmul.py with the validation layer, and
  the OpInfo sweep's float16 products (mm, bmm, addmm, baddbmm, matmul,
  linear, einsum, tensordot, inner) strict.
- Next: bfloat16 (VK_KHR_shader_bfloat16's cooperative matrices; ash 0.38
  predates it), and NVIDIA's VK_NV_cooperative_matrix2 for larger tiles.
