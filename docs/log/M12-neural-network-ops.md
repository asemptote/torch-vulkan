# Neural-network ops (M12)

A small CNN trains on `"vk"` with every op on the GPU (`tests/python/test_nn.py`
checks forward results and gradients against the CPU).

- **Convolution** (`csrc/convolution.cpp`): 1-3 spatial dims, groups,
  stride, padding, dilation, transposed. Forward is im2col (one copy of a
  strided view of the padded input) and a batched GEMM; a transposed
  convolution is a plain one of the zero-inserted input with the flipped
  kernel; grad_input is a transposed convolution and grad_weight a GEMM.
  Integer convolutions run on the CPU.
- **Batch norm** (all entry points, with running-statistics updates) and
  the attention path (`_scaled_dot_product_flash_attention_for_cpu`, chosen by
  PyTorch for non-CUDA devices) compose existing kernels
  (`csrc/normalization.cpp`).
- **PyTorch's decompositions** (`python/torch_vulkan/_decomp.py`): ops with
  no Vulkan kernel but a decomposition in `torch._decomp` (~170: softmax and
  log-softmax, layer and group norm, losses, max pooling, upsampling,
  `grid_sampler_2d`, `im2col`/`col2im`, most activation backward ops, ...) are
  registered as "vk" kernels written in terms of ops that run on Vulkan.
  Only functional and out= overloads are registered, never over an existing
  kernel; a decomposition that calls itself or declines its arguments runs
  on the CPU (counted in `fallback_counts()`).
- **Python kernels from Vulkan ops** (`python/torch_vulkan/_kernels.py`) for
  ops with neither: average pooling (a depthwise convolution with a ones
  kernel over window sizes), adaptive pooling, upsampling backward and
  antialiased upsampling (separable per-dim matrices: interpolation weights
  from PyTorch's formulas, applied with matmul; the backward pass applies the
  transposes; uint8 bilinear/bicubic as the CPU's fixed-point kernel, bit
  for bit: int16 weights, a width then a height pass of integer sums,
  shifted and clamped), grid sampling backward and 3-D forward, fractional
  max pooling, `cdist`/`pdist`, `rrelu` noise, `embedding_bag`.
- 8- and 16-bit types (uint8, int8, int16, float16, bfloat16) now compute in
  element-wise ops, in int32/float32 and cast back as on the CPU.

OpInfo sweep: 446 of 689 ops fully on the GPU (371 after M11), no
mismatches.
