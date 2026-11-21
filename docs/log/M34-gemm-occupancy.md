# Matrix products: tiles by SM count, split k (M34)

Issue #51. At 2048^3, gemmcm.slang's 128 x 256 tiles make 128 workgroups,
fewer than the H100 NVL's 132 SMs or the L40S's 142, and weight gradients
(`x.t() @ y`: a small result summed over many rows) used a fraction of
either GPU for the whole of k: a 1024 x 1024 result is 32 tiles.

## What changed

- **The GPU's size.** vkcore's `Capabilities::compute_units` holds the
  SM or compute-unit count, as llama.cpp's Vulkan backend reads it:
  VK_NV_shader_sm_builtins (`shaderSMCount`; 132 on the H100 NVL, 142 on
  the L40S), VK_AMD_shader_core_properties2 or VK_AMD_shader_core_properties,
  VK_ARM_shader_core_builtins, else a table of Intel's discrete GPUs (Xe
  cores by device ID). 0 where unknown (the HD 530, lavapipe, Mali without
  the Arm extension): products then run as before. `TORCH_VK_COMPUTE_UNITS`
  overrides it; `torch_vulkan._native.compute_units()` reads it.
- **Three tiles in gemmcm.slang** (a second axis, TILE): l 128 x 256 in 8
  subgroups (as before), m 128 x 128 and s 64 x 128 in 4 subgroups. The
  kernel was already written in terms of BM, BN, WM and WN.
- **Split k.** With the SPLIT spec constant, gemmcm.slang and gemmsg.slang
  run workgroup z = batch * S + s over k in [s kc, (s + 1) kc) only (kc a
  multiple of 256) and store their sums, unscaled, to a float32 workspace
  (float64 for float64); a new kernel, splitk.slang, adds the S pieces in
  the order s = 0, 1, ..., scales, adds beta * Cin and rounds once. The
  order is fixed, so results are the same bit for bit from run to run (and
  16-bit results are still rounded once, from float32 sums). The workspace
  is S x batch x m x n values from PyTorch's caching allocator.
- **The choice** (csrc/linalg.cpp `coop_plan`), for float16 and bfloat16:
  PLACEHOLDER
- **float32 and float64** (gemmsg.slang): PLACEHOLDER
- **Launch order.** gemmcm.slang can launch its tiles in bands of GROUP
  tile rows (column by column within a band) for L2 reuse, as NVIDIA's
  sample and llama.cpp's coopmat2 kernel do. PLACEHOLDER
- Tuning switches, for tests and measurements: `TORCH_VK_SPLIT_K=n`,
  `TORCH_VK_GEMM_TILE=l|m|s`, `TORCH_VK_GEMM_GROUP=g` (docs/api.md).

## Results

PLACEHOLDER

## Accuracy

Splitting k changes the order of the sums, not their precision: each piece
sums in float32 (float64) as before, and the pieces' sums are added in
float32 (float64), so the error bound shrinks from k to k / S + S
roundings. 16-bit results are still rounded once. test_matmul's
test_split_k_gemm checks float32, float64 and, where the device has
cooperative matrices, float16 and bfloat16 against the CPU with the
existing tolerances, on forced splits of 2 to 5 pieces, every tile, the
banded launch order and transposed, strided, batched and broadcast
operands, and that three runs agree bit for bit.

## Checks

PLACEHOLDER
