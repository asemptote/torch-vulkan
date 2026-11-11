# bfloat16 GEMM on cooperative matrices (M27)

M22 put float16 products on cooperative matrices (tensor cores). bfloat16
products still ran in float32 through gemmsg, at 13 to 19 TFLOP/s, against
CUDA's 380 to 430 on the H100 (issue #26).

- **gemmcm.slang** gains a bfloat16 variant (axis T: bf16 = uint16_t, the
  storage type; fragments of Slang's `BFloat16`).
  - The staging moves 16-bit values as bits for both types, and the only
    per-type code is converting Cin and the result: bfloat16 is read by
    shifting its bits into a float32, and written rounded to nearest even
    (NaN to 0x7fc0), as cast.slang does.
  - As for float16, products accumulate in float32 and the result is rounded
    once.
- **vkcore** has a new capability, `Capabilities::cooperative_matrix_bf16`.
  It needs:
  - VK_KHR_shader_bfloat16, with shaderBFloat16Type and
    shaderBFloat16CooperativeMatrix;
  - a 16 x 16 x 16 cooperative-matrix shape with bfloat16 A and B and float32
    C;
  - the float16 path's other requirements (subgroup size control at 32 lanes,
    16-bit storage, the Vulkan memory model with device scope, 38 KiB of
    shared memory).

  ash 0.38 predates VK_KHR_shader_bfloat16, so the extension's name, feature
  structure and component type are declared in context.rs. Kernel loading
  checks the BFloat16TypeKHR and BFloat16CooperativeMatrixKHR capabilities.
  `TORCH_VK_DISABLE=bfloat16` hides the bfloat16 path (and `coopmat` hides
  both).
- **csrc/linalg.cpp**: `coop_gemm_16` takes float16 and bfloat16, and
  `vkt_capabilities` gains bit 32. test_matmul's float16 cooperative-matrix
  test is parametrised over both types; its tolerance is one ulp of the type
  plus float32 summation error.

## Results

`benchmarks/vs_cuda.py`, matmul bfloat16, TFLOP/s:

| GPU | size | before | now | CUDA | now / CUDA |
|---|---|---|---|---|---|
| H100 NVL | 2048^3 | 13.5 | 110 | 381 | 3.5x slower |
| H100 NVL | 8192^3 | 17.2 | 163 | 426 | 2.6x slower |
| L40S | 2048^3 | 16.7 | 123 | 223 | 1.8x slower |
| L40S | 8192^3 | 19.3 | 164 | 194 | 1.2x slower |

These are float16's rates. As M22 noted, the H100 is far below its tensor
peak, presumably because Vulkan's cooperative matrices do not reach Hopper's
warpgroup MMA instructions.

**Checks.**
- On both GPUs, test_matmul.py passes (62).
- On the H100, the cooperative-matrix tests are clean under the validation
  layer.
- tests/python on the H100: 1447 passed, 2 failed. The two failures were
  test_quant's tensor-wise `_scaled_mm` comparisons, which assumed exact
  results. float8 products times few-bit scales still exceed float32 once
  summed, in another order than the CPU's. The test now allows the float32
  summation error, as its row-wise cases already did, and the affected tests
  then passed on the H100 (60).
