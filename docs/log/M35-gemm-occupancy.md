# Matrix products: tiles by SM count, split k (M35)

Issue #51. At 2048^3, gemmcm.slang's 128 x 256 tiles make 128 workgroups,
fewer than the H100 NVL's 132 SMs or the L40S's 142. Weight gradients
(`x.t() @ y`: a small result summed over many rows) used a fraction of
either GPU for the whole of k: a 1024 x 1024 result is 32 tiles.

## What changed

- **The GPU's size.** vkcore's `Capabilities::compute_units` holds the SM
  or compute-unit count, read as llama.cpp's Vulkan backend reads it:
  - VK_NV_shader_sm_builtins (`shaderSMCount`: 142 on the L40S);
  - VK_AMD_shader_core_properties2, or VK_AMD_shader_core_properties;
  - VK_ARM_shader_core_builtins;
  - else a table of Intel's discrete GPUs (Xe cores by device ID).

  It is 0 where unknown (the HD 530, lavapipe, Mali without the Arm
  extension), and products then run as before.
  `TORCH_VK_COMPUTE_UNITS` overrides it, and
  `torch_vulkan._native.compute_units()` reads it.
- **Three tiles in gemmcm.slang**, on a second axis, TILE:
  - l: 128 x 256 in 8 subgroups (as before);
  - m: 128 x 128 in 4 subgroups;
  - s: 64 x 128 in 4 subgroups.

  The kernel was already written in terms of BM, BN, WM and WN.
- **Split k.**
  - With the SPLIT spec constant, gemmcm.slang and gemmsg.slang run
    workgroup z = batch * S + s over k in [s kc, (s + 1) kc) only (kc a
    multiple of 256). Each stores its sums, unscaled, to a float32
    workspace (float64 for float64).
  - A new kernel, splitk.slang, adds the S pieces in the order
    s = 0, 1, ..., scales, adds beta * Cin and rounds once.
  - The order is fixed, so results are the same bit for bit from run to
    run. 16-bit results are still rounded once, from float32 sums.
  - The workspace is S x batch x m x n values from PyTorch's caching
    allocator: one extra dispatch, and 2 S m n values of memory traffic.
- **The choice for float16 and bfloat16** (csrc/linalg.cpp `coop_plan`),
  with U the SM count:
  - At least U large tiles: whole rounds of large tiles, unless the medium
    tile's rounds would run fuller by a tenth. At 3072^3 the large tiles'
    third round runs 4 tiles on 142 SMs; the medium tiles' rounds run 81%
    full.
  - Fewer than U large tiles and k >= 2048: split k as llama.cpp does. When
    the tiles fill at most half the SMs, U / tiles pieces (at most 8); when
    they fill up to two thirds, 3 pieces.
  - Otherwise the large tile when it fills two thirds of the SMs (2048^3),
    else the small tile (1024^3: 128 tiles instead of 32).
- **float32 and float64** (gemmsg.slang): split k by the same rule against
  8 workgroups per SM (its 128-thread workgroups), for k >= 1024, at most
  16 pieces. Its tile stays fixed by the subgroup size.
- **Launch order.** gemmcm.slang can launch its tiles in bands of GROUP
  tile rows, column by column within a band, for L2 reuse, as NVIDIA's
  sample and llama.cpp's coopmat2 kernel do. On the L40S it gained 1% for
  the large tile at 8192^3, and 3-5% for the medium one at 4096^3 and
  8192^3. It stays off by default (`TORCH_VK_GEMM_GROUP`) until it is
  measured on the H100, whose L2 is half the L40S's.
- **Tuning switches** for tests and measurements (docs/api.md):
  `TORCH_VK_SPLIT_K=n`, `TORCH_VK_GEMM_TILE=l|m|s`,
  `TORCH_VK_GEMM_GROUP=g`.

## What was measured (L40S, bfloat16)

The sweep ran every tile, with k split 1 to 16 ways and launch bands of 4
to 32 rows. Times are medians of warm calls, in TFLOP/s; float16 follows
the same pattern a few per cent lower.

| shape | l | m | s | best split (l) | chosen |
|---|---|---|---|---|---|
| 1024^3 | 29.3 | 35.6 | **46.4** | - | s |
| 2048^3 | **139** | 132 | 104 | 102 (2 pieces) | l |
| 3072^3 | 127 | **139** | 129 | 115 (2) | m |
| 4096^3 | 158 | **166** | 133 | 119 (2) | l |
| 8192^3 | **165** | 152 | 102 | - | l |
| 1024 x 1024, k = 8192 | 38.6 | 50.7 | 75.6 | **117** (4) | l, 4 |
| 2048 x 2048, k = 16384 | **146** | 140 | 109 | 135 (3) | l |
| 768 x 3072, k = 32768 | 88.5 | 92.8 | 89.8 | **122** (3, 6) | l, 3 |
| 4096 x 4096, k = 65536 | 151 | 146 | 108 | **152** (4) | l |
| 1024 x 1024, k = 262144 | 40.6 | 54.7 | 83.7 | **144** (4) | l, 4 |

The k = ... rows are `x.t() @ y`, a weight gradient's layout.

- **2048^3 is not short of SMs.** Its 128 large tiles are one round on 142
  SMs. Nothing beat that: medium tiles 132, small 104, and two pieces of k
  102. The reduction then reads and writes 40 MB, about 45 us against
  123 us for the whole product.
- The gap between 2048^3 (139) and 8192^3 (165) is therefore per tile: a
  1-round product pays one prologue and epilogue per SM against 16 at
  8192^3. Splitting k only adds to that unless the SMs would otherwise
  idle.
- **4096^3:** the medium tile wins by 5% on the L40S (166 against 158;
  170 with launch bands). The rule keeps the large tile there, because
  both fill their rounds equally (512 / 568 and 1024 / 1136). It would
  need a per-GPU constant to do better.
- **float32** (gemmsg.slang) gains little from splitting: 13.7 to 16.5 at
  768 x 3072, k = 32768, and 14.6 to 15.2 at 1024 x 1024, k = 262144. At
  15-20 TFLOP/s on a GPU whose float32 peak is 91, it is bound by
  something other than occupancy. **float64** on the L40S runs at its
  1.4 TFLOP/s peak with or without splitting.

## Results

`benchmarks/vs_cuda.py`'s matrix products and the sweep's extra shapes,
TFLOP/s. The before column is this build with
`TORCH_VK_GEMM_TILE=l TORCH_VK_SPLIT_K=1`, which runs main's kernel and
launch. These match #59's figures for main on the L40S (131-139 at
2048^3, 159-165 at 8192^3).

L40S, driver 615.71.09:

| workload | before | after | CUDA |
|---|---|---|---|
| float16 2048^3 | 129 | 129 | 203-208 |
| float16 4096^3 | 150 | 150 | 219-222 |
| float16 8192^3 | 157 | 157 | 171-174 |
| bfloat16 2048^3 | 138 | 137 | 220-221 |
| bfloat16 3072^3 | 127 | 141 | 190 |
| bfloat16 4096^3 | 158 | 156 | 235-237 |
| bfloat16 8192^3 | 163 | 162 | 186 |
| bfloat16 1024^3 | 29.3 | 46.6 | 96.7 |
| float16 4096 x 4096, k = 65536 | 146 | 144 | 192 |
| float16 1024 x 1024, k = 262144 | 40.6 | 138 | 180 |
| bfloat16 4096 x 4096, k = 65536 | 147 | 148 | 205-207 |
| bfloat16 1024 x 1024, k = 262144 | 40.6 | 140 | 194-198 |
| bfloat16 1024 x 1024, k = 8192 | 38.6 | 113 | 193 |
| bfloat16 768 x 3072, k = 32768 | 88.5 | 121 | 196 |
| float32 2048^3 | 18.4 | 18.4 | 42.4-42.7 |
| float32 4096 x 4096, k = 65536 | 15.6 | 15.6 | 41.3-41.5 |
| float32 1024 x 1024, k = 262144 | 14.6 | 15.2 | 39.9-40.4 |
| float32 768 x 3072, k = 32768 | 13.7 | 16.5 | 39.9 |

- Weight gradients with a small result run 3-3.5x faster, at 70-75% of
  CUDA's rate (they were at 20%).
- Square products are unchanged where the large tile was already the
  best, and 11% faster at 3072^3.
- H100 NVL: the same sweep and benchmarks are queued; see the pull
  request.

The HD 530 and lavapipe report no SM count, so nothing changes there.
Alternating runs of main and this branch on the HD 530 (GFLOP/s, a @ b,
three runs each): float32 2048^3 94.2-95.6 against 93.7-94.7, float64
26.2-26.5 against 25.7-26.4, float16 86.9-87.3 against 87.2-88.5. Lavapipe's runs were within
its noise of each other.

## Accuracy

Splitting k changes the order of the sums, not their precision. Each piece
sums in float32 (float64) as before, and the pieces' sums are added in
float32 (float64), so the worst-case error falls from k to k / S + S
roundings. 16-bit results are still rounded once.

test_matmul's test_split_k_gemm checks float32, float64 and, where the
device has cooperative matrices, float16 and bfloat16 against the CPU with
the existing tolerances. It covers:
- forced splits of 2, 3 and 5 pieces, and the default choice with
  `TORCH_VK_COMPUTE_UNITS=300`;
- the m and s tiles and banded launches;
- transposed, strided, batched and broadcast operands, transposed results,
  and NaN bias with beta = 0;
- that three runs agree bit for bit.

On the L40S, every shape of the sweep gave the same bits on a second run,
split or not.

## Checks

- tests/python:
  - lavapipe: 1595 passed;
  - HD 530: 1595 passed;
  - L40S, test_matmul.py: 66 passed.
- test_matmul.py with the validation layer on the HD 530 and lavapipe: 64
  passed, nothing reported.
- The strict OpInfo sweep on lavapipe, default and with
  `TORCH_VK_SPLIT_K=3`: 689 ops, all on the GPU.
- PyTorch's test_linalg.py matrix-product tests on lavapipe (`-k "matmul
  or _mm or addmm or bmm or ..."`): 254 passed. The 41 failures are the
  runner's expected ones: torch.compile, int4 packed weights and the 4-bit
  quantisation bound.
- test_matmul_cuda.py is CUDA-only for vk: 1790 skipped. Its 2 failures
  call CUDA directly.
- `cargo clippy --release --all-targets -- -D warnings` is clean, and
  `cargo test --release -p vkcore` passes.
