# What CI's lavapipe runs, and the hardware check (M48)

CI runs every test on lavapipe, Mesa's CPU implementation of Vulkan. Several
kernels are chosen by what the device has, and lavapipe as CI configured it
had none of it, so those kernels never ran in CI. This note records which,
what lavapipe can be made to cover, and what only hardware can
(RELEASING.md's hardware check, `tools/cluster/hw-check.sh`).

## What CI did not run

lavapipe's subgroup size is its vector width over 32: 8 lanes at
256 bits, its default (`MIN2(max_vector_bits, 256)`, Mesa's
`lp_bld_init_common.c`; CI also set `LP_NATIVE_VECTOR_WIDTH=256`). The suite
on lavapipe at 256 bits (Mesa 26.0.8 on the lab machine; CI's Mesa 25.2.8
gives the same 1958 passed and 20 skipped) with `-rs`:

| test | skipped because |
|---|---|
| test_matmul.py::test_float4_gemm_16bit | no 16-lane subgroups (gemm4s) |
| test_nn.py::test_direct_conv | no 16-lane subgroups (conv4s) |
| test_matmul.py::test_coopmat_16bit_gemm (2) | no float16 / bfloat16 cooperative matrices (gemmcm) |
| test_matmul.py::test_float32_matmul_precision_tiles (6) | no bfloat16 cooperative matrices (bf16parts) |
| test_long_submissions.py (4) | opt-in (minutes of GPU work), and i915 only |
| test_no_float64.py (3) | the device has float64 |
| test_quant.py (3) | (the CPU's check: not a device path) |

More tests passed without taking the path they are about:
test_fp16_arithmetic.py (its kernels need 16 lanes; elsewhere it checks
that the flag changes nothing), test_float4_gemm_kernel (gemm4.slang instead
of gemm4s), test_float32_matmul_precision (without bfloat16 cooperative
matrices it checks that the products equal "highest"'s), test_split_k_gemm
(gemmsg only), and every convolution test (im2col; conv4s is the default
for stride-1 convolutions wherever 16-lane subgroups run). The Intel
stream's bounds (submissions in flight, workgroups per batch: M34) are on
only for Intel GPUs under Linux.

## 16 lanes on lavapipe

`LP_NATIVE_VECTOR_WIDTH=512` gives 16-lane subgroups (min = max = 16, with
size control and full subgroups) on any x86 CPU: on an AVX2 CPU LLVM splits
the 512-bit vectors into pairs of 256-bit instructions. On an AVX-512 CPU it
emits AVX-512, which is what faulted on some CI runners. The width alone
does not keep AVX-512 out of the code: Mesa passes the CPU's features to
LLVM's JIT whatever the width (`lp_build_fill_mattrs`). Mesa's
`GALLIUM_OVERRIDE_CPU_CAPS=avx` clears AVX-512 from those features
(`check_cpu_caps_override`, `-avx512f` and the rest). A float32 matmul's
JIT code on a Xeon 6730P (AVX-512), lavapipe in Mesa 25.2.7, with
`GALLIVM_DEBUG=asm`:

| settings | zmm lines | ymm lines | AVX-512 mask registers |
|---|---|---|---|
| width 256 | 0 | 927 | 60 |
| width 512 | 3643 | 250 | 458 |
| width 512, `GALLIUM_OVERRIDE_CPU_CAPS=avx` | 0 | 13431 | 0 |
| width 256, `GALLIUM_OVERRIDE_CPU_CAPS=avx` | 0 | 3393 | 0 |

So `LP_NATIVE_VECTOR_WIDTH=256` was lavapipe's default already, and its
code still used AVX-512VL's masks; CI now sets `GALLIUM_OVERRIDE_CPU_CAPS=avx`
(and the release workflow's lavapipe job too), and a second pytest pass at
512 bits runs the 16-lane kernels: gemm4s, conv4s for every stride-1
convolution, float16 arithmetic, and gemmsg at 16 lanes.

Mesa 26.0.8's lavapipe (LLVM 21, the lab machine) aborts ("stack smashing
detected") at 512 bits in float64 division, log, sin and pow, so complex128
ops too; Mesa 25.2.7 (LLVM 21) does not. The second pass therefore runs
with `TORCH_VK_DISABLE=float64`, which is also Mali's case (no float64), a
configuration CI did not run before: it found test_float8_e8m0fnu making a
float64 vk tensor where there are none (fixed). It also sets the Intel
stream's bounds (`TORCH_VK_MAX_IN_FLIGHT=2`,
`TORCH_VK_MAX_BATCH_GROUPS=8192`).

## What only hardware runs

- **Cooperative matrices.** gemmcm.slang needs 16 x 16 x 16 float16 (or
  bfloat16) products into float32 at subgroup scope, 32-lane subgroups and
  38 KiB of shared memory. lavapipe has VK_KHR_cooperative_matrix since Mesa
  26.0 (not in CI's 25.2.8), with 8 x 8 x 8 shapes only (float16, uint8,
  int8; no bfloat16: `lvp_device.c` hard-codes `MSize = NSize = KSize =
  8`) and 32 KiB of shared memory, whatever the width (1024 bits does
  report 32-lane subgroups); SwiftShader has no cooperative matrices. So gemmcm, the float32
  precisions by bfloat16 parts and gemmcm's split-k tiles run only on
  NVIDIA (and other) GPUs with those shapes.
- **i915's request watchdog**: only the kernel driver cancels work, so
  test_cancellation_is_reported needs an Intel GPU under Linux; the long
  submissions take minutes even there.

`tools/cluster/hw-check.sh` runs these before a release, on the lab's Intel
GPU and an L40S, and checks that the kernels each must run are the ones
that do (the tests of a missing feature skip, so a driver that stopped
exposing one would pass unnoticed).

## Cost

In CI (this change's first run), the 8-lane pass: 1958 passed, 20 skipped
in 241 s; the 16-lane pass without float64: 1472 passed, 495 skipped (480
of them float64) in 102 s; the whole check 13 minutes instead of about 10.
On the lab machine (4 lavapipe threads) the same two passes took 285 s and
222 s.
