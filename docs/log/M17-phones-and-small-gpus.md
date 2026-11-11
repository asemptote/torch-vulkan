# Phones and small GPUs (M17)

Aimed at a Mali-G57 (Galaxy A13 5G: Vulkan 1.1, no shaderInt64, no
shaderFloat64, a 32-bit Android userspace), with no CPU fallback on "vk":

- **Vulkan 1.1 devices**: the 1.2 features vkcore needs (buffer device
  address, scalar block layout, timeline semaphores, the memory model, 8/16-bit
  storage) taken from their extensions; kernels are SPIR-V 1.3. A kernel that
  declares a capability the device lacks is refused with an error rather than
  crashing the driver. `TORCH_VK_API=1.1` forces the 1.1 path and
  `TORCH_VK_DISABLE=int64,float64,int16,float16,int8,storage16,storage8,denorm`
  hides features, so every GPU can test the paths of a smaller one.
- **int64 without shaderInt64, on every GPU**: int64 tensors and
  accumulators are two 32-bit words (`_int64.slang`'s `I64`, with C++'s
  wrapping semantics; `_wide.slang` for index data), 64-bit kernel
  parameters travel as word pairs, float32 trigonometric reduction and the
  Philox generator use 32x32->64 products. One code path everywhere, so the
  emulation is what every test run exercises. Only float64/complex128
  kernels still use native 64-bit integers.
- **float64 without shaderFloat64 is a TypeError, as on MPS**: creating a
  float64 or complex128 vk tensor raises ("this GPU has no float64
  arithmetic ... use float32"); ops whose PyTorch implementations make
  float64 temporaries for float32 inputs (quantile, pinv, matrix_rank, zeta
  with a number, empty batch norm, triu/tril_indices, hash_tensor, printing)
  avoid them there (`_nofloat64.py`, `_kernels.py`). (A bit-exact software
  binary64 for the GPU was tried and dropped: 20-100x slower than float32
  and gigabytes of compiler memory per kernel.)
- **The last CPU fallbacks moved to the GPU**: `_standard_gamma`, `poisson`,
  `binomial`, `_sample_dirichlet` (the softmax of log-gamma samples, so small
  shapes do not underflow in float32) and the gamma/Dirichlet reparameterised
  gradients (`sample.slang`, PyTorch's algorithms); `unique` and
  `unique_consecutive` along a dim (rows ranked by batched column sorts);
  bicubic `grid_sample`'s backward; `repeat_interleave` with a tensor of
  repeats; int8/uint8/int16 ops without their own kernels in int32 on the GPU
  (wrapping arithmetic gives the narrow results). In the OpInfo sweep only
  `to_sparse` still fell back (sparse tensors: M18).
- **Matrix products for Mali** (`gemm4.slang`): on Arm GPUs, float32 and
  int32 products (float16/bfloat16 ones compute in float32) of row-major
  operands (16-byte aligned rows, K a multiple of 4) use a kernel without
  shared memory (on Mali it is ordinary cached memory) and with 16-byte
  loads, 2x8 outputs per thread. On the A13 a
  2048x2048 product takes 0.55-0.66 s (26-31 GFLOP/s), against 0.97 s for
  the shared-memory tiles of `gemm.slang` and 0.64 s on the 8 CPU cores.
  Operands in other layouts (`x @ W.t()`'s column-major `W.t()`) or with
  K or N not multiples of 4 are first copied row-major, zero-padded to
  multiples of 4, when the product's work is 16x the copies': `x @ W.t()`
  at 2048^3 takes 0.57 s against 1.2 s in the tiles, 2047^3 0.54 s against
  1.0 s (a dot-product kernel reading both operands along k reached only 14
  GFLOP/s). On the Intel UHD
  620 the float4 kernel is 2.5x slower than the tiles, so it is Arm-only
  (`TORCH_VK_GEMM=float4` or `tiled` overrides).
- **Building on the phone** (`tools/build-termux.sh`): Termux's
  `python-torch` (32-bit ARM), Rust and clang, 15 minutes on the A13; there
  is no slangc for Android, so the SPIR-V compiled on another machine is
  used as is (`VKCORE_PREBUILT_SPIRV`, crates/vkcore/build.rs). The GPU is
  reached through Android's own Vulkan loader (Termux's finds only
  llvmpipe, Mesa's CPU driver, whose 32-bit ARM shaders for reductions and
  softmax die with SIGBUS: an aligned NEON store, `vst1.64 {..}, [rN:128]`,
  to heap memory llvmpipe allocated 8-byte aligned, in Mesa 26.2.4 and
  26.2.6; importing torch_vulkan on Android warns when it gets llvmpipe). On the A13's Mali-G57 the local suite passes but for the
  float64 tests (skipped) and two references the phone's ARM CPU computes
  differently (complex log2 at 0, float16 linspace).
- **Typed errors on Android**: Android's libc++ matches exception types by
  type_info address, and this library (a Rust cdylib, its C++ symbols kept
  local) has its own copies of c10's: a `TypeError` or `IndexError` raised
  here reached Python as `RuntimeError`. On Android the typed checks
  (`TORCH_CHECK_TYPE`/`INDEX`/`VALUE`/`NOT_IMPLEMENTED`, `vk.h`) throw with
  the type_info libtorch_python's handlers use, found with `dlsym`.
- Run the phone's tests with `OMP_NUM_THREADS=1`: Termux's multithreaded CPU
  convolution, the tests' reference, gives a wrong result for about 2% of
  small conv1d calls (repeated calls on the same input differ), which looked
  like a GPU bug.
