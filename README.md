# torch-vulkan

A Vulkan compute backend for PyTorch aimed at **scientific** workloads
(float64, complex, FFT, linear algebra), not just neural networks.

Status: **M18** done (sparse tensors; M17: phone GPUs): device
`"vk"` is a native PyTorch backend (C++ on PrivateUse1). Element-wise ops (float32, float64, complex64,
complex128: arithmetic, comparisons, logical ops, the transcendental and
special functions, activations, clamp, where/masked_fill, lerp/addcmul/addcdiv),
copies, casts and fills for every dtype, indexing (advanced indexing,
index_put_, gather, scatter, index_select, take/put_, masked_select, nonzero),
cat, flip, roll, padding, arange/linspace/eye, sorting and order statistics
(sort, topk, kthvalue, median, mode, unique, searchsorted, cummax/cummin),
accumulation into repeated indices (index_add, scatter_add/scatter_reduce,
index_put_ with accumulate, bincount, histc), random numbers (rand, randn,
randint, bernoulli, multinomial, randperm, dropout, ...), neural-network ops
(convolution, pooling, normalization, softmax, losses, upsampling, grid
sampling, embeddings, attention, with their backward passes), reductions (sum,
mean, prod, max/min, argmax, var/std, norms, any/all), scans (cumsum,
cumprod), matrix products (mm, bmm, matmul, addmm, baddbmm, addmv, dot,
nn.Linear), FFTs (all of `torch.fft`, any length) and dense linear algebra
(solve, inv, det, slogdet, LU, Cholesky, QR, eigh, SVD, least squares,
matrix_exp, triangular solves, general eig, LDL, real and complex, any size
and batch, no LAPACK) and the special functions (Bessel, incomplete gamma,
zeta/polygamma, erfcx, orthogonal polynomials) run on the GPU, float16 and
bfloat16 included (computed in float32, rounded once), and 8/16-bit integers
(in int32), and sparse COO and CSR tensors (M18); every other op runs on the
CPU through PyTorch's boxed fallback. In the OpInfo sweep
(`tools/opinfo_coverage.py`) every op runs fully on the GPU and matches the
CPU. Tested on an Intel UHD 620 with the validation layer clean, on lavapipe
and NVIDIA GPUs (PyTorch's own tests, M15), and on a phone's Mali-G57
(M17).

```python
import torch, torch_vulkan

torch.manual_seed(0)            # seeds the "vk" generator too
x = torch.randn(1_000_000, dtype=torch.float64, device="vk")  # Philox on the GPU
y = (x * 2 + 1).relu()          # three Vulkan kernels, batched
print(y[:3], y.sum())           # sum is a Vulkan reduction
torch_vulkan.fallback_counts()  # ops that ran on the CPU instead, if any
```

## Phones and small GPUs (M17)

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
  shapes do not underflow in float32) and the gamma/Dirichlet reparameterized
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
  llvmpipe). On the A13's Mali-G57 the local suite passes but for the
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

## Sparse tensors (M18)

Sparse COO, CSR, CSC, BSR and BSC tensors on "vk"
(`crates/pyvk/csrc/sparse.cpp`): their indices and values are ordinary vk
tensors, so PyTorch's device-generic sparse code serves as is on
`SparsePrivateUse1` / `SparseCsrPrivateUse1` (construction, the getters,
copies, the zero-preserving unary functions, norms, cat, permute), and the
parts PyTorch writes per device are tensor ops here, running as Vulkan
kernels:

- coalescing (one int64 key per entry, a stable sort, runs summed in index
  order), and PyTorch's `flatten_indices` stub (its coalesced-constructor
  check) registered for vk;
- every conversion between dense and the five layouts, batched and hybrid,
  with PyTorch's structure: from a sparse source the specified elements
  (explicit zeros included) decide the entries and blocks, from a dense one
  the nonzeros; the conversions PyTorch refuses fail with its errors;
- sparse + sparse and dense + sparse (coalesced operands give coalesced
  sums), products with scalars and with dense and sparse tensors,
  broadcasting as PyTorch's (a dense operand's batch dims become sparse
  dims), `sparse_mask` and its gradient projection;
- `mm`/`addmm`/`addmv`/`bmm`/`hspmm`/`torch.sparse.mm` for any mix of dense
  and compressed operands, with autograd and the dense ops' errors;
  `sampled_addmm`, batched and broadcast;
- sparse `softmax`/`log_softmax` and their gradients (over the specified
  elements), the gradient of sparse `sum`, sum and prod of compressed
  tensors over dims, `triangular_solve` with a compressed matrix.

PyTorch's test_sparse.py and test_sparse_csr.py pass on vk but for tests
whose expectations name other devices (printing's per-device files,
error-message patterns for CPU/CUDA/MPS, products the CPU refuses), listed
with their reasons in `tools/pytorch_tests.py`. They run locally, not in
the cluster's full runs (their slowest gradchecks take minutes each).

Not yet (M18): the sparse-sparse product and the compressed reductions as
sparse kernels (they go through dense operands, so memory grows with the
dense size), and per-op overhead on tiny sparse tensors (each op is a few
kernel launches: about 10x the CPU's time at a handful of elements).

## Dedicated kernels (M16)

Ops that ran as chains of generic kernels (PyTorch's decompositions), chosen
by profiling (`benchmarks/composed.py`, which counts each op's launches and
CPU fallbacks) and by the slowest tests of the full runs on lavapipe:

- softmax / log_softmax and layer norm, forward and backward: one workgroup
  per row (`softmax.slang`, `layernorm.slang`);
- batch norm and group norm, forward and backward: per-channel (or per-group)
  moments and gradient sums, then one element-wise pass (`normgrad.slang`);
- max / average pooling, linear upsampling (a gather per dim for the
  backward) and grid sampling, forward and backward (`pool.slang`,
  `resample.slang`, `gridsample.slang`);
- nll_loss (so cross_entropy) forward and backward (`loss.slang`);
- embedding's backward and grid sampling's grad_input as rows summed by key:
  one sort of the keys, then whole rows folded in index order, as the CPU's
  loops (`segreduce.slang`'s row mode);
- the activations' backward (gelu, silu, mish, hardswish, hardsigmoid,
  log_sigmoid, sigmoid, tanh, softplus, elu/selu/celu, hardtanh, leaky_relu,
  logit) as one binary kernel each;
- reductions and arg-reductions of a few elements per output: one thread per
  output instead of a workgroup;
- the one-sided Jacobi SVD: column-major storage, and no rotations for
  column pairs already orthogonal to the sweep's tolerance or for columns
  negligible beside |A| (as LAPACK's gesvj), so rank-deficient matrices
  converge in a third of the sweeps.

Not done: rms_norm stays composed (its fused op `_fused_rms_norm` has no CPU
backward kernel, and PyTorch only takes that path on CUDA-like devices);
the smooth-L1/Huber losses' backward still take a few generic kernels;
convolution and attention are matrix products already.

## PyTorch's own tests (M15)

`tools/pytorch_tests.py` runs PyTorch's device-generic test files
(`test_ops.py`, `test_torch.py`, `test_binary_ufuncs.py`,
`test_unary_ufuncs.py`, `test_nn.py`, `test_linalg.py`, `test_reductions.py`,
`test_indexing.py`, `test_tensor_creation_ops.py`, `test_shape_ops.py`,
`test_type_promotion.py`, `test_sort_and_select.py`, `test_view_ops.py`) on
`"vk"`, in shards, from a sparse checkout of PyTorch's `test/` at the
installed torch's commit. Each test that fails on `"vk"` is run again on the
CPU: failures shared with the CPU are not ours. Known, explained differences
are listed in its `EXPECTED` table with the reason for each (sparse and
quantized tensors are out of scope; some tests hard-code CUDA's or the CPU's
behaviour). `tools/cluster/` runs the shards as a PBS job array on lavapipe
(Mesa's CPU Vulkan driver), and optionally with GPU-assisted validation
(bounds-checked buffer accesses).

What the tests asked of `"vk"`, beyond correct values:

- The CPU's argument checks and error messages, in C++ kernels and, for
  composed kernels and PyTorch's decompositions, in `python/torch_vulkan/
  _checks.py`; the dtypes the CPU accepts (`_dtypes.py`); out= semantics
  (resizing and its warning, exact or castable dtypes, out= on another
  device); memory-overlap errors; deterministic-algorithm alerts;
  copy-on-write inputs left unmaterialized; output strides as PyTorch's
  meta kernels predict them.
- Results that do not depend on the driver. float32 math no longer uses the
  GPU's transcendental built-ins, whose accuracy varies (lavapipe's asin is
  off by 4e-4, its tanh near 0 by 6%): `_math32.slang` has Cephes'
  algorithms for exp, log, asin, atan, tanh, sinh, cosh, and pow in
  double-float arithmetic. Shaders are compiled with `-fp-mode precise`
  (every float operation NoContraction), since Mesa reassociates
  expressions otherwise (Cody-Waite's reduction, error-free sums). gemm
  keeps fused multiply-adds.
- No assumptions about how a workgroup executes: lavapipe runs one in
  SIMD-width chunks, so a missing barrier (Jacobi eigh read entries other
  threads had already rotated) gives wrong results there and right ones on a
  GPU; and it performs loads on paths a lane does not take, so indices are
  kept in range even where their value is unused.
- Correct results on NVIDIA's driver, which the full runs now also use (an
  H100 job beside the lavapipe shards). It computes SPIR-V's signed
  remainder as an unsigned one when an operand is negative (integer
  remainder, fmod and floor division use a - (a / b) b instead); turns
  compare-and-select into min/max, which drops NaN (hardtanh tests NaN
  first); divides float32 as a * (1 / b), and 1 / b is a subnormal it flushes
  for |b| > 2^126 (m_div quarters both first); and drops a NaN's sign and
  payload when widening float32 to float64 or negating float64 (casts widen
  NaN bits as x86 does; copysign works on the bits). Half-precision
  reductions run on the GPU in float32: on a discrete GPU the CPU fallback
  copies the whole tensor each time.
- Memory a long run can live with. The pool returns cached blocks to the
  driver before growing when they would cover the new one (a blocked LU's
  shrinking trailing matrices otherwise stay reserved at every size), and
  complex matrix products accumulate into the output's real and imaginary
  parts instead of temporaries.

Every failure left in these files, on lavapipe and on the H100, also fails
on the CPU or has an `EXPECTED` entry giving the reason: out-of-scope layouts
(sparse, quantized, int4), CUDA-specific expectations, tolerances tuned to
one seed or device, random numbers that are not the CPU's, ill-conditioned
float32 comparisons, float32 subnormals on devices that flush them (the H100
cannot keep them in shaders), and tensors of 2^31 or more elements.

## Complex completion and the remaining ops (M14)

Coverage is measured, not listed: `tools/opinfo_coverage.py` runs PyTorch's
OpInfo samples (float32/float64/int64/bool/complex64) and reports per op
whether it ran without CPU fallbacks and matched the CPU (commit messages
record the numbers at each milestone).

- Complex: transcendental functions (`shaders/_cmath.slang`: exp/log/sqrt,
  trig and hyperbolic functions and their inverses, log1p/expm1, pow; the
  usual principal branches), matrix products (4 real GEMMs on the real and
  imaginary parts), reductions (sum/mean through the real view; prod and
  cumprod as log-depth trees of complex products; var/std/norms through
  |z|), accumulation (index_add, scatter_add, index_put/put with accumulate
  on the parts), logaddexp/logcumsumexp, lerp/addcmul/addcdiv, masked_fill,
  complex32 conversions.
- Complex linear algebra: `factor.slang` has complex variants following
  LAPACK's z* routines (izamax pivots, real Cholesky diagonals, zlarfg
  reflectors, Jacobi rotations that first remove the phase of the pivot
  entry), so LU, Cholesky, QR, eigh, SVD, lstsq and their relatives accept
  complex matrices with LAPACK's outputs.
- General `eig`/`eigvals` (`shaders/eig.slang`): Householder reduction to
  Hessenberg form, single-shift QR with Givens rotations to the complex Schur
  form, eigenvectors by back substitution, normalized as LAPACK's geev
  (unit norm, largest component real). Eigenvalues come in no particular
  order (as documented for `torch.linalg.eig`).
- `ldl_factor`/`ldl_solve`: Bunch-Kaufman as LAPACK's sytf2/hetf2 and
  sytrs/hetrs (same pivots and factors as the CPU).
- Special functions (`shaders/special.slang`, `_special.slang`): a
  mechanical port of PyTorch's own CPU implementations
  (`tools/port_special.py` regenerates it from the installed torch), in five
  modules (a single float64 module is too large for some drivers).
- float32 `sin`/`cos`/`tan` use Cody-Waite argument reduction and
  polynomials: the GPU built-ins are only accurate to ~1e-5 for moderate
  arguments, enough to break the Bessel functions' agreement with the CPU.
  It costs ~10-25% on sin/cos-heavy kernels (~2 ms per 16M elements).
- Integer and bool variants (rounding ops, fmax/fmin, heaviside, hardtanh,
  threshold, clamp on bools, max/min over bools, integer convolution and
  average pooling, int32 indices for gather/scatter) and composite kernels
  for `hash_tensor`, `frexp`, `histogram`/`histogramdd`, `segment_reduce`,
  CTC loss (forward and backward), `matrix_sqrth`, `polar`, `addbmm`.

## Linear algebra on the GPU (M13)

No LAPACK anywhere: every size and batch of float32/float64 matrices runs on
the GPU (`csrc/dense.cpp`, `shaders/factor.slang`).

- Blocked LU (rectangular too), Cholesky and Householder QR: the column-by-
  column steps LAPACK would do (panel factorization with partial pivoting
  and whole-row interchanges, Cholesky diagonal blocks, Householder panels
  and the compact-WY `T` factor, inverses of small triangular blocks, row
  interchanges) run in `factor.slang` with one workgroup per matrix; the
  trailing updates are batched GEMMs. Pivots, Householder signs and
  reflector storage follow LAPACK, so `lu_factor`, `geqrf`, `qr`,
  `householder_product` and `ormqr` match the CPU's outputs.
- `eigh`/`eigvalsh`: cyclic Jacobi with round-robin ordering (all disjoint
  index pairs rotated in parallel), converged to working precision.
- `svd`/`svdvals`: one-sided (Hestenes) Jacobi; left vectors of zero
  singular values and `full_matrices` are completed with Householder QR.
- `lstsq` (minimum-norm solution through the SVD, as LAPACK's gelsy/gelsd
  return), `matrix_exp` (Pade 13 with scaling and squaring; float32
  computed in float64), `cholesky_solve`, `cholesky_inverse`, `lu_unpack`,
  `linalg.lu`, right-side and batched triangular solves.
- `TORCH_VK_LINALG=cpu` still sends these ops to LAPACK.
- Eigenvectors and singular vectors are unique only up to sign: tests check
  reconstruction and orthonormality, the OpInfo sweep compares magnitudes.

## Neural-network ops (M12)

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
  transposes), grid sampling backward and 3-D forward, fractional max
  pooling, `cdist`/`pdist`, `rrelu` noise, `embedding_bag`.
- 8- and 16-bit types (uint8, int8, int16, float16, bfloat16) now compute in
  element-wise ops, in int32/float32 and cast back as on the CPU.

OpInfo sweep: 446 of 689 ops fully on the GPU (371 after M11), no
mismatches.

## Random numbers (M11)

- A Philox4x32-10 generator per device (`csrc/random.cpp`), registered with
  PyTorch: `torch.manual_seed`, `torch.Generator(device="vk")`,
  `torch.vk.get_rng_state`/`set_rng_state` and `torch.random.fork_rng`
  work as for CUDA.
- `shaders/rng.slang` draws element i's bits from counter (i, offset) under
  the seed, so results depend only on the seed and the sequence of calls,
  not on how the work is split across GPU threads.
- On the GPU: `uniform_` (`rand`), `normal_` (`randn`, Box-Muller, also with
  tensor mean/std), `random_` (`randint`, every integer and float dtype),
  `exponential_`, `geometric_`, `cauchy_`, `log_normal_`, `bernoulli`
  (scalar or tensor probabilities), `multinomial` (with replacement by
  inverting the running sum, without by Gumbel top-k), `randperm` (sorting
  random keys) and dropout.
- The numbers differ from the CPU's for the same seed (the CPU uses a
  Mersenne Twister), as CUDA's do; tests check reproducibility, ranges and
  distribution statistics instead. The OpInfo sweep does not compare samples
  whose CPU result depends on the seed (dropout inside attention, fractional
  pooling); it still records whether they ran on the GPU.

OpInfo sweep: 371 of 689 ops fully on the GPU (346 after M10; the sweep now
lists 689 OpInfo entries), no mismatches.

## Sorting, searching and statistics (M10)

- **Sort** (`shaders/sort_keys.slang`, `shaders/sort.slang`): values become
  unsigned keys in PyTorch's order (NaN largest, -0 equal to +0, inverted
  for descending), then a bitonic sort of (key, index) pairs runs over all
  rows at once, in shared memory for 1024-element blocks. Breaking ties by
  index makes the result exactly the stable sort, for every real dtype
  including the 8- and 16-bit ones.
- Built on it: `sort`, `argsort`, `msort`, `topk`, `kthvalue`, `median`,
  `nanmedian`, `mode` (`shaders/rowscan.slang` finds the longest run),
  `unique` (with inverse and counts) and `unique_consecutive`.
- `searchsorted` and `bucketize` (`shaders/searchsorted.slang`, a binary
  search per value with PyTorch's comparisons), `cummax`, `cummin` and
  `logcumsumexp` (sequential per row, matching the CPU's NaN and tie rules).
- **Accumulation into repeated indices** without atomics (this GPU has no
  64-bit or float atomics): source elements are sorted by destination and
  `shaders/segreduce.slang` folds each run in source order, the same order as
  the CPU's loop, so float sums agree bit for bit. Used by `index_add`,
  `index_reduce`, `scatter_add`, `scatter_reduce` (sum/prod/mean/amax/amin,
  with or without self), `scatter(reduce=...)`, `index_put_`/`put_` with
  `accumulate=True`, `bincount` and `histc`.
- Small reductions from existing kernels: `aminmax`, `var_mean`,
  `std_mean`, `nansum`, `trace`.
- Which of several equal elements `topk`, `kthvalue`, `median` and `mode`
  report is unspecified in PyTorch (the CPU uses unstable partial sorts); we
  return the stable-order one, and tests check values plus that the indices
  select them.

| sort, float32 | vk (UHD 620) | CPU (stable, 4 threads) |
|---|---|---|
| 1 x 2^20 | 78 ms | 533 ms |
| 64 x 2^16 | 145 ms | 546 ms |
| 4096 x 1000 | 61 ms | 326 ms |

OpInfo sweep: 346 of 659 ops fully on the GPU (317 after M9), no mismatches.

## Data movement and indexing (M9)

- **Every dtype moves on the GPU.** Copies, `clone`, `contiguous`, fills and
  all data movement use `shaders/move.slang`, which works on elements by size
  (1, 2, 4, 8 or 16 bytes), so any dtype is covered. Casts now also cover
  uint8, int8, int16, float16 and bfloat16 (where the device has 8/16-bit
  storage; this one does). Float-to-small-integer casts wrap like the CPU's
  (-3.0 -> 253), and float16 conversions round to nearest even explicitly
  (Vulkan leaves that rounding to the driver).
- **Indexing** is two steps: `shaders/index.slang` turns index tensors into
  element offsets (wrapping negative indices where PyTorch does: `index`,
  `index_put_`, `take`, `put_`, `index_fill_`; rejecting them where it
  rejects them: `gather`, `index_select`, `scatter`, `index_copy`), then a
  gather or scatter in `move.slang` reads or writes the source restrided to
  the result's shape. Advanced indexing (integer and boolean, adjacent or not)
  follows PyTorch's `make_info`; boolean masks go through the GPU `nonzero`.
- **Out-of-range indices** are reported like CUDA's device asserts: the kernel
  sets an error word, and the next synchronize (`.cpu()`, `.item()`,
  `torch_vulkan.synchronize()`) raises `IndexError`. Queued work is not
  stopped, and the result of the bad op is undefined.
- **nonzero** counts with a GPU running sum (one sync for the result size, as
  on CUDA) and writes coordinates with `shaders/nonzero.slang`;
  `masked_select`, `masked_scatter_` and boolean indexing build on it.
- **cat, flip, roll, reflection/replication padding (1-3d)**, `channel_shuffle`,
  `equal`, `isin` (by broadcast comparison, up to 2^24 pairs) and
  **arange, linspace, logspace, eye** (`shaders/iota.slang`, matching
  PyTorch's formulas and float16 rounding) run on the GPU.
- Still on the CPU: ops that accumulate into repeated indices (`index_put_`
  with `accumulate=True`, `index_add`, `scatter_add`, `scatter_reduce`,
  `put_(accumulate=True)`, `bincount`, `histc`). This GPU has no 64-bit or
  float atomics, so these will use sorting (M10).

OpInfo sweep: 317 of 659 ops fully on the GPU (272 after M8), no mismatches.

## Element-wise ops (M8)

Every dense element-wise aten op on float32/float64/int32/int64/bool now has a
Vulkan kernel: the remaining transcendental and special functions (`asin` ...
`atanh`, `log10`, `log1p`, `expm1`, `erf`, `erfc`, `erfinv`, `lgamma`,
`digamma`, `sinc`, `frac`), binary math (`pow`, `atan2`, `fmod`, `remainder`,
`floor_divide`, `logaddexp`, `xlogy`, `hypot`, `copysign`, `nextafter`,
`heaviside`, `fmax`/`fmin`, shifts, `gcd`/`lcm`), activations (`gelu`, `silu`,
`mish`, `softplus`, `elu`/`selu`, `logsigmoid`, `logit`, `hardtanh`,
`hardswish`, `leaky_relu`, shrinks, `threshold`), `clamp` (scalar and tensor
bounds), predicates (`isnan`, `isinf`, `signbit`, ...), logical ops, `where`,
`masked_fill`, `lerp`, `addcmul`, `addcdiv`, `nan_to_num`, `round(decimals)`.

Math precision, by dtype:

- **float32** uses the GPU's GLSL built-ins (`exp`, `log`, `sin`, `pow`,
  `atan2`, ...), so its accuracy follows Vulkan's precision rules, plus
  `shaders/_math32.slang` for functions with no built-in (`erf`, `erfinv`,
  `lgamma`, `digamma`, `log1p`, `expm1`, `sinpi`) and short wrappers where a
  built-in mishandles edge cases or is too inaccurate on Intel (`tanh`
  overflow, `sinh`/`asinh` cancellation, `acos` and `tan` accuracy, C99
  special values of `pow`/`atan2`). float32 kernels contain no float64 code,
  so they run on GPUs without `shaderFloat64` (a Rust test checks the SPIR-V).
- **float64** uses the software library in `_math64.slang` and
  `_math64_ext.slang` (SPIR-V has no 64-bit transcendentals), agreeing with
  the CPU to a few ULP.
- `fmod`/`remainder`/`floor_divide` are exact for both.

`shaders/_fmath.slang` gives both under one set of names (`m_exp`, `m_pow`,
...), so `unary.slang` and `binary.slang` are each one kernel per dtype.

Infrastructure cleanup in the same step:

- Operation codes live in `shaders/opcodes.txt`; the build generates
  `opcodes.slang` for the shaders and `opcodes.h` (`vk::op::UNARY_EXP`, ...)
  for C++, so the two cannot disagree.
- One strided launcher (`csrc/launch.cpp` + `csrc/strided.h`) packs scalar
  parameters per kernel dtype (int64 / double / float) for the three- and
  four-operand layouts; push-constant sizes and strides are 4-vectors, since
  Intel's driver misreads dynamically indexed push-constant arrays.
- Kernels are compiled with scalar block layout and denormal preservation
  (stripped at load time on devices that cannot preserve denormals); loading
  a kernel that needs float64/int64/int8 on a device without it is a clear
  error. Shader variants are rebuilt only when their inputs change.
- The C++ static library is linked whole, so every file's kernel
  registrations are kept without link-anchor functions.
- `opinfo_coverage.py` runs OpInfo samples in parallel worker processes:
  272 of 659 ops now run fully on the GPU (170 at M7).

## Dense linear algebra (M7)

- `csrc/dense.cpp`: blocked right-looking LU with partial pivoting and
  Cholesky, panels of 64 columns. Each panel / diagonal block is factored by
  LAPACK on the CPU (memory is shared: a copy plus a sync); the O(n³)
  trailing updates run on the GPU as GEMMs. Triangular solves apply inverses
  of the 64×64 diagonal blocks (computed on the CPU in float64) with GEMMs.
  Row interchanges are applied by a gather/scatter kernel
  (`shaders/rows.slang`) to the rows that move.
- Ops: `linalg.lu_factor(_ex)`, `linalg.solve(_ex)` (vector or matrix right-hand
  side), `linalg.inv(_ex)`, `linalg.det`, `linalg.slogdet`, `linalg.lu_solve`,
  `linalg.cholesky(_ex)` (lower/upper), `linalg.solve_triangular`, plus
  `tril`/`triu` kernels (`shaders/triangle.slang`, any dtype, batched).
  Pivots match LAPACK's exactly; `info` and `check_errors` behave as on the
  CPU (singular / not positive-definite matrices raise `LinAlgError`).
- The GPU path takes single float32/float64 matrices with n ≥ 128; smaller or
  batched inputs, right-side solves, QR, eigh and SVD run on the CPU.
  `TORCH_VK_LINALG=cpu` sends everything to LAPACK.

Measured (`python benchmarks/linalg.py`, UHD 620 vs 4-thread CPU with MKL):

| n = 2048 | vk float32 | CPU | vk float64 | CPU |
|---|---|---|---|---|
| `solve` (8 right-hand sides) | 494 ms | 112 ms | 636 ms | 216 ms |
| `cholesky` | 283 ms | 64 ms | 458 ms | 101 ms |
| `inv` | 587 ms | 305 ms | 1640 ms | 572 ms |

Correct, but slower than MKL on this machine: the trailing updates use the
M5 GEMM (itself slower than MKL here), and each panel synchronizes, leaving
the GPU idle while the CPU factors it. Look-ahead (factoring panel k+1 while
the GPU updates the rest) and a faster GEMM are the next steps; on an
integrated GPU like this one, `TORCH_VK_LINALG=cpu` is the faster choice.

## Complex numbers and FFT (M6)

- complex64 / complex128 tensors (stored as float2 / double2): casts to and
  from every real dtype, fills, `+ - * /` (with complex scalars and real
  `alpha`; division uses Smith's algorithm), `neg`, `abs` (overflow-safe),
  `conj` / `conj_physical`, and the views `real`, `imag`, `view_as_real`,
  `view_as_complex` (they alias: `x.real += 1` updates `x`).
- Lazy conjugate/negative bits: `_copy_from` passes through PyTorch's
  Conjugate/Negative layers and applies the bits itself (otherwise
  materializing them recursed forever). The CPU fallback now refuses view
  ops (`error_on_views`) instead of silently returning copies.
- `shaders/fft.slang` + `csrc/fft.cpp`: `_fft_c2c`, `_fft_r2c`, `_fft_c2r`,
  which every `torch.fft` function reduces to. Power-of-two lengths use
  radix-2 Stockham passes (rows up to 2048 complex64 / 1024 complex128 run
  all passes in shared memory in one dispatch); other lengths use
  Bluestein's algorithm. Twiddle and chirp tables are computed on the host in
  float64 (the GPU's float32 sin/cos would be far too inaccurate) and cached.
  r2c transforms as complex and keeps n/2+1 bins; c2r rebuilds the Hermitian
  spectrum and keeps the real part. Error vs PyTorch's CPU FFT is a few ULP x
  log2(n); a 65536-point complex128 FFT agrees to < 1e-11.

Measured (`python benchmarks/fft.py`, UHD 620 vs 4-thread CPU):

| | vk complex64 | CPU | vk complex128 | CPU |
|---|---|---|---|---|
| fft 2^20 | 33 ms | 14 ms | 49 ms | 25 ms |
| fft2 1024×1024 | 12 ms | 6 ms | 27 ms | 12 ms |
| fft 1,000,003 (prime, Bluestein) | 119 ms | 212 ms | 249 ms | 345 ms |
| rfft 2^22 | 127 ms | 62 ms | 273 ms | 104 ms |

The FFTs are memory-bound radix-2 passes plus transposes between dims; on
this integrated GPU only long prime-length transforms beat the CPU so far.
Radix-4/8 passes, a shared-memory four-step FFT for long rows, and a real
r2c (half-length complex) path are the next steps.

## Matrix products (M5)

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

## Reductions, scans, float64 math (M4)

- `shaders/reduce.slang` + `csrc/reduce.cpp`: tree reduction over any set of
  dims (up to 4 non-mergeable kept and reduced dims each). Large reductions
  are split into chunks across workgroups and the partials reduced again, so
  float32 sums stay as accurate as the CPU's. When the reduced dims are
  strided and outputs are adjacent (column reductions), each thread owns one
  output so reads are coalesced. Integer and bool inputs accumulate in int64,
  as in PyTorch. Ops: `sum`, `mean`, `prod`, `amax`/`amin`, `max`/`min`
  (NaN-propagating), `var`/`std` (two passes, any correction),
  `linalg.vector_norm` (ord 1, 2, ±inf), `any`/`all`.
- `shaders/argreduce.slang`: (value, index) reduction for `argmax`/`argmin`
  and `max`/`min` along a dim; ties and NaNs resolve to the first index, as on
  the CPU.
- `shaders/scan.slang`: `cumsum`/`cumprod` along any dim: per-chunk workgroup
  scans, the chunk totals scanned recursively, then added back.
- `shaders/_math64.slang`: float64 `exp`, `exp2`, `log`, `log2`, `sin`, `cos`,
  `tan`, `tanh`, `sigmoid` (SPIR-V has no 64-bit transcendentals). Measured
  against PyTorch's CPU over 200k inputs each plus special values: exp/exp2 ≤ 1
  ULP, log/log2/sigmoid/tan ≤ 2 ULP, tanh ≤ 3 ULP, sin/cos ≤ 1 ULP of scale.
  sin/cos/tan reduce arguments exactly up to |x| ≈ 1.6e6.
- Kernels preserve denormals (`-denorm-mode-fp32/fp64 preserve`): the UHD 620
  otherwise flushes subnormal float64 results to zero.

Measured (`python benchmarks/elementwise.py`, 16M elements, UHD 620 vs 4-thread CPU):

| | vk float32 | CPU float32 | vk float64 | CPU float64 |
|---|---|---|---|---|
| `sum()` | 2.5 ms | 2.7 ms | 8.7 ms | 4.9 ms |
| `var(dim=1)`, 4096×4096 | 6.3 ms | 93 ms | 17 ms | 88 ms |
| `amax(dim=0)`, 4096×4096 | 4.7 ms | 6.3 ms | 13 ms | 14 ms |
| `cumsum` | 23 ms | 31 ms | 52 ms | 34 ms |

## PyTorch backend (M3)

- `crates/pyvk/csrc/backend.cpp`: a `c10::Allocator` over the Rust memory pool
  (`data_ptr()` is the host-mapped address; the GPU address is derived from the
  allocation context), a device guard, `PrivateUse1HooksInterface`, and aten
  kernels for `empty`/`empty_strided`, views (`view`, `as_strided`,
  `_reshape_alias`), `resize_` (grows storage, keeps contents), `set_`,
  `_copy_from`/`_copy_from_and_resize` (CPU <-> vk and vk <-> vk with
  broadcasting and dtype conversion), `_local_scalar_dense`, `fill_`, `zero_`.
- `crates/pyvk/csrc/ops.cpp`: the `.out` overloads of structured ops (PyTorch's
  generated functional/in-place variants call them): 19 unary ops plus `relu`,
  11 binary ops (scalar operands, `alpha`, scalar-first `sub`), 6 comparisons
  (Tensor and Scalar), all with PyTorch's type promotion. Cases without a
  kernel (e.g. float64 `exp`) run the same overload through
  `at::native::cpu_fallback`, so results stay correct.
- Every other op: a boxed PrivateUse1 fallback to the CPU, counted in
  `torch_vulkan.fallback_counts()` and warned about once per op. Its results
  are written back with `_copy_from_and_resize`, which resizes `out=`
  arguments (the empty-result bug of torch-directml and pytorch_ocl is tested).
- Strided launches merge contiguous dims; tensors with more than 4
  non-mergeable dims loop over the outer dims on the host (one dispatch each).
- `crates/pyvk/src/ffi.rs`: the C ABI from C++ to the Rust runtime (alloc, free,
  dispatch, synchronize); the C++ is compiled by `crates/pyvk/build.rs` against
  the libtorch of the target Python and linked into the same extension module.
- `python/torch_vulkan`: names the backend `vk`, registers `torch.vk`, adds
  `Tensor.vk()`, and exposes `synchronize`, `memory_stats`, `empty_cache`,
  `fallback_counts`.

The M2 prototype (a `__torch_dispatch__` tensor subclass) is replaced; its
kernels and tests carry over.

Kernels are Slang templates compiled per dtype (`// axis T: ...` lines in the
shader; see `build.rs`), with the op chosen by a specialization constant:
`unary`, `binary`, `compare`, `cast` (all 25 pairs of f32/f64/i32/i64/bool;
also strided copies) and `fill`. Operands are addressed as pointer + up to 4
coalesced (size, stride) dims, so broadcasting, transposes and slices need no copies.

Precision: float32 math uses the GPU's built-ins, which Vulkan only requires
to be accurate to ~2^-11 absolute (sin/cos) or a few ULP; tests use those
bounds. float64 uses `_math64.slang` (see M4 and M8).

Measured (`python benchmarks/elementwise.py`, UHD 620 vs 4-thread CPU, torch 2.14.1):

| | vk (M3, C++) | vk (M2, Python) | CPU |
|---|---|---|---|
| per-op overhead, 1024 floats | 9 µs | 136 µs | 7 µs |
| `relu(x*2+1)`, 16M float32 | 27 ms | 29 ms | 27 ms |
| `relu(x*2+1)`, 16M float64 | 37 ms | 40 ms | 51 ms |

Large arrays are memory-bound (~15–22 GB/s on this integrated GPU); per-op
overhead is now close to the CPU's.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| GPU API | Vulkan 1.1+ via [`ash`](https://crates.io/crates/ash) (1.2/1.3 features as extensions) | raw device pointers, explicit memory types, low dispatch overhead; runs on Intel, AMD, NVIDIA, and phone GPUs (Mali) |
| Core language | Rust (`crates/vkcore`) | Memory/queue/pipeline management; exposed through a C ABI so only a thin layer is tied to PyTorch's C++ ABI |
| PyTorch integration | C++ backend on `PrivateUse1` (`crates/pyvk/csrc`), calling the Rust core through a C ABI | PyTorch's backend API is C++-only; only this layer is tied to a PyTorch release |
| Device name | `vk` | PyTorch already has a built-in `vulkan` device type (its old, unmaintained mobile backend) |
| Tensor addressing | `VK_KHR_buffer_device_address`: tensors are `(pointer, offset, sizes, strides)` passed in push constants | Views and strides become pointer arithmetic, as in CUDA; no descriptor sets |
| Memory | Prefer device-local + host-visible + coherent memory | On integrated GPUs (UHD 620, Strix Halo) CPU and GPU share tensors with no copies |
| Kernels | [Slang](https://shader-slang.org) → SPIR-V (`crates/vkcore/shaders`, compiled by `build.rs`) | Generics, pointers, autodiff; can later also target CUDA/Metal |
| FFT | VkFFT (planned) | Complex + float64 FFT on Vulkan |
| Matmul | Own kernels, modeled on llama.cpp's `ggml-vulkan` | No mature Vulkan BLAS exists |

## Verified on this machine (M0)

`cargo run --release --example smoke` on Intel UHD 620, driver 101.2134:

- Vulkan 1.3.215; float64, int64, int16, float16, buffer device address,
  scalar block layout, Vulkan memory model: all supported
- Subgroup size 32 (8–32), 32 KiB shared memory, max workgroups.x 65536,
  max single allocation 7.94 GiB, max buffer 4 GiB
- One 7.94 GiB heap with a device-local + host-visible memory type (unified memory)
- Not supported: int64 atomics, float32/float64 atomic add, cooperative matrix
  (tensor cores) -- reductions must use subgroup ops / shared memory instead
- Kernel `y = x² + 1/3` over 16M float64 values via device-address pointers:
  max error 1.8e-15 vs CPU (FMA contraction), 15 ms GPU vs 22 ms single-thread CPU,
  ~18 GB/s effective bandwidth

## Core runtime (M1)

- **Memory pool** (`memory.rs`): tensors are sub-allocated (256 B aligned,
  best fit, coalescing) from 64 MiB persistently mapped, device-addressable
  blocks; larger requests get their own block. Freed memory is cached and
  reused in stream order, like PyTorch's CUDA allocator; `trim()` returns
  empty blocks to Vulkan, and allocation retries after trimming when out of memory.
- **Kernel cache** (`kernels.rs`): one shared pipeline layout (no descriptor
  sets, 128-byte push-constant block). Pipelines are keyed by
  (kernel, specialization constants); compiled code persists across runs in
  `%LOCALAPPDATA%\torch-vulkan\pipelines.bin` (override with `TORCH_VK_CACHE_DIR`).
- **Stream** (`stream.rs`): in-order like a CUDA stream. Commands are recorded
  with a full memory barrier between them and submitted in batches of up to
  256, tracked by a timeline semaphore; command buffers are recycled.
  The host must `synchronize()` before touching memory that queued work uses.
- **Validation** (`context.rs`): `TORCH_VK_VALIDATE=1` enables the Khronos
  validation layer with synchronization and best-practices checks; every
  message is printed and counted, and the tests fail on any.

Measured on the UHD 620 (`cargo run --release --example dispatch_bench`):
synchronizing after every small kernel costs **486 µs/op**; batching brings it
to **5.6 µs/op** (~87× less).

## Layout

```
Cargo.toml                 workspace
crates/vkcore/             Rust Vulkan core
  src/context.rs           device + queue, capabilities, validation reporting
  src/memory.rs            caching sub-allocator (Pool, Allocation, RangeAllocator)
  src/kernels.rs           pipeline cache + embedded SPIR-V table
  src/stream.rs            batched in-order command stream
  src/lib.rs               Runtime (all of the above) + helpers
  shaders/*.slang          compute kernels (templated per dtype), compiled to SPIR-V by build.rs
  shaders/opcodes.txt      operation codes shared by shaders and C++ (generated headers)
  shaders/_strided*.slang  shared push-constant layouts + strided index math
  shaders/_fmath.slang     float math by dtype: built-ins + _math32 (float32), _math64* (float64)
  shaders/test/            kernels for vkcore's own tests (feature test-kernels)
  tests/runtime.rs         GPU tests (batching, dependencies, reuse, copies, pool, cache)
  examples/smoke.rs        hardware check
  examples/dispatch_bench.rs  per-op overhead, synchronized vs batched
crates/pyvk/               extension module torch_vulkan._native
  csrc/backend.cpp         allocator, device guard, hooks, views, resize, copies, CPU fallback
  csrc/launch.cpp          strided element-wise launches (csrc/strided.h plans dims)
  csrc/ops.cpp             unary, binary and comparison aten kernels
  csrc/elementwise.cpp     activations, clamp, predicates, where, lerp/addcmul, compositions
  csrc/indexing.cpp        indexing, accumulation, nonzero, cat/flip/roll/padding, arange/linspace/eye
  csrc/sorting.cpp         sort and order statistics, unique, searchsorted, cummax/cummin
  csrc/random.cpp          Philox generator and random fills, multinomial, randperm, dropout
  csrc/convolution.cpp     convolution (im2col + GEMM), transposed, backward
  csrc/normalization.cpp   batch norm entry points, attention math path
  csrc/reduce.cpp          reductions, arg-reductions, scans
  csrc/linalg.cpp          matrix products (batched strided GEMM)
  csrc/fft.cpp             FFTs (Stockham + Bluestein, r2c/c2r)
  csrc/dense.cpp           LU, Cholesky, QR, Jacobi eigh/SVD, lstsq, eig, LDL, triangular solves, tril/triu
  csrc/special.cpp         special functions (Bessel, gamma family, zeta, polynomials)
  shaders/factor.slang     factorization panels, Jacobi rotations, LDL (one workgroup per matrix)
  shaders/eig.slang        Hessenberg reduction, shifted QR, Schur eigenvectors
  shaders/special.slang    special functions; _special.slang is generated by tools/port_special.py
  src/ffi.rs               C ABI from C++ to the Rust runtime
  src/lib.rs               Python functions (synchronize, stats, fallback counts)
  build.rs                 compiles csrc/ against the target Python's libtorch
python/torch_vulkan/       backend naming, torch.vk device module, _decomp.py (PyTorch
                           decompositions as kernels), _kernels.py (Python kernels)
tests/python/test_*.py     results on "vk" vs CPU (statistics for random ops)
tools/opinfo_coverage.py   runs PyTorch's OpInfo samples on "vk": GPU / fallback / mismatch per op
tools/port_special.py      regenerates shaders/_special.slang from torch's ATen/native/Math.h
tools/pytorch_tests.py     runs PyTorch's device-generic test files on "vk" (M15), with a CPU re-check
tools/cluster/             runs them on a PBS cluster (full-run.sh), and a dev copy there (dev.sh)
tools/dev/                 pt.py (one test file on "vk", tracebacks), optrace.py (ops differing from
                           the CPU inside a test), seedsweep.py (a test under shifted seeds)
.github/workflows/ci.yml   Linux CI on lavapipe (Mesa's CPU Vulkan) with validation
benchmarks/elementwise.py  per-op overhead, element-wise, reduction and scan throughput vs CPU
benchmarks/matmul.py       matrix multiply GFLOP/s vs CPU
benchmarks/fft.py          FFT timings vs CPU
benchmarks/linalg.py       solve / cholesky / inv timings vs CPU
pyproject.toml             maturin build of the Python package
```

## Building

Requirements: Rust (rustup, MSVC toolchain), Visual Studio Build Tools with C++,
a Vulkan 1.3 driver, and the Vulkan SDK (1.4.363 here: `slangc`, validation
layers, `vkconfig`). `build.rs` finds `slangc` via `SLANGC`, then `VULKAN_SDK`, then PATH.

```powershell
$env:PATH = "$HOME\.cargo\bin;$env:PATH"
cargo run --release --example smoke
```

Tests and examples under validation (synchronization and best-practices checks included):

```powershell
$env:TORCH_VK_VALIDATE = "1"
cargo test --release
cargo run --release --example smoke
```

All 15 Rust tests (8 unit tests, 7 GPU tests) and both examples run with no
validation messages.

Python package (editable install into `.venv`, CPU build of PyTorch):

```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest maturin
.\.venv\Scripts\maturin develop --release --uv   # compiles csrc/ against this venv's torch
$env:TORCH_VK_VALIDATE = "1"; .\.venv\Scripts\python -m pytest -q
```

All Python tests pass with no validation messages. Op coverage:
`python tools/opinfo_coverage.py --jobs 4` (writes `tools/opinfo_coverage.json`).

## Development workflow

Checks, cheapest first; a change goes on to the next only once it passes:

1. Local: `.\.venv\Scripts\python -m pytest -q` (seconds), and the affected
   PyTorch tests with `python tools/dev/pt.py test_linalg.py -k "..."`.
2. Lavapipe, in the dev copy on the cluster (a second checkout with its own
   venv, rebuildable while a full run uses the main one), from the local
   repo root:

   ```sh
   tools/cluster/dev.sh build                  # sync the working tree, rebuild (~1 min for a C++ change)
   tools/cluster/dev.sh run ID...              # test ids through the runner, CPU re-check included
   tools/cluster/dev.sh pt test_nn.py -k ...   # pytest with tracebacks
   tools/cluster/dev.sh rss ID...              # time and peak memory per test
   tools/cluster/dev.sh py script.py           # any script, on lavapipe
   ```

   Lavapipe finds what real GPUs hide: races between a workgroup's SIMD
   chunks, loads on untaken paths, inaccurate built-ins.
3. Full run, once per batch of changes and before every push: on the head
   node, after `tools/cluster/sync.sh ws3 --build -J GATEWAY`,
   `bash tools/cluster/full-run.sh NAME --gpu` (48 lavapipe shards and one
   H100 job, about an hour; see `tools/cluster/README.md`). Push only if no
   test that passed in an earlier run fails.

For a failure, `tools/dev/optrace.py` shows which ops differ from the CPU
inside a test, and `tools/dev/seedsweep.py` whether the CPU passes the same
test under other seeds. A failure that is not a vk bug (a tolerance tuned to
one seed or device, a CUDA-specific expectation, a time or memory limit of
the runner) gets an `EXPECTED` entry in `tools/pytorch_tests.py` with the
evidence as its reason, rather than a workaround.

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. ~~**M2 Python prototype**~~: done; superseded by M3.
3. ~~**M3 native backend**~~: done (see above). Not yet: multiple devices,
   pinned memory. (A GPU random-number generator came with M11; bool binary
   ops and 8/16-bit accumulation run on the GPU since.)
4. ~~**M4 reductions, scans, float64 math**~~: done (see above); the ops
   it left (`logcumsumexp`, `cummax`, `median`, `masked_select`, float64
   `pow`/`atan2`, the arg-reduction thread-per-output mode) came later. Not
   yet: Payne-Hanek reduction for huge sin/cos arguments.
5. ~~**M5 matrix products**~~: done (see above). Not yet: a faster
   subgroup/cooperative-matrix kernel, integer and half-precision GEMM,
   convolutions.
6. ~~**M6 complex numbers + FFT**~~: done (see above), with an own FFT
   rather than VkFFT (VkFFT's Vulkan backend needs the raw device/queue
   handles and runtime glslang). Not yet: faster FFT kernels. (Complex
   reductions, scans, matmul and transcendentals came with M14.)
7. ~~**M7 dense linear algebra**~~: done (see above). Not yet: blocked QR
   (compact-WY updates), batched GPU paths, look-ahead, eigh/SVD on the GPU.
8. ~~**M8 element-wise completeness**~~ and infrastructure cleanup: done (see
   above). Since: float16/bfloat16 arithmetic (in float32, M16), complex
   transcendentals (M14), `polygamma`, Bessel and the other `torch.special`
   functions. Not yet: 8/16-bit integer arithmetic in its own width (it
   computes in int32).

Goal for M9-M14: every dense aten op runs on Vulkan (`TORCH_VK_STRICT=1` with
no fallbacks in the OpInfo sweep).

9. ~~**M9 data movement and indexing**~~: done (see above). (`repeat_interleave`
   with a tensor of repeats and `tril_indices`/`triu_indices` came with M17.)
10. ~~**M10 sorting, searching and statistics**~~: done (see above).
    (`histogram`/`histogramdd` came with M14, `unique(dim=...)` with M17.)
    Not yet: a radix sort for very long
    rows (bitonic is O(n log^2 n)), parallel scans for single long rows in
    cummax/cummin/logcumsumexp.
11. ~~**M11 random numbers**~~: done (see above). (`rrelu`'s noise came
    with M12; `poisson`, `_standard_gamma`, `binomial`, `_sample_dirichlet`
    with M17. `random_` on complex dtypes is an error on the CPU too.)
12. ~~**M12 neural-network ops**~~: done (see above), CTC loss,
    `segment_reduce` and integer convolution included (integer pooling
    without float64 and bicubic `grid_sample`'s backward came with M17,
    native kernels for the hottest composed ops with M16).
13. ~~**M13 linear algebra on the GPU**~~: done (see above). Not yet:
    general (nonsymmetric) `eig`/`eigvals`, `ldl_factor`/`ldl_solve`
    (Bunch-Kaufman pivoting), complex matrices (M14).
14. ~~**M14 complex completion**~~: done (see above). Not yet: the ops
    the sweep still sends to the CPU (see `tools/opinfo_coverage.json`),
    accuracy-faithful complex edge cases (infinities, branch cuts at signed
    zeros), a faster eig (multishift QR, aggressive deflation).
15. ~~**M15 PyTorch's own tests**~~: done (see above). Not yet: tensors of
    2^31 or more elements (32-bit element offsets; lavapipe's buffers stop at
    2 GiB), float32 subnormals on devices that flush them, and long float64
    kernels on Windows GPUs (the float64 zeta test outlasts the Intel UHD
    620's watchdog and loses the device). (lavapipe's slowest tests, the
    SVD-heavy ones and grid_sampler_3d's, got faster in M16.)
16. ~~**M16 dedicated kernels**~~: done (see above). Not yet: rms_norm and
    the smooth-L1/Huber backward as single kernels.
17. ~~**M17 phones and small GPUs**~~: done (see above). Not yet: int64
    Mali products in gemm4 (they use the tiles).
18. ~~**M18 sparse tensors**~~: done (see above). Not yet: sparse kernels
    for the sparse-sparse product and compressed reductions.
Testing throughout: PyTorch OpInfo samples compared against the CPU (`tools/opinfo_coverage.py`).
