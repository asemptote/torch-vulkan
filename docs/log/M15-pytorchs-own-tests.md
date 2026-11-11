# PyTorch's own tests (M15)

`tools/pytorch_tests.py` runs PyTorch's device-generic test files
(`test_ops.py`, `test_torch.py`, `test_binary_ufuncs.py`,
`test_unary_ufuncs.py`, `test_nn.py`, `test_linalg.py`, `test_reductions.py`,
`test_indexing.py`, `test_tensor_creation_ops.py`, `test_shape_ops.py`,
`test_type_promotion.py`, `test_sort_and_select.py`, `test_view_ops.py`) on
`"vk"`, in shards, from a sparse checkout of PyTorch's `test/` at the
installed torch's commit. Each test that fails on `"vk"` is run again on the
CPU: failures shared with the CPU are not ours. Known, explained differences
are listed in its `EXPECTED` table with the reason for each (sparse and
quantised tensors are out of scope; some tests hard-code CUDA's or the CPU's
behaviour). `tools/cluster/` runs the shards as a PBS job array on lavapipe
(Mesa's CPU Vulkan driver), and optionally with GPU-assisted validation
(bounds-checked buffer accesses).

What the tests asked of `"vk"`, beyond correct values:

- The CPU's argument checks and error messages, in C++ kernels and, for
  composed kernels and PyTorch's decompositions, in `python/torch_vulkan/
  _checks.py`; the dtypes the CPU accepts (`_dtypes.py`); out= semantics
  (resizing and its warning, exact or castable dtypes, out= on another
  device); memory-overlap errors; deterministic-algorithm alerts;
  copy-on-write inputs left unmaterialised; output strides as PyTorch's
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
(sparse, quantised, int4), CUDA-specific expectations, tolerances tuned to
one seed or device, random numbers that are not the CPU's, ill-conditioned
float32 comparisons, float32 subnormals on devices that flush them (the H100
cannot keep them in shaders), and tensors of 2^31 or more elements.
