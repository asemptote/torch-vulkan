# Matching the CPU's rounding (M25)

PyTorch's device-generic tests (issue #21) compare vk with the CPU, and
`test_decomp.py` compares each op with its decomposition, both run on vk, at
float32's or the 16-bit dtypes' tolerances. A vk op that is less accurate
than the CPU's fails these tests, and so does one that is *more* accurate,
because it rounds intermediates differently. The rule in both cases is to
round as the CPU does, step by step.

- **sqrt is correctly rounded** (`_fmath.slang`, `m_sqrt`). Vulkan's Sqrt is
  only as accurate as 1 / inversesqrt, and ANV's float32 roots differed from
  the correctly rounded ones in 8% of cases.
  - float32, in integers: x = m' 2^t with t even and m' < 2^25, and R = m' 2^S
    (47 or 48 bits, S = 22 or 24). The 24-bit root c = floor(sqrt(R)) is found
    from the GPU's float root by exact comparisons of c^2 with R, in products
    of 16-bit halves (no 64-bit integers, which not every GPU has). It rounds
    up iff R > c (c + 1), as (c + 1/2)^2 = c (c + 1) + 1/4 and R is an
    integer.
  - A first version corrected the GPU's root in floating point, with Dekker
    residuals. It was exact on ANV and on lavapipe on AVX2 CPUs. On the CI
    runners' AVX-512 CPUs, lavapipe left 481 of 65547 results 1 ulp off,
    11096 before a Newton step was added. Integers depend on no driver's
    rounding.
  - Checked against float64's root rounded to float32 (correctly rounded, as
    53 >= 2 * 24 + 2) for every non-negative float32 bit pattern on ANV: no
    differences.
  - float64 keeps the floating-point method: a Newton step on a Dekker
    residual, then neighbour tests, without FMAs, after scaling x by 2^±600
    into [2^-500, 2^500].
  - rsqrt is `m_div(1, m_sqrt(x))`, rounded twice, like the CPU's
    `1 / std::sqrt(x)`. pow(x, 0.5) is sqrt, as on the CPU.
  - Caveat: the CPU's own vectorised sqrt is not always correctly rounded.
    It is MKL's vsSqrt (ATen's vml.h), through ATen's AVX2 kernel on AVX-512
    CPUs too, where MKL picks its own code path: float64 had 1.4% of results
    1 ulp off and float32 about 0.1% of all inputs on an AVX2 CPU, and the CI
    runners' (AVX-512) differed from the correctly rounded root in 481 of the
    test's 65547 values. (An earlier note here blamed Sleef on AVX-512 CPUs:
    no PyTorch path calls Sleef for sqrt.) The tests compare against the
    correctly rounded root, which is what the CPU's scalar path gives, and vk
    keeps correct rounding.
- **log1p32**:
  - Old method: Goldberg's `log(u) x / (u - 1)` inherited log32's error, up
    to 2.8 ulp.
  - New method: 1 + x is taken as u + c exactly (two-sum). The result is ln u
    as two floats (`ln_hilo32`, the logarithm `pow32` already used), plus
    c / u, rounded once.
  - The quotient c / u is corrected with its exact residual. For x near
    -2^-25 the result cancels down to c's own size, so OpFDiv's 2.5 ulp
    would show.
  - Result: within 0.511 float32 eps of the true value; the CPU's is within
    0.525.
- **scatter_reduce / index_reduce "mean"** (`segreduce.slang`) divide with
  `m_div`. Their sums were already in the CPU's order, but OpFDiv's quotient
  was not correctly rounded.
- **Complex division** (`cbinary.slang`, `_complex.slang`'s `cdiv`): Smith's
  algorithm, as c10::complex's scalar `operator/`, with each quotient
  correctly rounded (`m_div`, under `CDIV_EXACT`), where it divided with
  OpFDiv. Complex scatter means divide each part with `m_div`, and
  test_scatter_reduce_mean (complex64) compares them with `x / count` on vk,
  which then differed by an ulp. (The CPU's vectorised complex division
  scales by 1 / max(|c|, |d|) instead, so it is not matched bit for bit.)
  The linear-algebra kernels' cdiv keeps OpFDiv.
- **16-bit ops whose intermediates the CPU rounds to the 16-bit dtype**:
  - nll_loss_backward rounds g = -(grad / total_weight) to the 16-bit dtype,
    then rounds each w g once. vk used to widen the whole op to float32.
  - ldexp takes 2^other in the 16-bit dtype before the product. Only where
    that power overflows (2^20 in float16) is it taken in float32.
  - log_sigmoid_backward reads the forward's buffer e^-|x|, rounded to the
    dtype, as the CPU's kernel does. Without a buffer (CUDA's convention)
    the GPU kernel recomputes it.
  - Result: all three are bit-identical to the CPU's in float16 and
    bfloat16.
- **Lazy conjugates inside kernels**. Inside a TorchDispatchMode
  (`test_decomp`'s cross-reference, FakeTensor, torch.compile's tracing),
  PyTorch excludes the Conjugate and Negative keys while the mode runs an op.
  A C++ kernel that makes a conjugate view and passes it to another at:: op
  then has the view read as raw data. tanh/sigmoid backward for complex
  inputs now use `conj_physical`. dot, which falls through the Conjugate key
  and so receives conjugate views, resolves them first.
- **linalg_pinv.atol_rtol_tensor** has its own derivative (pinv_backward), so
  vk's kernel for it (`_nofloat64.py`) is now a backend (PrivateUse1) kernel.
  pinv's composite overloads call it with their tolerances as tensors.
  - Before, registered on AutogradPrivateUse1, autograd differentiated through
    the SVD. Batched gradients then reached svd_backward's allclose check,
    which vmap has no batching rule for, and forward-mode AD failed on a view.
- **rrelu's noise under vmap**. The vk rrelu kernel (Python) draws its noise
  with `uniform_`, which goes through the dispatcher, so vmap's guards against
  random ops raised. The CPU's kernel draws its noise without the dispatcher,
  so the guards don't reach it.
  - Two guards are involved: torch.func's FuncTorchVmapMode key, and the
    legacy `torch._vmap_internals` VmapMode, which gradcheck's batched
    forward-mode check uses and which is entered per nesting level.
  - The kernel already runs below the batching layers, on plain tensors, so
    it now draws its noise outside both guards.

## Barriers between commands (#35)

- `vkcore::Stream` puts a barrier before every command. Since cccade8 it is a
  memory dependency from the previous command's stage and writes to the next
  command's stage and accesses (compute -> compute between dispatches,
  transfer bits only next to a copy or fill), no longer a full compute +
  transfer barrier: about 0.5 us less per dispatch on the H100 NVL.
  Every command still depends on the one before it, and so, through the
  chain, on all earlier ones.
- The evidence is the spec argument and tests, not the validation layer:
  synchronisation validation tracks accesses to resources bound through
  descriptors and by transfer commands, and every vk kernel reads and writes
  through buffer device addresses in push constants, which it does not see.
  So cccade8's clean TORCH_VK_VALIDATE=1 run checked the transfers'
  barriers only. (cccade8's message suggests more.) The checks that do bear
  on it: long chains of small dependent ops (element-wise, FFT round trips,
  index_add_ into one buffer, small matmuls, writes into views) bitwise equal
  with the narrowed and with full barriers on the H100 and L40S, and
  PyTorch's test files on both GPUs.
- `TORCH_VK_BARRIERS=full` restores the full barrier, to rule the narrowed
  ones out when a race is suspected.

## Large inputs (found on the way)

- **Sorts of more than 2^26 entries** (`sort.slang`): each local pass
  dispatched one workgroup per 1024-entry block, and lavapipe and most
  drivers allow at most 65535 workgroups per dispatch. The local passes now
  loop over the blocks with a grid stride, as the global ones already did.
- **Submissions on Intel GPUs** (`vkcore/src/stream.rs`): a submission that
  kept a Gen9 GPU (HD 530, Mesa's ANV, Linux's i915) busy for a couple of
  seconds came back with its later commands' results missing, and no lost
  device was reported. A 2^27-entry sort was unsorted, and a scatter of
  2^26 values wrote nothing. The stream now also submits once a batch holds
  2^13 workgroups on Intel GPUs, so a big dispatch goes alone while small
  ones still batch. `TORCH_VK_MAX_BATCH_GROUPS` overrides this, and 0 removes
  the limit.
- **sum_rows_by_key** (embedding's and grid_sampler's backward): keys it
  skips (zero padding's corners) all went to one spare row. That made one
  run, which one thread folds serially, about 1.6 s per call in
  grid_sampler_3d's gradgradcheck. They are now spread over 1024 spare rows.
- **scatter_add / scatter_reduce**:
  - With an index expanded along every dim but 0
    (`idx.unsqueeze(1).expand(n, k)`), they take whole rows: one sort key per
    row and segreduce over rows, as CUDA's vectorised path does.
  - An index of more than 2^26 elements is otherwise taken in slices along
    another dim. The slices' destinations are disjoint, so the order of
    accumulation is unchanged. This keeps each slice's per-element offsets
    and sort keys well inside a buffer's 4 GiB.

## rfft: the half-spectrum split in the last register pass (#35)

- rfft of 2n real values is an n-point complex FFT of their pairs followed by
  FFT_R2C, X[k] = (Z[k] + conj(Z[n - k])) / 2 + w^k (Z[k] - conj(Z[n - k])) / (2i).
  Where fft_pow2 takes the row in fft_reg.slang's register passes, the last
  pass computes X itself (`EPILOGUE = 1`). In that pass (ns = n / R), Z[k +
  q ns]'s mirror Z[n - k - q ns] is in column ns - k, so a workgroup takes g /
  2 columns k and their mirrors. Its last stage writes shared memory, and the
  epilogue reads each value and its mirror there and writes X, in rows of n +
  1. w^k comes from the two-level table for 2n.
- With `EPILOGUE = 0` the kernel is unchanged. Rows the register passes do
  not take keep the separate FFT_R2C pass, which reads each mirrored pair and
  one twiddle once (db43372).
- H100 NVL, synchronised calls against CUDA: rfft of 2^24 float32 points went
  from 2.6x to 1.6x CUDA's time (0.543 -> 0.324 ms), and 2^20 from 2.1x to
  1.8x. On the L40S, 2^24 float32 takes 0.642 ms, against CUDA's 0.777.
  Accuracy is the CPU's: relative errors of 1.9-2.8e-7 (float32) and
  6-9.6e-16 (float64) against a float64 reference, from 2^15 to 2^23 points.
