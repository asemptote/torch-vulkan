# Matching the CPU's rounding (M25)

PyTorch's device-generic tests (issue #21) compare vk with the CPU, and
`test_decomp.py` compares each op with its decomposition, both run on vk, at
float32's or the 16-bit dtypes' tolerances. A vk op that is less accurate
than the CPU's fails these tests, and so does one that is *more* accurate,
because it rounds intermediates differently. The rule in both cases is to
round as the CPU does, step by step.

- **sqrt is correctly rounded** (`_fmath.slang`, `m_sqrt`). Vulkan's Sqrt is
  only as accurate as 1 / inversesqrt, and ANV's float32 roots differed from
  the CPU's in 8% of cases.
  - Method: x is scaled by an even power of two so its significand M lies in
    [1, 4). The GPU's root c of M is then moved to a neighbour while the true
    root lies beyond the midpoint on that side.
  - Test: (c + u/2)^2 = c (c + u) + u^2 / 4, and M - c (c + u) is a multiple
    of u^2. So the root exceeds the midpoint exactly when M - c next(c) > 0.
    That residual is computed exactly from Dekker's product, so its sign is
    exact.
  - float64 runs the same tests, also without FMAs, after scaling x by
    2^±600 into [2^-500, 2^500].
  - rsqrt is `m_div(1, m_sqrt(x))`, rounded twice, like the CPU's
    `1 / std::sqrt(x)`. pow(x, 0.5) is sqrt, as on the CPU.
  - Result: float32 sqrt, rsqrt and pow(x, 0.5) are bit-identical to the
    CPU's (0 of 2^20 values differ).
  - One caveat: the CPU's own float64 sqrt on large contiguous tensors (MKL,
    on the Linux wheels) is not correctly rounded. It differs from
    `math.sqrt` in 1.4% of cases, by 1 ulp. vk keeps correct rounding, which
    is what the CPU's scalar path gives. The CPU's float64 rsqrt does match
    1 / (correctly rounded sqrt), and so does vk's.
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
