# Fused Adam with bfloat16 state (M42)

Issue #90. PyTorch 2.15's CUDA kernels for `_fused_adam_` and
`_fused_adamw_` take float32 parameters and gradients with bfloat16
optimiser state (`exp_avg`, `exp_avg_sq`, `max_exp_avg_sq`), which halves
the state's memory; 2.15's test_optim.py checks it on any accelerator
(`test_fused_mixed_precision_*`, 9 tests). The CPU's kernels refuse mixed
dtypes, so vk sent them to the CPU, which raised.

## What CUDA does

`fused_adam_utils.cuh`: when the state's dtype is not the parameters', the
only configuration accepted is float32 parameters and gradients with
bfloat16 state (`validate_mixed_precision_dtypes`, whose errors vk now
raises for any other mix). Each element's state is read into float32, the
step is computed in float32 exactly as for float32 state, and the new state
is stored rounded to bfloat16. So a mixed-precision step equals a float32
step whose state is rounded to bfloat16 after it, which is how PyTorch's
test builds its reference (the moments used for this step's parameter
update are the unrounded ones).

(The issue's title has the dtypes the other way round, bfloat16
parameters with float32 state; CUDA refuses that mix, and so does vk.)

## On vk

`shaders/optim.slang` gains a second template axis, `S`, the state's
storage type: `T` (as before) or bfloat16 (its 16 bits). The kernel reads
the state through `load` (bfloat16 to float32 is exact) and writes it
through `store` (round to nearest even, NaN as 0x7fc0, as `cast.slang` and
c10's `BFloat16`), and the rest of the step is unchanged: vk's float32
step with the CPU's formulas. The step therefore matches vk's own float32
step with the state rounded afterwards bit for bit (the new test checks
this), as CUDA's does its own.

The state is not converted to float32 copies around the launch (as the
float16 and bfloat16 parameter paths do): that would add two kernels and a
float32 temporary per state tensor and step, and use the memory the
bfloat16 state is meant to save. The launch is still one dispatch per
parameter tensor, declaring its operands' allocations through its push
constants as every dispatch (M37), so the barriers are unchanged.

The foreach and for-loop paths need nothing: PyTorch's
`_multi_tensor_adam` and `_single_tensor_adam` have no mixed-precision
handling, and refuse bfloat16 state with float32 parameters on every
device (`lerp_`'s dtype check: "expected dtype c10::BFloat16 for `end`
but got dtype float"), on vk as on the CPU.
