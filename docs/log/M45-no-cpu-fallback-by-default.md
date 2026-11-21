# No CPU fallback by default (M45)

Issue #101. Until 0.2, an op with no Vulkan kernel for its arguments ran on
the CPU: its "vk" tensors copied over and its results back, with a warning
the first time each op did so, counted in `fallback_counts()`.
`TORCH_VK_STRICT=1` made that an error. Every op PyTorch's 36 test files and
the OpInfo sweep exercise now runs on the GPU (M36, M44), so the default is
flipped, as PyTorch's MPS backend does (`PYTORCH_ENABLE_MPS_FALLBACK=1`):

- **Default:** an op that would compute on the CPU raises
  `NotImplementedError` naming the op and the switch:

      torch_vulkan: aten::sort.values_stable has no Vulkan kernel for these
      arguments, and the CPU fallback is off. To run it on the CPU instead
      (slower), set TORCH_VK_CPU_FALLBACK=1 or call
      torch_vulkan.set_cpu_fallback(True). Please report the op: <issues>

  "for these arguments": most such cases are an op that has a kernel but
  not for a dtype or layout (uint32 sort, say), so the message does not
  claim the op is missing.
- **Opt-in:** `TORCH_VK_CPU_FALLBACK=1` at start-up, or
  `torch_vulkan.set_cpu_fallback(True)` at any time (with
  `cpu_fallback_enabled()` to read it): the behaviour of 0.2, warning and
  counting included.

## What still calls the CPU with the fallback off

- **A CPU call that only raises.** Where vk does not repeat a check the CPU
  makes (a dtype or argument error), the op runs on CPU copies of its
  inputs, as `TORCH_VK_STRICT` did (`strict_probe`): if the CPU raises, its
  error reaches the user (counted as "op (raised)"); only if it would
  compute does vk raise `NotImplementedError`. The copies keep the caller's
  tensors unchanged either way. The probe does the CPU's work once in the
  second case, before raising; it is the error path.
- **GradScaler on GPUs without float64** lends its one-element scale to the
  CPU for the reciprocal (`_nofloat64.py`): host arithmetic on a CPU tensor,
  not a vk op on the CPU, so no fallback.
- **`TORCH_VK_LINALG=cpu`**, which sends dense linear algebra to LAPACK as a
  testing reference, turns the fallback on: it asks for CPU work, and
  without it every such op would raise.

## The switches

`TORCH_VK_CPU_FALLBACK` wins when set (non-empty): anything but `0` turns
the fallback on. Otherwise `TORCH_VK_STRICT`, kept for scripts written for
0.2: `1` is the new default and `0` turns the fallback on (0.2's default).
Otherwise `TORCH_VK_LINALG=cpu` turns it on. The switch is one atomic flag
in `backend.cpp`, read on every fallback, so `set_cpu_fallback` applies to
the next op in every thread; `_decomp.py`'s Python fallback reads the same
flag through `_native.cpu_fallback_enabled()`.

There is no middle mode that warns and then runs on the CPU: that is what
the opt-in does (one warning per op). A mode that raises for some ops and
not others did not seem worth its complexity: the switch is process-wide.

## Counting refusals

A refused call is counted in `fallback_counts()` as "op (refused)", without
a warning. `tools/pytorch_tests.py` records each test's fallback counts, so
its `cpu_ops.txt` still lists the ops without kernels that the tests reach,
now as refusals and failing their tests. With `--cpu-fallback` it runs them
on the CPU as before, to see whether the rest of a test passes;
`tools/opinfo_coverage.py --cpu-fallback` likewise reports such ops as
"fallback" rather than "error".

## Tests and CI

CI's `pytest` and strict OpInfo sweep run in the default mode, as users
will. The release workflow's wheel check, the Termux CI and
`tools/release_smoke.py` no longer set `TORCH_VK_STRICT=1`: they check that
the fallback is off. `tests/python/test_core.py` checks the refusal and its
message, the opt-in (`set_cpu_fallback`), and each environment variable and
their precedence in a subprocess.
