"""Argument checks of PyTorch's CPU kernels that composed "vk" kernels lack.

A kernel composed from other ops (_kernels.py, the decompositions _decomp.py
registers) checks only what those ops check. The CPU kernel of the same op
may reject more (e.g. an in-place backward it cannot compute); the table
here repeats those checks, with the CPU's messages, per aten op (overload
packet name).
"""

import functools


def _inplace_negative_slope(name):
    def check(slope, is_result):
        if is_result and slope < 0:
            raise RuntimeError(
                f"In-place {name} backward calculation is triggered with a negative slope which is not supported. "
                "This is caused by calling in-place forward function with a negative slope, please call "
                "out-of-place version instead.")
    return check


_leaky_relu = _inplace_negative_slope("leakyReLu")
_elu = _inplace_negative_slope("elu")

CHECKS = {
    # (grad_output, self, negative_slope, self_is_result)
    "leaky_relu_backward": lambda grad, x, slope, is_result, *rest, **kw: _leaky_relu(slope, is_result),
    # (grad_output, alpha, scale, input_scale, is_result, self_or_result)
    "elu_backward": lambda grad, alpha, scale, input_scale, is_result, *rest, **kw: _elu(alpha, is_result),
}


def with_cpu_checks(name, kernel):
    """`kernel` for aten op `name` (overload packet name), first running the CPU's argument checks."""
    check = CHECKS.get(name)
    if check is None:
        return kernel

    @functools.wraps(kernel)
    def checked(*args, **kwargs):
        check(*args, **kwargs)
        return kernel(*args, **kwargs)

    return checked
