"""Argument checks of PyTorch's CPU kernels that composed "vk" kernels lack.

A kernel composed from other ops (_kernels.py, the decompositions _decomp.py
registers) checks only what those ops check. The CPU kernel of the same op
may reject more: an out= tensor on another device (copy_ would quietly
write it), or an in-place backward it cannot compute. with_cpu_checks adds
the out= device check to every out= overload and the CHECKS table's checks,
with the CPU's messages, per aten op (overload packet name).
"""

import functools
import re

import torch


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


def _overlap(a, b):
    """True if tensors a and b share memory."""
    if a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr() or a.numel() == 0 or b.numel() == 0:
        return False

    def span(t):
        lo = t.storage_offset()
        hi = lo + sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if st > 0)
        return lo * t.element_size(), (hi + 1) * t.element_size()
    (a0, a1), (b0, b1) = span(a), span(b)
    return a0 < b1 and b0 < a1


def _no_overlap(out, *inputs):
    for t in inputs:
        if isinstance(t, torch.Tensor) and isinstance(out, torch.Tensor) and _overlap(out, t):
            raise RuntimeError("unsupported operation: some elements of the input tensor and the written-to tensor "
                               "refer to a single memory location. Please clone() the tensor before performing the "
                               "operation.")


CHECKS = {
    # (self, other, dim, *, out)
    "linalg_cross": lambda a, b, *rest, out=None, **kw: _no_overlap(out, a, b),
    # (grad_output, self, negative_slope, self_is_result)
    "leaky_relu_backward": lambda grad, x, slope, is_result, *rest, **kw: _leaky_relu(slope, is_result),
    # (grad_output, alpha, scale, input_scale, is_result, self_or_result)
    "elu_backward": lambda grad, alpha, scale, input_scale, is_result, *rest, **kw: _elu(alpha, is_result),
}


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for t in value if isinstance(t, torch.Tensor)]
    return []


def _out_device_check(op):
    """For an out= overload: a check that out= tensors are on "vk" when an input is."""
    outs = [a.name for a in op._schema.arguments if a.kwarg_only and a.alias_info and a.alias_info.is_write]
    if not outs:
        return None

    def check(*args, **kwargs):
        inputs = [t for v in (*args, *(v for k, v in kwargs.items() if k not in outs)) for t in _tensors(v)]
        if not any(t.device.type == "vk" for t in inputs):
            return
        for name in outs:
            for t in _tensors(kwargs.get(name)):
                if t.device.type != "vk":
                    raise RuntimeError("Expected all tensors to be on the same device, but found at least two "
                                       f"devices, vk:0 and {t.device}!")

    return check


# c10's names for dtypes in error messages.
_C10_NAMES = {"float32": "Float", "float64": "Double", "float16": "Half", "bfloat16": "BFloat16", "complex64":
              "ComplexFloat", "complex128": "ComplexDouble", "int8": "Char", "int16": "Short", "int32": "Int",
              "int64": "Long", "uint8": "Byte", "bool": "Bool"}
_UNSAFE_CAST = re.compile(r"Attempting to cast from torch\.(\w+) to out tensor with dtype torch\.(\w+), but this "
                          r"can't be cast because it is not safe")


def _cast_error(error):
    """PyTorch's decompositions word an unsafe out= cast differently from its
    kernels: the kernels' wording ('result type Float can't be cast to the
    desired output type Int'), or None."""
    m = _UNSAFE_CAST.search(str(error))
    if m is None:
        return None
    a, b = (_C10_NAMES.get(name, name) for name in m.groups())
    return RuntimeError(f"result type {a} can't be cast to the desired output type {b}")


def with_cpu_checks(op, kernel):
    """`kernel` for aten OpOverload `op`, first running the CPU's argument checks."""
    checks = [c for c in (_out_device_check(op), CHECKS.get(op._schema.name.split("::", 1)[1])) if c is not None]
    if not checks and not op._schema.is_mutable:
        return kernel

    @functools.wraps(kernel)
    def checked(*args, **kwargs):
        for check in checks:
            check(*args, **kwargs)
        try:
            return kernel(*args, **kwargs)
        except RuntimeError as e:
            translated = _cast_error(e)
            if translated is None:
                raise
            raise translated from e

    return checked
