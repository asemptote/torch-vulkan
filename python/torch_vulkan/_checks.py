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


def _sizes(t):
    return "[" + ", ".join(map(str, t.shape)) + "]"


def _nll_loss(x, target, weight=None, *rest, **kw):
    if not 1 <= x.dim() <= 2:
        raise RuntimeError("input tensor should be 1D or 2D")
    if target.dim() > 1:
        raise RuntimeError("0D or 1D target tensor expected, multi-target not supported")
    if x.dim() == 1 and target.dim() == 1 and target.shape[0] != 1:
        raise ValueError(f"For 1D input, 1D target must have size 1, but got target size: {target.shape[0]}")
    if not (x.dim() == 1 and target.dim() == 0) and x.shape[0] != target.shape[0]:
        raise RuntimeError(f"size mismatch (got input: {_sizes(x)}, target: {_sizes(target)})")
    n = x.shape[-1]
    if weight is not None and (weight.dim() != 1 or weight.numel() != n):
        raise RuntimeError(f"weight tensor should be defined either for all {n} classes or no classes but got "
                           f"weight tensor of shape: {_sizes(weight)}")
    if target.dtype == torch.uint8:  # (the CPU kernel also takes byte targets)
        return (x, target.long(), weight, *rest), kw
    return None


def _nll_loss2d(x, target, weight=None, *rest, **kw):
    if target.dim() != 3:
        raise RuntimeError(f"only batches of spatial targets supported (3D tensors) but got targets of dimension: "
                           f"{target.dim()}")
    if x.dim() != 4:
        raise RuntimeError(f"only batches of spatial inputs supported (4D tensors), but got input of dimension: "
                           f"{x.dim()}")
    if weight is not None and weight.numel() != x.shape[1]:
        raise RuntimeError("weight tensor should be defined either for all or no classes")
    if (x.shape[0], *x.shape[2:]) != tuple(target.shape):
        raise RuntimeError(f"size mismatch (got input: {_sizes(x)} , target: {_sizes(target)}")
    if target.dtype != torch.int64:
        raise RuntimeError(f"expected scalar type Long but found {_C10_NAMES.get(str(target.dtype)[6:])}")
    return None


def _sliding_blocks(kernel_size, dilation, padding, stride):
    """The CPU's im2col/col2im shape checks of the block parameters."""
    (kh, kw), (dh, dw), (ph, pw), (sh, sw) = kernel_size, dilation, padding, stride
    if kh <= 0 or kw <= 0:
        raise RuntimeError(f"kernel size should be greater than zero, but got kernel_height: {kh} kernel_width: {kw}")
    if dh <= 0 or dw <= 0:
        raise RuntimeError(f"dilation should be greater than zero, but got dilation_height: {dh} dilation_width: {dw}")
    if ph < 0 or pw < 0:
        raise RuntimeError(f"padding should be non-negative, but got pad_height: {ph} pad_width: {pw}")
    if sh <= 0 or sw <= 0:
        raise RuntimeError(f"stride should be greater than zero, but got stride_height: {sh} stride_width: {sw}")
    if kh * kw >= 2 ** 63:
        raise RuntimeError(f"Given kernel_width = {kw} and kernel_height = {kh} the product of kernel_width and "
                           "kernel_height overflowed.")


def grid_sampler(input, grid, mode, k):
    """The CPU's grid_sampler_{2,3}d checks, with its messages."""
    if input.shape[0] != grid.shape[0]:
        raise RuntimeError(f"grid_sampler(): expected grid and input to have same batch size, but got input with "
                           f"sizes {list(input.shape)} and grid with sizes {list(grid.shape)}")
    if grid.shape[-1] != input.dim() - 2:
        raise RuntimeError(f"grid_sampler(): expected grid to have size {input.dim() - 2} in last dimension, but got "
                           f"grid with sizes {list(grid.shape)}")
    for d in range(2, input.dim()):
        if input.shape[d] <= 0:
            raise RuntimeError(f"grid_sampler(): expected input to have non-empty spatial dimensions, but input has "
                               f"sizes {list(input.shape)} with dimension {d} being empty")
    if input.dim() != k + 2 or grid.dim() != input.dim():
        raise RuntimeError(f"grid_sampler(): expected {k + 2}D input and grid with same number of dimensions, but got "
                           f"input with sizes {list(input.shape)} and grid with sizes {list(grid.shape)}")
    if k == 3 and mode == 2:
        raise RuntimeError("grid_sampler(): bicubic interpolation only supports 4D input")


CHECKS = {
    # (input, grid, interpolation_mode, padding_mode, align_corners)
    "grid_sampler_2d": lambda x, grid, mode, *rest, **kw: grid_sampler(x, grid, mode, 2),
    # (self, kernel_size, dilation, padding, stride) and (self, output_size, kernel_size, ...)
    "im2col": lambda x, *blocks, **kw: _sliding_blocks(*blocks[:4]) if len(blocks) >= 4 else None,
    "col2im": lambda x, output_size, *blocks, **kw: _sliding_blocks(*blocks[:4]) if len(blocks) >= 4 else None,
    # (self, target, weight, reduction, ignore_index); these may return replacement (args, kwargs)
    "nll_loss_forward": _nll_loss,
    "nll_loss2d_forward": _nll_loss2d,
    # (grad_output, self, target, ...)
    "nll_loss_backward": lambda grad, x, target, *rest, **kw: (
        ((grad, x, target.long(), *rest), kw) if target.dtype == torch.uint8 else None),
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
            replaced = check(*args, **kwargs)
            if replaced is not None:
                args, kwargs = replaced
        try:
            return kernel(*args, **kwargs)
        except RuntimeError as e:
            translated = _cast_error(e)
            if translated is None:
                raise
            raise translated from e

    return checked
