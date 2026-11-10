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


def _ints(v):
    return [v] if isinstance(v, int) else list(v)


def _pooling_output_shape(n, kernel, pad, stride, dilation, ceil_mode):
    """The CPU's pooling_output_shape, with its checks."""
    if stride == 0:
        raise RuntimeError("stride should not be zero")
    if pad < 0:
        raise RuntimeError(f"pad must be non-negative, but got pad: {pad}")
    if pad > ((kernel - 1) * dilation + 1) // 2:
        raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={pad}, "
                           f"kernel_size={kernel} and dilation={dilation}")
    out = (n + 2 * pad - dilation * (kernel - 1) - 1 + (stride - 1 if ceil_mode else 0)) // stride + 1
    if ceil_mode and (out - 1) * stride >= n + pad:
        out -= 1
    return out


def _max_pool(k):
    """The CPU's max_pool2d/3d_with_indices checks, with its messages."""
    count = "two" if k == 2 else "three"

    def check(x, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False, *rest, **kw):
        ks, st, pd, dl = _ints(kernel_size), _ints(stride), _ints(padding), _ints(dilation)
        if len(ks) not in (1, k):
            raise RuntimeError(f"max_pool{k}d: kernel_size must either be a single int, or a tuple of {count} ints")
        if len(st) not in (0, 1, k):
            raise RuntimeError(f"max_pool{k}d: stride must either be omitted, a single int, or a tuple of {count} "
                               "ints")
        if len(pd) not in (1, k):
            raise RuntimeError(f"max_pool{k}d: padding must either be a single int, or a tuple of {count} ints")
        if len(dl) not in (1, k):
            raise RuntimeError(f"max_pool{k}d: dilation must be either a single int, or a tuple of {count} ints")
        ks, pd, dl = (ks * k)[:k], (pd * k)[:k], (dl * k)[:k]
        st = (st * k)[:k] if st else ks
        if k == 2 and x.dim() not in (3, 4):
            raise RuntimeError("non-empty 3D or 4D (batch mode) tensor expected for input")
        if k == 3 and x.dim() not in (4, 5):
            raise RuntimeError("non-empty 4D or 5D (batch mode) tensor expected for input")
        spatial = list(x.shape[-k:])
        out = [_pooling_output_shape(spatial[d], ks[d], pd[d], st[d], dl[d], ceil_mode) for d in range(k)]
        names = ["H", "W"] if k == 2 else ["T", "H", "W"]
        if min(ks) <= 0:
            raise RuntimeError("kernel size should be greater than zero, but got " +
                               " ".join(f"k{n}: {v}" for n, v in zip(names, ks)))
        if min(st) <= 0:
            raise RuntimeError("stride should be greater than zero, but got " +
                               " ".join(f"d{n}: {v}" for n, v in zip(names, st)))
        if min(dl) <= 0:
            raise RuntimeError("dilation should be greater than zero, but got " +
                               " ".join(f"dilation{n}: {v}" for n, v in zip(names, dl)))
        if k == 2:
            n = x.dim()
            ok = (n == 3 and min(x.shape) != 0) or (n == 4 and min(x.shape[1:]) != 0)
            if not ok:
                raise RuntimeError("Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, "
                                   f"but got:{_sizes(x)}")
        else:
            fn = "max_pool3d_with_indices_out_cpu_template()"
            for i in range(1 if x.dim() == 5 else 0, x.dim()):
                if x.shape[i] <= 0:
                    raise RuntimeError(f"{fn}: Expected input's non-batch dimensions to have positive length, but "
                                       f"input has a shape of {_sizes(x)} and non-batch dimension {x.shape[i]} has "
                                       "length zero!")
        if min(out) < 1:
            planes = x.shape[-k - 1]
            raise RuntimeError(f"Given input size: ({'x'.join(map(str, (planes, *spatial)))}). Calculated output "
                               f"size: ({'x'.join(map(str, (planes, *out)))}). Output size is too small")

    return check


def _adaptive_max_pool(k):
    """The CPU's adaptive_max_pool2d/3d checks, with its messages."""
    def check(x, output_size, *rest, **kw):
        fn = f"adaptive_max_pool{k}d()"
        if x.dim() not in (k + 1, k + 2):
            raise RuntimeError(f"{fn}: Expected {k + 1}D or {k + 2}D tensor, but got: {_sizes(x)}")
        for i in range(1, x.dim()):
            if x.shape[i] <= 0:
                raise RuntimeError(f"{fn}: Expected input to have non-zero size for non-batch dimensions, but input "
                                   f"has sizes {_sizes(x)} with dimension {i} being empty")
        if len(output_size) != k:
            raise RuntimeError(f"{fn}: internal error: output_size.size() must be {k}")
        shape = [*x.shape[:-k], *output_size]
        for v in shape:
            if v < 0:
                raise RuntimeError(f"Trying to create tensor with negative dimension {v}: {shape}")
    return check


def _margin_input(x):
    if not ((x.dim() == 2 and x.shape[1] != 0) or (x.dim() == 1 and x.shape[0] != 0) or x.dim() == 0):
        raise RuntimeError(f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {_sizes(x)}")
    return (1, 1 if x.dim() == 0 else x.shape[0]) if x.dim() <= 1 else tuple(x.shape)


def _long_target(target):
    if target.dtype != torch.int64:
        raise RuntimeError(f"expected scalar type Long but found {_C10_NAMES.get(str(target.dtype)[6:])}")


def _multi_margin_loss(x, target, p=1, margin=1, weight=None, *rest, **kw):
    if p not in (1, 2):
        raise RuntimeError("only p == 1 and p == 2 supported")
    nframe, dim = _margin_input(x)
    if target.dim() > 1 or target.numel() != nframe:
        raise RuntimeError("multi_margin_loss: target tensor should be 1-D with size equal to the number of input "
                           f"samples (batch size). Expected target size [{nframe}], but got {_sizes(target)}. Input "
                           f"has shape {_sizes(x)}.")
    if weight is not None and (weight.dim() > 1 or weight.numel() != dim):
        raise RuntimeError(f"inconsistent weight size, expected {dim} but got {_sizes(weight)}")
    _long_target(target)


def _multilabel_margin_loss(x, target, *rest, **kw):
    nframe, dim = _margin_input(x)
    ok = (target.dim() <= 1 and target.numel() == dim) if x.dim() <= 1 else (
        target.dim() == 2 and tuple(target.shape) == (nframe, dim))
    if not ok:
        raise RuntimeError(f"inconsistent target size: {_sizes(target)} for input of size: {_sizes(x)}")
    _long_target(target)


def _cross(a, b, dim, out):
    if a.dim() != b.dim():
        raise RuntimeError("linalg.cross: inputs must have the same number of dimensions.")
    if not -max(a.dim(), 1) <= dim < max(a.dim(), 1):
        raise IndexError(f"Dimension out of range (expected to be in range of [{-max(a.dim(), 1)}, "
                         f"{max(a.dim(), 1) - 1}], but got {dim})")
    if a.shape[dim] != 3 or b.shape[dim] != 3:
        raise RuntimeError(f"linalg.cross: inputs dimension {dim} must have length 3. Got {a.shape[dim]} and "
                           f"{b.shape[dim]}")
    _no_overlap(out, a, b)


CHECKS = {
    # (self, kernel_size, stride, padding, dilation, ceil_mode)
    "max_pool2d_with_indices": _max_pool(2),
    "max_pool3d_with_indices": _max_pool(3),
    # (self, output_size)
    "adaptive_max_pool2d": _adaptive_max_pool(2),
    "adaptive_max_pool3d": _adaptive_max_pool(3),
    # (self, target, p, margin, weight, reduction) and (self, target, reduction)
    "multi_margin_loss": _multi_margin_loss,
    "multilabel_margin_loss_forward": _multilabel_margin_loss,
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
    "linalg_cross": lambda a, b, dim=-1, *rest, out=None, **kw: _cross(a, b, dim, out),
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
