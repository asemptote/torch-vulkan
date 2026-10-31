"""Output layouts of Python kernels as PyTorch predicts them.

PyTorch's shape-only machinery (meta kernels, FakeTensor and therefore
torch.compile) predicts each op's output shapes and strides; a backend's
outputs must agree. Kernels written in Python (_kernels.py, and PyTorch's
decompositions registered by _decomp.py) build their results from other
ops, so their strides follow whatever those ops produced. `with_layout`
wraps such a kernel: the first call with a given input signature runs the
op under FakeTensorMode on fake "vk" tensors to learn the layout (cached),
and results whose strides differ are copied into that layout.
"""

import functools
import threading

import torch
from torch.utils._pytree import tree_flatten, tree_map

_MISSING = object()
_cache = {}
_local = threading.local()


def _signature(value):
    """A hashable description of an argument: tensors by layout, others by value."""
    if isinstance(value, torch.Tensor):
        return ("T", tuple(value.shape), tuple(value.stride()), value.dtype, value.device.type)
    if isinstance(value, (list, tuple)):
        return tuple(_signature(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _signature(v)) for k, v in value.items()))
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _fake_layouts(op, args, kwargs):
    """[(shape, stride) or None per tensor output] as FakeTensor predicts, or None."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    _local.busy = True
    try:
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            def fake(t):
                return mode.from_tensor(t) if isinstance(t, torch.Tensor) else t

            out = op(*tree_map(fake, args), **tree_map(fake, kwargs))
        leaves, _ = tree_flatten(out)
        return [(tuple(t.shape), tuple(t.stride())) if isinstance(t, torch.Tensor) else None for t in leaves]
    except Exception:
        return None  # no meta kernel, data-dependent shapes, ...: keep our layout
    finally:
        _local.busy = False


def _relayout(t, layout):
    if not isinstance(t, torch.Tensor) or layout is None:
        return t
    shape, stride = layout
    if tuple(t.shape) != shape:
        return t
    # A view into a larger buffer (nonzero offset) is copied too: an op's
    # output starts at its own storage, as PyTorch predicts.
    if t.storage_offset() == 0 and (tuple(t.stride()) == stride or t.numel() <= 1):
        return t
    out = torch.empty_strided(shape, stride, dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def with_layout(op, kernel):
    """`kernel` (implementing OpOverload `op`) with outputs laid out as FakeTensor predicts."""

    @functools.wraps(kernel)
    def wrapped(*args, **kwargs):
        # A kernel runs below the dispatcher: torch function modes active
        # in the caller (e.g. torch._refs' TorchRefsMode) must not see the
        # ops it is made of, as they would not see a C++ kernel's.
        with torch._C.DisableTorchFunction():
            return _laid_out(*args, **kwargs)

    def _laid_out(*args, **kwargs):
        result = kernel(*args, **kwargs)
        if getattr(_local, "busy", False) or op._schema.is_mutable:
            return result  # (in-place and out= kernels write into given tensors)
        key = (op, _signature(args), _signature(kwargs))
        layouts = _cache.get(key, _MISSING)
        if layouts is _MISSING:
            layouts = _fake_layouts(op, args, kwargs)
            _cache[key] = layouts
        if layouts is None:
            return result
        leaves, spec = tree_flatten(result)
        if len(leaves) != len(layouts):
            return result
        return spec.unflatten([_relayout(t, layout) for t, layout in zip(leaves, layouts)])

    return wrapped
