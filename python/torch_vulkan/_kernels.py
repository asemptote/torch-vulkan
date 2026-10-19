"""Launching the strided element-wise kernels (shaders/_strided.slang).

Every kernel takes the same 128-byte push-constant block: up to three tensor
pointers, a scalar, the element count, and sizes/strides of up to 4 dims
after merging dims that are contiguous in every operand.
"""

import struct

import torch

from . import _native

# Element types the kernels are compiled for (kernel name suffixes).
LABELS = {
    torch.float32: "f32",
    torch.float64: "f64",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
}
NUMERIC = (torch.float32, torch.float64, torch.int32, torch.int64)
FLOATS = (torch.float32, torch.float64)

MAX_DIMS = 4
_PARAMS = struct.Struct("<QQQdqIIII8x4I4i4i4i")
assert _PARAMS.size == 128

# Op codes; must match shaders/unary.slang, binary.slang, compare.slang.
# kind: "num" = all numeric dtypes, "float" = float32/float64, "f32" = float32 only,
# "int" = integer dtypes only.
UNARY = {
    "neg": (0, "num"),
    "abs": (1, "num"),
    "relu": (2, "num"),
    "sign": (4, "num"),
    "reciprocal": (10, "float"),
    "sqrt": (11, "float"),
    "rsqrt": (12, "float"),
    "floor": (13, "float"),
    "ceil": (14, "float"),
    "trunc": (15, "float"),
    "round": (16, "float"),
    "exp": (20, "f32"),
    "log": (21, "f32"),
    "sin": (22, "f32"),
    "cos": (23, "f32"),
    "tanh": (24, "f32"),
    "sigmoid": (25, "f32"),
    "exp2": (26, "f32"),
    "log2": (27, "f32"),
    "tan": (28, "f32"),
}
BINARY = {
    "add": (0, "num"),
    "sub": (1, "num"),
    "mul": (2, "num"),
    "maximum": (3, "num"),
    "minimum": (4, "num"),
    "div": (10, "float"),
    "pow": (20, "f32"),
    "atan2": (21, "f32"),
    "bitwise_and": (30, "int"),
    "bitwise_or": (31, "int"),
    "bitwise_xor": (32, "int"),
}
COMPARE = {"eq": 0, "ne": 1, "lt": 2, "le": 3, "gt": 4, "ge": 5}


class Unsupported(Exception):
    """The kernels cannot run this case; the caller falls back to the CPU."""


def kind_allows(kind, dtype):
    if kind == "num":
        return dtype in NUMERIC
    if kind == "float":
        return dtype in FLOATS
    if kind == "f32":
        return dtype == torch.float32
    return dtype in (torch.int32, torch.int64)


def broadcast_strides(t, shape):
    """Element strides of `t` broadcast to `shape` (0 where it broadcasts)."""
    pad = len(shape) - t.dim()
    if pad < 0:
        raise Unsupported("operand has more dims than the output")
    out = [0] * pad
    for size, stride, target in zip(t.shape, t.stride(), shape[pad:]):
        if size == target:
            out.append(stride if size != 1 else 0)
        elif size == 1:
            out.append(0)
        else:
            raise Unsupported(f"cannot broadcast {tuple(t.shape)} to {tuple(shape)}")
    return out


def coalesce(shape, strides):
    """Merge dims that are contiguous in every operand.

    Returns [(size, [stride per operand])], innermost dim first.
    """
    dims = []
    for d in reversed(range(len(shape))):
        size = shape[d]
        if size == 1:
            continue
        st = [s[d] for s in strides]
        if dims:
            prev_size, prev_st = dims[-1]
            if all(st[k] == prev_st[k] * prev_size for k in range(len(st))):
                dims[-1] = (prev_size * size, prev_st)
                continue
        dims.append((size, st))
    return dims or [(1, [0] * len(strides))]


def launch(kernel, spec, out, a=None, b=None, alpha=0.0, alpha_i=0):
    """Queues `kernel` writing `out` from tensor operands `a`, `b` (or None)."""
    n = out.numel()
    if n == 0:
        return
    if n >= 2**32:
        raise Unsupported("more than 2^32 elements")
    shape = tuple(out.shape)
    operands = [out, a, b]
    strides = [list(out.stride())] + [broadcast_strides(t, shape) if t is not None else [0] * len(shape) for t in (a, b)]
    dims = coalesce(shape, strides)
    if len(dims) > MAX_DIMS:
        raise Unsupported(f"{len(dims)} non-mergeable dims (max {MAX_DIMS})")
    # Element offsets are 32-bit in the shaders. Only large tensors can exceed that.
    for t, st in zip(operands, strides):
        if t is not None and t.numel() >= 2**20 and sum((s - 1) * x for s, x in zip(shape, st)) >= 2**31:
            raise Unsupported("element offsets beyond 2^31")
    sizes = [d[0] for d in dims] + [1] * (MAX_DIMS - len(dims))
    per_operand = [[d[1][k] for d in dims] + [0] * (MAX_DIMS - len(dims)) for k in range(3)]
    groups, grid_stride = _native.grid_1d(n)
    push = _PARAMS.pack(
        out.vk_address,
        a.vk_address if a is not None else 0,
        b.vk_address if b is not None else 0,
        float(alpha),
        int(alpha_i),
        n,
        len(dims),
        grid_stride,
        0,
        *sizes,
        *per_operand[0],
        *per_operand[1],
        *per_operand[2],
    )
    _native.dispatch(kernel, list(spec), push, groups)
