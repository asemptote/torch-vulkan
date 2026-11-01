"""Kernels for "vk" from PyTorch's own decompositions.

Many aten ops (losses, normalization, upsampling, grid sampling, most
backward ops, ...) have no Vulkan kernel of their own but have a
decomposition in torch._decomp: a Python function, tested against eager
mode by PyTorch, that computes the op from simpler aten ops. Registering
those functions as the PrivateUse1 kernels runs these ops as sequences of
our Vulkan kernels instead of on the CPU.

Only functional ops and out= variants are registered (no in-place ops), and
only ops with no "vk" kernel of their own, so C++ kernels are never shadowed. A
decomposition that ends up calling its own op (or raises) runs that call on
the CPU instead, counted in fallback_counts() like any other fallback.
"""

import threading

import torch
from torch._decomp import decomposition_table
from torch.utils._pytree import tree_flatten, tree_map

from . import _checks, _dtypes, _layout, _native

# Decompositions that are wrong or much slower than the CPU for eager use on
# "vk" (filled in from the OpInfo sweep and tests).
DENYLIST = {
    # Declines (returns NotImplemented for) most shapes; _kernels.py has a general version.
    "aten::adaptive_max_pool2d",
    "aten::adaptive_max_pool3d",
}

_active = threading.local()
_lib = None


def _has_kernel(name, key):
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(name, key)
    except RuntimeError:
        return False


def _device_of(args, kwargs):
    for x in (*args, *kwargs.values()):
        if isinstance(x, torch.Tensor) and x.device.type == "vk":
            return x.device
        if isinstance(x, (list, tuple)):
            for y in x:
                if isinstance(y, torch.Tensor) and y.device.type == "vk":
                    return y.device
    return torch.device("vk")


_HALF = (torch.float16, torch.bfloat16)
# Ops left to the CPU even with 16-bit floats (as backend.cpp's half_on_cpu):
# linear algebra and FFTs (no 16-bit CPU kernels), nan_to_num (dtype-extreme
# defaults), softshrink/hardshrink (threshold rounded to the dtype).
_HALF_ON_CPU = ("linalg_", "_linalg_", "cholesky", "lu", "_lu", "triangular_solve", "svd", "_svd", "geqrf", "ormqr",
                "orgqr", "qr", "householder", "eig", "_eig", "lstsq", "ldl", "inverse", "det", "logdet", "slogdet",
                "matrix_exp", "_fft", "fft_", "stft", "istft", "nan_to_num", "softshrink", "hardshrink")


def _is_vk(x):
    return isinstance(x, torch.Tensor) and x.device.type == "vk"


def _half_via_float(op, args, kwargs):
    """The op with its float16/bfloat16 "vk" tensors widened to float32, on the
    GPU, results rounded once (as CUDA computes most 16-bit ops); NotImplemented
    where that does not apply. As backend.cpp's half_via_float: the floating
    vk tensors must all be of one 16-bit dtype (mixed-dtype errors stay the
    CPU's), no view results, and the op must run on meta tensors of the
    original dtypes, which applies PyTorch's checks and gives the result
    dtypes."""
    floating = {x.dtype for x in tree_flatten((args, kwargs))[0]
                if _is_vk(x) and (x.is_floating_point() or x.is_complex())}
    if len(floating) != 1 or next(iter(floating)) not in _HALF:
        return NotImplemented
    if op._schema.name.split("::", 1)[1].startswith(_HALF_ON_CPU):
        return NotImplemented
    if any(r.alias_info is not None and not r.alias_info.is_write for r in op._schema.returns):
        return NotImplemented

    def to_meta(x):
        return torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta") if _is_vk(x) else x

    try:
        meta = op(*tree_map(to_meta, args), **tree_map(to_meta, kwargs))
    except Exception:  # noqa: BLE001 (the CPU decides)
        return NotImplemented

    def widen(x):
        return x.to(torch.float32) if _is_vk(x) and x.dtype in _HALF else x

    wide_args, wide_kwargs = tree_map(widen, args), tree_map(widen, kwargs)
    try:
        result = op(*wide_args, **wide_kwargs)
    except Exception:  # noqa: BLE001 (e.g. a dtype= the widened tensors contradict: the CPU decides)
        return NotImplemented
    written = [a.name for a in op._schema.arguments if a.alias_info and a.alias_info.is_write]
    if written:
        for name in written:
            target, source = kwargs[name], wide_kwargs[name]
            if target is not source:
                if target.shape != source.shape:
                    target.resize_(source.shape)
                target.copy_(source)
        outs = tuple(kwargs[name] for name in written)
        return outs[0] if len(outs) == 1 else outs
    flat_r, spec = tree_flatten(result)
    flat_m, _ = tree_flatten(meta)
    if len(flat_r) != len(flat_m):
        return result
    cast = [r.to(m.dtype) if isinstance(r, torch.Tensor) and isinstance(m, torch.Tensor) and r.dtype != m.dtype else r
            for r, m in zip(flat_r, flat_m)]
    return spec.unflatten(cast)


def _cpu_fallback(op, name, args, kwargs):
    result = _half_via_float(op, args, kwargs)
    if result is not NotImplemented:
        return result
    if not _native.record_fallback(name):
        raise NotImplementedError(f"torch_vulkan: {name} has no Vulkan kernel (TORCH_VK_STRICT forbids the CPU)")
    device = _device_of(args, kwargs)

    def to_cpu(x):
        return x.cpu() if isinstance(x, torch.Tensor) and x.device.type == "vk" else x

    def to_vk(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    cpu_args, cpu_kwargs = tree_map(to_cpu, args), tree_map(to_cpu, kwargs)
    result = op(*cpu_args, **cpu_kwargs)
    # out= arguments (keyword-only, written): copy the CPU results back into the
    # caller's tensors and return those, as the op would.
    written = [a.name for a in op._schema.arguments if a.alias_info and a.alias_info.is_write]
    if written:
        for name in written:
            target, source = kwargs[name], cpu_kwargs[name]
            if target.shape != source.shape:
                target.resize_(source.shape)
            target.copy_(source)
        outs = tuple(kwargs[name] for name in written)
        return outs[0] if len(outs) == 1 else outs
    return tree_map(to_vk, result)


def _make_kernel(op, decomp, name):
    def kernel(*args, **kwargs):
        active = _active.__dict__.setdefault("ops", set())
        if name in active:  # the decomposition calls its own op
            return _cpu_fallback(op, name, args, kwargs)
        active.add(name)
        try:
            result = decomp(*args, **kwargs)
        finally:
            active.discard(name)
        if result is NotImplemented:  # the decomposition declines these arguments
            return _cpu_fallback(op, name, args, kwargs)
        return result

    return kernel


def _python_kernels():
    """Ops with a kernel in _kernels.py, which take precedence over decompositions."""
    from . import _kernels
    return {"aten::" + name for name in _kernels._kernels()}


def candidates():
    """(OpOverload, decomposition, name) for every functional aten op that has
    a decomposition, a CPU kernel and no kernel for "vk" yet."""
    found = []
    for op, decomp in decomposition_table.items():
        if not isinstance(op, torch._ops.OpOverload) or op.namespace != "aten":
            continue
        name = op.name()
        schema = op._schema
        if name in DENYLIST or name in _python_kernels() or not _functional_or_out(schema):
            continue
        if not schema.returns:
            continue
        if not _has_kernel(name, "CPU") or _has_kernel(name, "PrivateUse1"):
            continue
        # Composite kernels already route to other ops.
        if _has_kernel(name, "CompositeImplicitAutograd") or _has_kernel(name, "CompositeExplicitAutograd"):
            continue
        # Structured functional ops reach their out= variant: only decompose
        # them when no out= variant has a "vk" kernel.
        if _has_kernel(name, "CompositeExplicitAutogradNonFunctional") and _out_variant_on_vk(op):
            continue
        found.append((op, decomp, name))
    return found


def _functional_or_out(schema):
    """True for functional schemas and out= variants (only keyword-only
    arguments are written), not for in-place ops."""
    written = [a for a in schema.arguments if a.alias_info and a.alias_info.is_write]
    return all(a.kwarg_only for a in written)


def _out_variant_on_vk(op):
    """True if an out= overload of `op` (same inputs plus outputs) has a "vk" kernel."""
    inputs = [a.name for a in op._schema.arguments]
    packet = getattr(torch.ops.aten, op._schema.name.split("::", 1)[1])
    for overload in packet.overloads():
        other = getattr(packet, overload)
        schema = other._schema
        if not schema.is_mutable:
            continue
        ins = [a.name for a in schema.arguments if not (a.alias_info and a.alias_info.is_write)]
        if ins == inputs and _has_kernel(other.name(), "PrivateUse1"):
            return True
    return False


def register():
    """Registers the decompositions as "vk" kernels; returns their names."""
    global _lib
    if _lib is not None:
        return []
    _lib = torch.library.Library("aten", "IMPL")
    names = []
    for op, decomp, name in candidates():
        overload = op._overloadname
        qualified = op._schema.name.split("::", 1)[1] + ("" if overload == "default" else "." + overload)
        base = op._schema.name.split("::", 1)[1]
        kernel = _checks.with_cpu_checks(op, _dtypes.with_cpu_dtypes(base, _make_kernel(op, decomp, name)))
        _lib.impl(qualified, _layout.with_layout(op, kernel), "PrivateUse1")
        names.append(name)
    return names
