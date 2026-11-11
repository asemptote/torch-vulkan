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

import os
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
_TRACE = os.environ.get("TORCH_VK_FALLBACK_TRACE", "0") not in ("", "0")
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


_SMALL_INT = {torch.int8: (-128, 127), torch.uint8: (0, 255), torch.int16: (-32768, 32767)}


def _half_via_float(op, args, kwargs):
    """The op with its float16/bfloat16 "vk" tensors widened to float32 (and
    int8/uint8/int16 ones to int32), on the GPU, results rounded once (as CUDA
    computes most 16-bit ops; small integers wrap as in their own width);
    NotImplemented where that does not apply. As backend.cpp's
    half_via_float: the floating vk tensors must all be of one dtype
    (mixed-dtype errors stay the CPU's), integer scalars must fit the small
    integer dtypes, no view results, and the op must run on meta tensors of
    the original dtypes, which applies PyTorch's checks and gives the result
    dtypes."""
    flat = tree_flatten((args, kwargs))[0]
    floating = {x.dtype for x in flat if _is_vk(x) and (x.is_floating_point() or x.is_complex())}
    small = {x.dtype for x in flat if _is_vk(x) and x.dtype in _SMALL_INT}
    if len(floating) > 1 or (floating and next(iter(floating)) not in _HALF and not small) or not (floating or small):
        return NotImplemented
    if floating and next(iter(floating)) not in _HALF:
        floating = set()  # (only the small integers widen)
    name = op._schema.name.split("::", 1)[1]
    if name.startswith(_HALF_ON_CPU):
        return NotImplemented
    if small and any(_is_vk(x) and x.dtype in _SMALL_INT and x.numel() >= 1 << 29 for x in flat):
        return NotImplemented  # (an int32 copy beyond 2 GiB, more than some devices' largest buffer)
    if small:
        lo = max(_SMALL_INT[d][0] for d in small)
        hi = min(_SMALL_INT[d][1] for d in small)
        ints = []  # (the Scalar arguments' integers, and 0-dim CPU integer tensors)
        for i, a in enumerate(op._schema.arguments):
            v = args[i] if i < len(args) else kwargs.get(a.name)
            if "Scalar" in str(a.type) and isinstance(v, int) and not isinstance(v, bool):
                ints.append(v)
            elif (isinstance(v, torch.Tensor) and v.dim() == 0 and v.device.type == "cpu" and v.dtype != torch.bool
                  and not v.is_floating_point() and not v.is_complex()):
                ints.append(int(v))
        if name.startswith("random") or any(not lo <= v <= hi for v in ints):
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
        if _is_vk(x) and x.dtype in _HALF:
            return x.to(torch.float32)
        return x.to(torch.int32) if _is_vk(x) and x.dtype in _SMALL_INT else x

    wide_args, wide_kwargs = tree_map(widen, args), tree_map(widen, kwargs)
    # (a dtype= of a narrow type: the widened one, the result cast back below)
    for i, a in enumerate(op._schema.arguments):
        if a.name == "dtype":
            narrow = wide_args[i] if i < len(wide_args) else wide_kwargs.get("dtype")
            wide = torch.int32 if narrow in _SMALL_INT else torch.float32 if narrow in _HALF else narrow
            if i < len(wide_args):
                wide_args = (*wide_args[:i], wide, *wide_args[i + 1:])
            elif "dtype" in wide_kwargs:
                wide_kwargs = {**wide_kwargs, "dtype": wide}
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


_STRICT = os.environ.get("TORCH_VK_STRICT", "0") not in ("", "0")
_DETAIL = os.environ.get("TORCH_VK_FALLBACK_DETAIL", "0") not in ("", "0")


def _describe(op, args, kwargs):
    """The argument signature backend.cpp gives with TORCH_VK_FALLBACK_DETAIL."""
    def one(v):
        if isinstance(v, torch.Tensor):
            s = repr(v.dtype).removeprefix("torch.")
            return f"{_C10_NAMES.get(s, s)}{' cpu' if v.device.type == 'cpu' else ''} {v.dim()}d" +                 (" empty" if v.numel() == 0 else "")
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            return f"list({one(v[0])})"
        if v is None:
            return "None"
        return {bool: "bool", int: "int", float: "float", complex: "complex"}.get(type(v), str(v))
    values = list(args) + [kwargs[a.name] for a in op._schema.arguments[len(args):] if a.name in kwargs]
    return " [" + ", ".join(one(v) for v in values) + "]"


_C10_NAMES = {"float32": "Float", "float64": "Double", "float16": "Half", "bfloat16": "BFloat16", "complex32": "ComplexHalf",
              "complex64": "ComplexFloat", "complex128": "ComplexDouble", "bool": "Bool", "uint8": "Byte", "int8": "Char",
              "int16": "Short", "int32": "Int", "int64": "Long", "uint16": "UInt16", "uint32": "UInt32",
              "uint64": "UInt64"}


def _dtype_signature(name, op, args, kwargs):
    """The key of _cpu_dtype_errors.TABLE (as tools/dev/cpu_dtype_errors.py's dtype_signature)."""
    import re

    parts = []
    for text in _describe(op, args, kwargs)[2:-1].split(", ") if (args or kwargs) else []:
        parts.append(text)
    # (_describe's parts, rejoined where a list's values held commas)
    joined, depth = [], 0
    for p in parts:
        if depth:
            joined[-1] += ", " + p
        else:
            joined.append(p)
        depth += p.count("[") + p.count("(") - p.count("]") - p.count(")")
    keys = []
    for text in joined:
        m = re.match(r"^(\w+)( cpu)? \d+d( empty)?", text)
        if m:
            keys.append(m.group(1) + (" empty" if m.group(3) else ""))
        elif text.startswith("list("):
            keys.append("list")
        elif re.match(r"^\[[-\d, ]*\]$", text):
            keys.append("IntList")
        else:
            keys.append(text)
    return f"{name} [{', '.join(keys)}]"


def _raise_cpu_dtype_error(op, name, args, kwargs):
    """Where the CPU would only raise a dtype error, raise it here (as backend.cpp's table)."""
    from ._cpu_dtype_errors import TABLE

    hit = TABLE.get(_dtype_signature(name, op, args, kwargs))
    if hit is not None:
        raise (NotImplementedError if hit[0] else RuntimeError)(_render_type_names(hit[1]))


_TYPE_NAMES = {}


def _render_type_names(msg):
    """{c10:Long} in a table message: this platform's C++ name for the dtype
    (c10's TypeMeta name, as the CPU's message gives it), found once from the
    CPU's own out= dtype error on 1x1 tensors."""
    import re

    def name(m):
        c10 = m.group(1)
        if c10 not in _TYPE_NAMES:
            dt = {v: k for k, v in _C10_NAMES.items()}.get(c10)
            dtype = getattr(torch, dt) if dt else None
            found = c10
            if dtype is not None:
                src = torch.float64 if dtype == torch.float32 else torch.float32
                try:
                    torch.mm(torch.ones(1, 1, dtype=src), torch.ones(1, 1, dtype=src), out=torch.empty(1, 1, dtype=dtype))
                except RuntimeError as e:
                    got = re.search(r"but got (.*) instead", str(e))
                    found = got.group(1) if got else c10
            _TYPE_NAMES[c10] = found
        return _TYPE_NAMES[c10]

    return re.sub(r"\{c10:(\w+)\}", name, msg)


def _check_devices(args, kwargs):
    """A tensor on a third device (meta, as FakeTensor's cross-checks pass) among
    vk ones: the error the CPU would raise, without it."""
    flat, _ = tree_flatten((args, kwargs))
    ts = [t for t in flat if isinstance(t, torch.Tensor)]
    if any(t.device.type == "meta" for t in ts):  # (FakeTensor's cross-checks: no data to copy over)
        raise NotImplementedError("Cannot copy out of meta tensor; no data!")
    if any(t.device.type == "vk" for t in ts):
        for t in ts:
            if t.device.type not in ("vk", "cpu"):
                raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices, "
                                   f"vk:0 and {t.device}!")


def _cpu_fallback(op, name, args, kwargs):
    _check_devices(args, kwargs)
    _raise_cpu_dtype_error(op, name, args, kwargs)
    result = _half_via_float(op, args, kwargs)
    if result is not NotImplemented:
        return result
    key = name + (_describe(op, args, kwargs) if _DETAIL else "")
    if _STRICT:
        # (as backend.cpp's strict_probe: the CPU on copies, so that its own
        # error reaches the caller; NotImplementedError only where it computes)
        def to_cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) and x.device.type == "vk" else x

        try:
            op(*tree_map(to_cpu, args), **tree_map(to_cpu, kwargs))
        except Exception:
            _native.record_fallback(key + " (raised)")
            raise
        raise NotImplementedError(f"torch_vulkan: {name} has no Vulkan kernel (TORCH_VK_STRICT forbids the CPU)")
    if _TRACE:  # (as backend.cpp's TORCH_VK_FALLBACK_TRACE: the call and where it came from)
        import sys
        import traceback
        print(f"torch_vulkan fallback (Python): {name}{_describe(op, args, kwargs)}\n"
              + "".join(traceback.format_stack(limit=8)[:-1]), file=sys.stderr, flush=True)
    try:
        return _run_on_cpu(op, args, kwargs, key)
    except Exception:
        _native.record_fallback(key + " (raised)")  # (as backend.cpp: the CPU only checked)
        raise


def _run_on_cpu(op, args, kwargs, key):
    device = _device_of(args, kwargs)

    def to_cpu(x):
        return x.cpu() if isinstance(x, torch.Tensor) and x.device.type == "vk" else x

    def to_vk(x):
        return x.to(device) if isinstance(x, torch.Tensor) else x

    cpu_args, cpu_kwargs = tree_map(to_cpu, args), tree_map(to_cpu, kwargs)
    result = op(*cpu_args, **cpu_kwargs)
    _native.record_fallback(key)
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
        # A decomposition may call its own op with other arguments (vdot of two
        # conjugate views calls vdot of their conjugates): a few levels are
        # allowed; deeper, it is taken to recur on the same ones.
        active = _active.__dict__.setdefault("ops", {})
        if active.get(name, 0) >= 3:
            return _cpu_fallback(op, name, args, kwargs)
        active[name] = active.get(name, 0) + 1
        try:
            result = decomp(*args, **kwargs)
        finally:
            active[name] -= 1
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
        # An out= variant whose functional op has a Python kernel: backend.cpp's
        # rescue_out runs that kernel (a decomposition may decline, to the CPU).
        if schema.is_mutable and schema.name in _python_kernels():
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
