"""VkTensor: a tensor whose data lives in Vulkan memory, and its op dispatch.

Every aten op on a VkTensor arrives in `VkTensor.__torch_dispatch__` (below
autograd, so gradients work unchanged). It is handled by, in order:

1. a GPU handler registered with `@handles(...)`;
2. for view ops (transpose, slice, expand, ...), the shape/stride/offset
   computed on a meta tensor, sharing the same Vulkan buffer;
3. a CPU fallback: copy operands to the CPU, run the op there, copy results
   (and mutated arguments) back. Counted in `fallback_counts()`.
"""

import collections
import warnings

import torch
from torch.utils._pytree import tree_flatten, tree_map

from . import _native
from ._kernels import BINARY, COMPARE, FLOATS, LABELS, NUMERIC, UNARY, Unsupported, kind_allows, launch

aten = torch.ops.aten
DEVICE = torch.device("vk", 0)

_HANDLERS = {}
_FALLBACKS = collections.Counter()
_WARNED = set()


def fallback_counts():
    """How often each op has run on the CPU instead of the GPU."""
    return dict(_FALLBACKS)


class VkTensor(torch.Tensor):
    """Tensor metadata (shape, strides, dtype) plus a Vulkan buffer.

    `_vk_buf` is a `_native.Buffer`; the tensor's storage starts `_vk_base`
    bytes into it, and `storage_offset()` counts elements from there.
    Views share the buffer object, which frees the memory when the last
    tensor using it dies.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, buf, base, size, stride, storage_offset, dtype, requires_grad=False):
        t = torch.Tensor._make_wrapper_subclass(
            cls, size, strides=stride, storage_offset=storage_offset, dtype=dtype, device=DEVICE, requires_grad=requires_grad
        )
        t._vk_buf = buf
        t._vk_base = base
        return t

    @property
    def vk_address(self):
        """GPU address of element 0."""
        return self._vk_buf.address + self._vk_base + self.storage_offset() * self.element_size()

    def __repr__(self, *, tensor_contents=None):
        text = repr(self.detach().cpu())
        return text[:-1] + f", device='{DEVICE}')"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        handler = _HANDLERS.get(func)
        if handler is not None:
            try:
                return handler(*args, **kwargs)
            except Unsupported:
                pass
        if func.is_view:
            return _view(func, args, kwargs)
        return _cpu_fallback(func, args, kwargs)


def handles(*ops):
    def register(fn):
        for op in ops:
            _HANDLERS[op] = fn
        return fn

    return register


# ---------------------------------------------------------------- memory / transfer


def _span(shape, stride, storage_offset):
    """Elements of storage covered by a view, counted from element 0."""
    if any(s == 0 for s in shape):
        return storage_offset
    return storage_offset + sum((s - 1) * st for s, st in zip(shape, stride)) + 1


def empty_strided(shape, stride, dtype):
    shape, stride = tuple(shape), tuple(stride)
    buf = _native.alloc(max(_span(shape, stride, 0), 1) * dtype.itemsize)
    return VkTensor(buf, 0, shape, stride, 0, dtype)


def contiguous_strides(shape):
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(size, 1)
    return tuple(reversed(strides))


def empty(shape, dtype=None):
    shape = tuple(shape)
    return empty_strided(shape, contiguous_strides(shape), dtype or torch.get_default_dtype())


def _storage_bytes(t):
    """(byte offset into its buffer, byte length) of the storage range `t` covers."""
    start = t.storage_offset()
    count = _span(t.shape, t.stride(), start) - start
    return t._vk_base + start * t.element_size(), count * t.element_size()


def to_cpu(t):
    """Copy of `t` as an ordinary CPU tensor (waits for queued GPU work)."""
    offset, nbytes = _storage_bytes(t)
    flat = torch.empty(nbytes // t.element_size(), dtype=t.dtype)
    if nbytes:
        _native.download(t._vk_buf, offset, flat.data_ptr(), nbytes)
    return flat.as_strided(t.shape, t.stride(), 0).clone()


def from_cpu(t):
    """Copy of CPU tensor `t` on the Vulkan device (contiguous)."""
    t = t.detach().contiguous()
    out = empty(t.shape, t.dtype)
    nbytes = t.numel() * t.element_size()
    if nbytes:
        _native.upload(out._vk_buf, out._vk_base, t.data_ptr(), nbytes)
    return out


def to_vk(t):
    """`t` on the Vulkan device (returned as-is if already there)."""
    return t if isinstance(t, VkTensor) else from_cpu(t)


def _write_from_cpu(dst, src):
    """dst[...] = src for a CPU tensor `src` broadcastable to dst."""
    src = src.to(dst.dtype).expand(dst.shape)
    if dst.is_contiguous() and src.is_contiguous():
        if dst.numel():
            _native.upload(dst._vk_buf, dst._vk_base + dst.storage_offset() * dst.element_size(), src.data_ptr(), dst.numel() * dst.element_size())
        return
    if dst.dtype in LABELS:
        tmp = from_cpu(src.contiguous())
        launch(f"cast_{LABELS[dst.dtype]}_{LABELS[dst.dtype]}", [], dst, tmp)
        return
    # Read-modify-write the storage range on the host.
    offset, nbytes = _storage_bytes(dst)
    flat = torch.empty(nbytes // dst.element_size(), dtype=dst.dtype)
    _native.download(dst._vk_buf, offset, flat.data_ptr(), nbytes)
    flat.as_strided(dst.shape, dst.stride(), 0).copy_(src)
    _native.upload(dst._vk_buf, offset, flat.data_ptr(), nbytes)


# ---------------------------------------------------------------- views and fallback


def _meta(t):
    storage = torch.empty(_span(t.shape, t.stride(), t.storage_offset()), dtype=t.dtype, device="meta")
    return storage.as_strided(t.shape, t.stride(), t.storage_offset())


def _view(func, args, kwargs):
    base = next(a for a in tree_flatten((args, kwargs))[0] if isinstance(a, VkTensor))
    meta_args, meta_kwargs = tree_map(lambda a: _meta(a) if isinstance(a, VkTensor) else a, (args, kwargs))
    out = func(*meta_args, **meta_kwargs)

    def wrap(m):
        if not isinstance(m, torch.Tensor):
            return m
        return VkTensor(base._vk_buf, base._vk_base, m.shape, m.stride(), m.storage_offset(), m.dtype)

    return tree_map(wrap, out)


def _cpu_fallback(func, args, kwargs):
    _FALLBACKS[str(func)] += 1
    if str(func) not in _WARNED:
        _WARNED.add(str(func))
        warnings.warn(f"torch_vulkan: {func} has no Vulkan kernel yet; running it on the CPU", stacklevel=3)

    cpu_of = {}

    def to_host(a):
        if isinstance(a, VkTensor):
            cpu_of[id(a)] = (a, to_cpu(a))
            return cpu_of[id(a)][1]
        if isinstance(a, torch.device) and a.type == "vk":
            return torch.device("cpu")
        return a

    cpu_args, cpu_kwargs = tree_map(to_host, (args, kwargs))
    if "device" in cpu_kwargs and cpu_kwargs["device"] is not None and torch.device(cpu_kwargs["device"]).type == "vk":
        cpu_kwargs["device"] = torch.device("cpu")
    out = func(*cpu_args, **cpu_kwargs)

    # Write back arguments the op mutates (in-place ops, out= arguments).
    schema_args = func._schema.arguments
    mutated = {}
    for i, arg in enumerate(schema_args):
        value = args[i] if i < len(args) else kwargs.get(arg.name)
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        for v in tree_flatten(value)[0]:
            if isinstance(v, VkTensor):
                original, host = cpu_of[id(v)]
                if host.shape != original.shape or host.stride() != original.stride():
                    raise NotImplementedError(f"torch_vulkan: {func} changes tensor metadata in place, which VkTensor does not support yet")
                _write_from_cpu(original, host)
                mutated[id(host)] = original

    def result(o):
        if isinstance(o, torch.Tensor):
            if id(o) in mutated:
                return mutated[id(o)]
            return from_cpu(o)
        return o

    return tree_map(result, out)


# ---------------------------------------------------------------- helpers


def _is_scalar(x):
    """Python number, or a CPU 0-dim tensor (used as a scalar operand)."""
    return isinstance(x, (bool, int, float)) or (isinstance(x, torch.Tensor) and not isinstance(x, VkTensor) and x.dim() == 0)


def _scalar_value(x):
    return x.item() if isinstance(x, torch.Tensor) else x


_PROXIES = {}


def _proxy(x):
    """A stand-in for type promotion: same dtype and 0-dim-ness, no data."""
    if not isinstance(x, torch.Tensor):
        return x
    key = (x.dtype, x.dim() == 0)
    proxy = _PROXIES.get(key)
    if proxy is None:
        proxy = _PROXIES[key] = torch.empty(() if key[1] else (0,), dtype=x.dtype, device="meta")
    return proxy


def _result_type(a, b):
    return torch.result_type(_proxy(a), _proxy(b))


def _broadcast_shape(*xs):
    return torch.broadcast_shapes(*(tuple(x.shape) for x in xs if isinstance(x, torch.Tensor)))


def _cast(t, dtype):
    """`t` converted to `dtype` (contiguous copy), or `t` itself if already `dtype`."""
    if t.dtype == dtype:
        return t
    if t.dtype not in LABELS or dtype not in LABELS:
        raise Unsupported(f"no cast kernel for {t.dtype} -> {dtype}")
    out = empty(t.shape, dtype)
    launch(f"cast_{LABELS[t.dtype]}_{LABELS[dtype]}", [], out, t)
    return out


def _scalar_slots(dtype, value):
    """(alpha, alpha_i) push-constant values carrying scalar `value`."""
    if dtype in FLOATS:
        return float(value), 0
    return 0.0, int(value)


# ---------------------------------------------------------------- factories, copies, fills


@handles(aten.empty.memory_format)
def _empty(size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return empty(size, dtype)


@handles(aten.empty_strided.default)
def _empty_strided(size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    return empty_strided(size, stride, dtype or torch.get_default_dtype())


def _on_vk(device):
    return device is None or torch.device(device).type == "vk"


@handles(aten.empty_like.default)
def _empty_like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    if not _on_vk(device):
        raise Unsupported
    return empty(x.shape, dtype or x.dtype)


@handles(aten.new_empty.default)
def _new_empty(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if not _on_vk(device):
        raise Unsupported
    return empty(size, dtype or x.dtype)


def fill_(t, value):
    if t.dtype not in LABELS:
        raise Unsupported
    alpha, alpha_i = _scalar_slots(t.dtype, value)
    launch(f"fill_{LABELS[t.dtype]}", [], t, alpha=alpha, alpha_i=alpha_i)
    return t


@handles(aten.fill_.Scalar)
def _fill_scalar(t, value):
    return fill_(t, value)


@handles(aten.fill_.Tensor)
def _fill_tensor(t, value):
    if isinstance(value, VkTensor) or value.dim() != 0:
        raise Unsupported
    return fill_(t, value.item())


@handles(aten.zero_.default)
def _zero(t):
    return fill_(t, 0)


def _filled_like(value):
    def like(x, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
        if not _on_vk(device):
            raise Unsupported
        return fill_(empty(x.shape, dtype or x.dtype), value)

    return like


_HANDLERS[aten.zeros_like.default] = _filled_like(0)
_HANDLERS[aten.ones_like.default] = _filled_like(1)


@handles(aten.full_like.default)
def _full_like(x, value, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    if not _on_vk(device):
        raise Unsupported
    return fill_(empty(x.shape, dtype or x.dtype), value)


@handles(aten.copy_.default)
def _copy(dst, src, non_blocking=False):
    if not isinstance(dst, VkTensor):  # cpu_tensor.copy_(vk_tensor)
        return dst.copy_(to_cpu(src))
    if not isinstance(src, VkTensor):
        _write_from_cpu(dst, src)
        return dst
    if dst.dtype not in LABELS or src.dtype not in LABELS:
        _write_from_cpu(dst, to_cpu(src))
        return dst
    launch(f"cast_{LABELS[src.dtype]}_{LABELS[dst.dtype]}", [], dst, src)
    return dst


@handles(aten._copy_from.default)
def _copy_from(src, dst, non_blocking=False):
    # Reached from C++ copy_ between devices (e.g. the CPU -> vk step of .to("vk")).
    return _copy(dst, src)


@handles(aten._copy_from_and_resize.default)
def _copy_from_and_resize(src, dst):
    if tuple(src.shape) != tuple(dst.shape):
        raise NotImplementedError("torch_vulkan: resizing a tensor during copy is not supported yet")
    return _copy(dst, src)


@handles(aten.clone.default)
def _clone(x, *, memory_format=None):
    if x.dtype not in LABELS:
        raise Unsupported
    out = empty(x.shape, x.dtype)
    launch(f"cast_{LABELS[x.dtype]}_{LABELS[x.dtype]}", [], out, x)
    return out


@handles(aten._to_copy.default)
def _to_copy(x, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None):
    dtype = dtype or x.dtype
    if device is not None and torch.device(device).type == "cpu":
        return to_cpu(x).to(dtype)
    if x.dtype in LABELS and dtype in LABELS:
        return _clone(x) if dtype == x.dtype else _cast(x, dtype)
    return from_cpu(to_cpu(x).to(dtype))


@handles(aten._local_scalar_dense.default)
def _item(x):
    return to_cpu(x).item()


# ---------------------------------------------------------------- element-wise math

_FLOAT_RESULT = {"reciprocal", "sqrt", "rsqrt", "exp", "log", "sin", "cos", "tanh", "sigmoid", "exp2", "log2", "tan"}


def _unary(name):
    code, kind = UNARY[name]

    def functional(x):
        if name in _FLOAT_RESULT and x.dtype not in FLOATS:
            x = _cast(x, torch.get_default_dtype())  # torch.sqrt(int) -> float
        if not kind_allows(kind, x.dtype):
            raise Unsupported
        out = empty(x.shape, x.dtype)
        launch(f"unary_{LABELS[x.dtype]}", [code], out, x)
        return out

    def inplace(x):
        if not kind_allows(kind, x.dtype):
            raise Unsupported
        launch(f"unary_{LABELS[x.dtype]}", [code], x, x)
        return x

    return functional, inplace


for _name in UNARY:
    _f, _i = _unary(_name)
    _HANDLERS[getattr(aten, _name).default] = _f
    if hasattr(aten, _name + "_"):
        _HANDLERS[getattr(aten, _name + "_").default] = _i


def _binary(name, a, b, alpha=1, out=None):
    """Shared path for binary ops; `out` given means in place into `out`."""
    code, kind = BINARY[name]
    if not isinstance(a, VkTensor):
        raise Unsupported  # scalar first operand (rare; e.g. torch.maximum(2, x))
    dtype = _result_type(a, b)
    if name == "div" and dtype not in FLOATS:
        dtype = torch.get_default_dtype()  # true division
    if not kind_allows(kind, dtype):
        raise Unsupported
    if out is not None and out.dtype != dtype:
        raise Unsupported  # e.g. int_tensor.add_(0.5): let the CPU path apply torch's rules
    a = _cast(a, dtype)
    shape = _broadcast_shape(a, b) if isinstance(b, VkTensor) else tuple(a.shape)
    if out is None:
        out = empty(shape, dtype)
    elif tuple(out.shape) != tuple(shape):
        raise Unsupported
    label = LABELS[dtype]
    if _is_scalar(b):
        value = _scalar_value(b)
        if name in ("add", "sub"):
            value = value * alpha  # fold alpha into the scalar
        s_alpha, s_alpha_i = _scalar_slots(dtype, value)
        launch(f"binary_{label}", [code, 1], out, a, alpha=s_alpha, alpha_i=s_alpha_i)
    elif isinstance(b, VkTensor):
        b = _cast(b, dtype)
        m_alpha, m_alpha_i = _scalar_slots(dtype, alpha)
        launch(f"binary_{label}", [code, 0], out, a, b, alpha=m_alpha, alpha_i=m_alpha_i)
    else:
        raise Unsupported  # a CPU tensor operand that is not 0-dim
    return out


def _register_binary(name, overloads, inplace_overloads):
    def functional(a, b, alpha=1):
        return _binary(name, a, b, alpha)

    def inplace(a, b, alpha=1):
        return _binary(name, a, b, alpha, out=a)

    for o in overloads:
        _HANDLERS[o] = functional
    for o in inplace_overloads:
        _HANDLERS[o] = inplace


_register_binary("add", [aten.add.Tensor, aten.add.Scalar], [aten.add_.Tensor, aten.add_.Scalar])
_register_binary("sub", [aten.sub.Tensor, aten.sub.Scalar], [aten.sub_.Tensor, aten.sub_.Scalar])
_register_binary("mul", [aten.mul.Tensor, aten.mul.Scalar], [aten.mul_.Tensor, aten.mul_.Scalar])
_register_binary("div", [aten.div.Tensor, aten.div.Scalar], [aten.div_.Tensor, aten.div_.Scalar])
_register_binary("maximum", [aten.maximum.default], [])
_register_binary("minimum", [aten.minimum.default], [])
_register_binary("pow", [aten.pow.Tensor_Tensor, aten.pow.Tensor_Scalar], [aten.pow_.Tensor, aten.pow_.Scalar])
_register_binary("atan2", [aten.atan2.default], [aten.atan2_.default])
for _name in ("bitwise_and", "bitwise_or", "bitwise_xor"):
    _register_binary(_name, [getattr(aten, _name).Tensor, getattr(aten, _name).Scalar], [getattr(aten, _name + "_").Tensor, getattr(aten, _name + "_").Scalar])


@handles(aten.rsub.Scalar)
def _rsub(x, other, alpha=1):
    # other - alpha * x
    scaled = _binary("mul", x, -alpha)
    return _binary("add", scaled, other)


def _compare(name):
    code = COMPARE[name]

    def functional(a, b):
        if not isinstance(a, VkTensor):
            raise Unsupported
        dtype = _result_type(a, b)
        if dtype not in NUMERIC:
            raise Unsupported
        a = _cast(a, dtype)
        label = LABELS[dtype]
        if _is_scalar(b):
            out = empty(a.shape, torch.bool)
            alpha, alpha_i = _scalar_slots(dtype, _scalar_value(b))
            launch(f"compare_{label}", [code, 1], out, a, alpha=alpha, alpha_i=alpha_i)
        elif isinstance(b, VkTensor):
            b = _cast(b, dtype)
            out = empty(_broadcast_shape(a, b), torch.bool)
            launch(f"compare_{label}", [code, 0], out, a, b)
        else:
            raise Unsupported
        return out

    return functional


for _name in COMPARE:
    _fn = _compare(_name)
    _HANDLERS[getattr(aten, _name).Tensor] = _fn
    _HANDLERS[getattr(aten, _name).Scalar] = _fn
