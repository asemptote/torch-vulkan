"""torch_vulkan: run PyTorch ops on a Vulkan GPU as device "vk".

    import torch, torch_vulkan
    x = torch.randn(1000, device="vk")
    y = (x * 2 + 1).relu()
    print(y.cpu())

Prototype (M2): tensors are `VkTensor` subclass instances; element-wise ops,
casts, copies and fills run as Vulkan kernels, everything else runs on the
CPU with a one-time warning per op (see `fallback_counts()`).
"""

import atexit

import torch
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend

# Device "vk" (PyTorch's built-in "vulkan" device type belongs to its old,
# unmaintained mobile backend). This also registers the device guard, hooks
# and device module that autograd and factory functions expect.
_setup_privateuseone_for_python_backend("vk")

from . import _native  # noqa: E402
from ._tensor import VkTensor, empty, fallback_counts, from_cpu, to_cpu, to_vk  # noqa: E402

__all__ = [
    "VkTensor",
    "device_name",
    "empty",
    "empty_cache",
    "fallback_counts",
    "from_cpu",
    "memory_stats",
    "synchronize",
    "to_cpu",
    "to_vk",
]


def device_name():
    return _native.device_name()


def synchronize():
    """Waits for all queued GPU work."""
    _native.synchronize()


def memory_stats():
    allocated, reserved, blocks = _native.memory_stats()
    return {"allocated_bytes": allocated, "reserved_bytes": reserved, "blocks": blocks}


def empty_cache():
    """Returns cached, unused device memory to the driver."""
    _native.empty_cache()


# Factory functions with device="vk" reach PrivateUse1 kernels. `empty` and
# `empty_strided` create VkTensors directly; ops like zeros/ones/randn are
# built from them plus in-place ops that VkTensor dispatch handles. Factories
# that fill an `out=` tensor of unknown size (arange, linspace, eye) are
# computed on the CPU and uploaded.
_lib = torch.library.Library("aten", "IMPL")


def _factory_empty(size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    return empty(size, dtype)


def _factory_empty_strided(size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    from ._tensor import empty_strided

    return empty_strided(size, stride, dtype or torch.get_default_dtype())


def _via_cpu(op):
    def factory(*args, **kwargs):
        kwargs["device"] = torch.device("cpu")
        return from_cpu(op(*args, **kwargs))

    return factory


def _routed(op):
    # Ops that reach the vk backend key without passing through VkTensor's
    # Python dispatch (e.g. C++ copy_ calling _copy_from) get the same handling.
    def kernel(*args, **kwargs):
        return VkTensor.__torch_dispatch__(op, (), args, kwargs)

    return kernel


for _name, _op in [
    ("_copy_from", torch.ops.aten._copy_from.default),
    ("_copy_from_and_resize", torch.ops.aten._copy_from_and_resize.default),
]:
    _lib.impl(_name, _routed(_op), "PrivateUse1")

_lib.impl("empty.memory_format", _factory_empty, "PrivateUse1")
_lib.impl("empty_strided", _factory_empty_strided, "PrivateUse1")
for _name, _op in [
    ("arange", torch.ops.aten.arange.default),
    ("arange.start", torch.ops.aten.arange.start),
    ("arange.start_step", torch.ops.aten.arange.start_step),
    ("linspace", torch.ops.aten.linspace.default),
    ("eye", torch.ops.aten.eye.default),
    ("eye.m", torch.ops.aten.eye.m),
]:
    _lib.impl(_name, _via_cpu(_op), "PrivateUse1")


@atexit.register
def _shutdown():
    # Finish queued work and save the pipeline cache before interpreter teardown.
    _native.shutdown()
