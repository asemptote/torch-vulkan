"""torch_vulkan: run PyTorch ops on a Vulkan GPU as device "vk".

    import torch, torch_vulkan
    x = torch.randn(1000, device="vk")
    y = (x * 2 + 1).relu()
    print(y.cpu())

Importing this module registers a PyTorch backend on the PrivateUse1 key
(implemented in C++, see crates/pyvk/csrc) and names it "vk". Element-wise
ops, casts, copies and fills run as Vulkan kernels; every other op runs on
the CPU with a one-time warning (see `fallback_counts()`).
"""

import atexit
import sys
import types

import torch

# "vk" rather than "vulkan": PyTorch's built-in "vulkan" device type belongs
# to its old, unmaintained mobile backend.
torch.utils.rename_privateuse1_backend("vk")

from . import _native  # noqa: E402  (registers the C++ backend)

torch.utils.generate_methods_for_privateuse1_backend()

__all__ = [
    "device_count",
    "device_name",
    "empty_cache",
    "fallback_counts",
    "is_available",
    "memory_stats",
    "synchronize",
]


def is_available():
    try:
        _native.device_name()
        return True
    except RuntimeError:
        return False


def device_count():
    return 1 if is_available() else 0


def device_name():
    return _native.device_name()


def synchronize(device=None):
    """Waits for all queued GPU work."""
    _native.synchronize()


def memory_stats():
    allocated, reserved, blocks = _native.memory_stats()
    return {"allocated_bytes": allocated, "reserved_bytes": reserved, "blocks": blocks}


def empty_cache():
    """Returns cached, unused device memory to the driver."""
    _native.empty_cache()


def fallback_counts():
    """{op: count} of aten ops that ran on the CPU instead of the GPU."""
    return dict(_native.fallback_counts())


def _device_module():
    # torch.vk: what PyTorch expects of a backend's device module.
    mod = types.ModuleType("torch.vk")
    mod.is_available = is_available
    mod.device_count = device_count
    mod.current_device = lambda: 0
    mod.set_device = lambda device: None
    mod.synchronize = synchronize
    mod.is_initialized = lambda: True
    mod._is_in_bad_fork = lambda: False
    mod.manual_seed = lambda seed: None
    mod.manual_seed_all = lambda seed: None
    mod.get_amp_supported_dtype = lambda: [torch.float32]
    mod._lazy_init = lambda: None
    return mod


torch._register_device_module("vk", _device_module())
sys.modules["torch.vk"] = torch.vk


@atexit.register
def _shutdown():
    # Finish queued work and save the pipeline cache before interpreter teardown.
    _native.shutdown()
