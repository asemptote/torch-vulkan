"""torch_vulkan: run PyTorch ops on a Vulkan GPU as device "vk".

    import torch, torch_vulkan
    x = torch.randn(1000, device="vk")
    y = (x * 2 + 1).relu()
    print(y.cpu())

Importing this module registers a PyTorch backend on the PrivateUse1 key
(implemented in C++, see crates/pyvk/csrc) and names it "vk". Element-wise
op with a Vulkan kernel runs on the GPU, many others through PyTorch's
decompositions into such ops (see _decomp.py). An op with no Vulkan kernel
for its arguments raises NotImplementedError, unless the CPU fallback is on
(TORCH_VK_CPU_FALLBACK=1 or `set_cpu_fallback(True)`): then it runs on the
CPU, with a one-time warning (see `fallback_counts()`).
"""

import atexit
import importlib.metadata
import sys
import types

import torch

try:
    __version__ = importlib.metadata.version("torch-vulkan")
except importlib.metadata.PackageNotFoundError:  # a source tree on the path, not installed
    __version__ = "unknown"


def _check_torch_build():
    """Termux builds (tools/build-termux.sh, and the releases' Android
    wheels) record the torch they were compiled against in _build_info.py.
    libtorch's C++ ABI changes between releases, so after a `pkg upgrade` of
    python-torch the module could fail to load or crash; refuse it here
    instead. (pip keeps the other wheels' torch within one minor release.)"""
    try:
        from ._build_info import TORCH_VERSION
    except ImportError:
        return
    installed = torch.__version__.split("+")[0]
    if installed != TORCH_VERSION:
        raise ImportError(
            f"torch_vulkan {__version__} was compiled against PyTorch {TORCH_VERSION}, but PyTorch "
            f"{installed} is installed (on Termux, python-torch was upgraded?). Install the torch-vulkan "
            f"build for PyTorch {installed} (see the README's 'Installing'), or rebuild with "
            "tools/build-termux.sh.")


_check_torch_build()

# "vk" rather than "vulkan": PyTorch's built-in "vulkan" device type belongs
# to its old, unmaintained mobile backend.
torch.utils.rename_privateuse1_backend("vk")

from . import _native  # noqa: E402  (registers the C++ backend)

torch.utils.generate_methods_for_privateuse1_backend()

from . import _decomp  # noqa: E402

# Ops with no Vulkan kernel of their own run through PyTorch's decompositions
# (as sequences of Vulkan kernels) rather than on the CPU.
_decomposed = _decomp.register()

from . import _kernels  # noqa: E402

# Python kernels (from Vulkan ops) for ops with neither a C++ kernel nor a
# decomposition: average pooling, adaptive pooling, upsampling backward.
_python_kernels = _kernels.register()

from . import _nofloat64  # noqa: E402

# Composite ops whose PyTorch implementations make float64 temporaries
# (quantile, pinv, matrix_rank, zeta with a number, batch norm of an empty
# input), for GPUs without float64.
_python_kernels += _nofloat64.register()

__all__ = [
    "device_count",
    "device_name",
    "cpu_fallback_enabled",
    "empty_cache",
    "fallback_counts",
    "is_available",
    "memory_stats",
    "set_cpu_fallback",
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


def _warn_llvmpipe_on_android():
    """On Android, Termux's Vulkan loader finds only Mesa's llvmpipe (the
    CPU), whose 32-bit ARM shaders fault with SIGBUS in reductions and
    softmax (an aligned NEON store to 8-byte aligned memory): say so at
    import, before the first sum kills the process."""
    if not (sys.platform == "android" or hasattr(sys, "getandroidapilevel")):
        return
    try:
        name = _native.device_name()
    except RuntimeError:
        return
    if name.startswith("llvmpipe"):
        import warnings

        warnings.warn(
            "torch_vulkan: Vulkan found only llvmpipe (Mesa's CPU driver), not the phone's GPU; on 32-bit "
            "Android its shaders crash (SIGBUS) in reductions such as sum and softmax. Use Android's own "
            "Vulkan loader: `pkg install vulkan-loader-android`, or put a libvulkan.so linked to "
            "/system/lib64/libvulkan.so (/system/lib on 32-bit) first on LD_LIBRARY_PATH (see the README's "
            "'Installing').", RuntimeWarning, stacklevel=2)


_warn_llvmpipe_on_android()


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
    """{op: count} of aten ops that ran on the CPU instead of the GPU (as
    "op (raised)" where the CPU only raised, a dtype or argument check, say,
    and "op (refused)" where the CPU fallback was off and the op raised
    NotImplementedError instead)."""
    return dict(_native.fallback_counts())


def set_cpu_fallback(enabled):
    """Lets ops with no Vulkan kernel for their arguments run on the CPU
    (True), with a one-time warning per op and counted in fallback_counts(),
    or makes them raise NotImplementedError (False, the default), for the
    whole process. TORCH_VK_CPU_FALLBACK=1 in the environment turns it on at
    start-up."""
    _native.set_cpu_fallback(bool(enabled))


def cpu_fallback_enabled():
    """Whether ops with no Vulkan kernel for their arguments may run on the
    CPU (see set_cpu_fallback)."""
    return _native.cpu_fallback_enabled()


def manual_seed(seed):
    """Seeds the "vk" random generator (torch.manual_seed does this too)."""
    _native.manual_seed(int(seed) & 0xFFFF_FFFF_FFFF_FFFF)


def seed():
    """Seeds the "vk" generator nondeterministically and returns the seed."""
    return _native.seed()


def initial_seed():
    return _native.initial_seed()


def get_rng_state(device="vk"):
    """The "vk" generator state as a uint8 tensor (seed, offset)."""
    s, o = _native.get_rng_state()
    return torch.tensor(list(s.to_bytes(8, "little") + o.to_bytes(8, "little")), dtype=torch.uint8)


def set_rng_state(state, device="vk"):
    data = bytes(state.to(torch.uint8).tolist())
    _native.set_rng_state(int.from_bytes(data[:8], "little"), int.from_bytes(data[8:16], "little"))


class _DeviceContext:
    """torch.vk.device(d): a context manager selecting device d (there is one)."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _device_module():
    # torch.vk: what PyTorch expects of a backend's device module.
    mod = types.ModuleType("torch.vk")
    mod.is_available = is_available
    mod.device_count = device_count
    mod.current_device = lambda: 0
    mod.device = _DeviceContext
    mod.set_device = lambda device: None
    mod.synchronize = synchronize
    mod.is_initialized = lambda: True
    mod._is_in_bad_fork = lambda: False
    mod.manual_seed = manual_seed
    mod.manual_seed_all = manual_seed
    mod.seed = seed
    mod.seed_all = seed
    mod.initial_seed = initial_seed
    mod.get_rng_state = get_rng_state
    mod.set_rng_state = set_rng_state
    mod.get_rng_state_all = lambda: [get_rng_state()]
    mod.set_rng_state_all = lambda states: set_rng_state(states[0])
    # (torch.autocast("vk"): CUDA's policy, csrc/autocast.cpp; float16 by default)
    mod.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16, torch.float32]
    mod._lazy_init = lambda: None
    return mod


torch._register_device_module("vk", _device_module())
sys.modules["torch.vk"] = torch.vk

# Printing a float tensor converts it to double on its device
# (torch._tensor_str.tensor_totype); GPUs without float64 print in float32,
# as PyTorch does for MPS.
_tensor_totype = torch._tensor_str.tensor_totype


def _vk_tensor_totype(t):
    if t.device.type == "vk" and not _kernels._native_has_float64():
        return t.to(dtype=torch.float)
    return _tensor_totype(t)


torch._tensor_str.tensor_totype = _vk_tensor_totype


@atexit.register
def _shutdown():
    # Finish queued work and save the pipeline cache before interpreter teardown.
    _native.shutdown()
