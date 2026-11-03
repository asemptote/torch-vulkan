"""Session-wide checks for the torch_vulkan tests.

Run with TORCH_VK_VALIDATE=1 to also fail on any Vulkan validation message.
"""

import pytest

from torch_vulkan import _native


@pytest.fixture(scope="session", autouse=True)
def validation_clean():
    yield
    _native.synchronize()
    assert _native.validation_issues() == 0, "Vulkan validation layer reported problems (see stderr)"


def _has_float64():
    return bool(_native.capabilities() & 1)


def pytest_configure(config):
    config.addinivalue_line("markers", "float64: needs a GPU with float64 arithmetic (shaderFloat64)")


def pytest_collection_modifyitems(config, items):
    """On GPUs without float64 (where vk float64 / complex128 tensors are a
    TypeError, as on MPS), skip the tests of those dtypes: parametrized with
    one, or marked `float64`."""
    if _has_float64():
        return
    import torch

    wide = (torch.float64, torch.complex128)
    skip = pytest.mark.skip(reason="this GPU has no float64 arithmetic")
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        values = [v for p in params.values() for v in (p if isinstance(p, (tuple, list)) else (p,))]
        if any(v in wide for v in values if isinstance(v, torch.dtype)) or item.get_closest_marker("float64"):
            item.add_marker(skip)
