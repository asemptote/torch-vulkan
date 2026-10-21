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
