"""Runs a PyTorch test with every torch.manual_seed(s) shifted by OFFSET.

    python tools/dev/seedsweep.py DEVICE OFFSET test_nn.py -k EXPR

DEVICE is vk or cpu. For telling a vk bug from a tolerance tuned to one seed:
if the CPU also fails for other seeds, the tolerance is the problem.
"""
import os
import sys

dev, offset = sys.argv[1], int(sys.argv[2])
os.environ["PYTORCH_TESTING_DEVICE_ONLY_FOR"] = dev
import pytest  # noqa: E402
import torch  # noqa: E402

if dev == "vk":
    import torch_vulkan  # noqa: F401

_seed = torch.manual_seed
torch.manual_seed = lambda s: _seed(s + offset)
os.chdir(os.environ.get("PYTORCH_TEST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                       "pytorch-src", "test")))
sys.path.insert(0, os.getcwd())
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", "--tb=line", "-W", "ignore"] + sys.argv[3:]))
