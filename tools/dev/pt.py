"""Runs one of PyTorch's test files on device "vk" with full tracebacks.

    python tools/dev/pt.py test_linalg.py -k "inverse and complex" [pytest options]

Stops at the first failure (add --maxfail=N to go on). PyTorch's tests are
taken from $PYTORCH_TEST_DIR (default ~/pytorch-src/test); see
tools/pytorch_tests.py for running whole files with the CPU re-check.
"""
import os
import sys

os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "vk")
import pytest  # noqa: E402
import torch  # noqa: E402,F401
import torch_vulkan  # noqa: E402,F401

os.chdir(os.environ.get("PYTORCH_TEST_DIR", os.path.expanduser("~/pytorch-src/test")))
sys.path.insert(0, os.getcwd())
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", "-x", "--tb=short"] + sys.argv[1:]))
