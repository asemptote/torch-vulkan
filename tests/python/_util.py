"""Shared helpers for the torch_vulkan tests (imported by every test module)."""

import contextlib
import math

import pytest
import torch

import torch_vulkan as tv

NUMERIC = [torch.float32, torch.float64, torch.int32, torch.int64]
ALL = NUMERIC + [torch.bool]
FLOATS = [torch.float32, torch.float64]
COMPLEX = [torch.complex64, torch.complex128]

# GPUs without float64 arithmetic (shaderFloat64) hold no float64 tensors
# (conftest.py skips the tests of that dtype there); WIDE is the widest float.
HAS_FLOAT64 = bool(tv._native.capabilities() & 1)
WIDE = torch.float64 if HAS_FLOAT64 else torch.float32


@contextlib.contextmanager
def on_gpu():
    """Fails if anything inside falls back to the CPU."""
    before = tv.fallback_counts()
    yield
    after = tv.fallback_counts()
    new = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    assert not new, f"ran on the CPU: {new}"


def make(shape, dtype, seed=0, positive=False):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if dtype.is_floating_point:
        x = torch.randn(shape, generator=g, dtype=dtype)
        return x.abs() + 0.1 if positive else x
    lo = 1 if positive else -50
    return torch.randint(lo, 50, shape, generator=g, dtype=dtype)


def make_complex(shape, dtype, seed=0):
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    return torch.complex(make(shape, real, seed), make(shape, real, seed + 100))


def check(vk_result, cpu_result, **tol):
    assert vk_result.device.type == "vk", vk_result.device
    torch.testing.assert_close(vk_result.cpu(), cpu_result, **tol)
