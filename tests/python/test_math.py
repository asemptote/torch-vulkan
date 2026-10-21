"""Transcendental and special functions on "vk" vs the CPU, float32 and float64.

float32 kernels use the GPU's built-ins (Vulkan precision rules) plus
shaders/_math32.slang, so they get float32-scale tolerances; float64 uses the
software library and is held to near-ULP agreement.
"""

import pytest
import torch
import torch.nn.functional as F

from _util import FLOATS, check, on_gpu


def uniform(n, lo, hi, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(n, dtype=dtype).uniform_(lo, hi, generator=g)


def tolerance(dtype, loose=False):
    if dtype == torch.float32:
        return dict(rtol=1e-4, atol=1e-4) if loose else dict(rtol=2e-5, atol=2e-5)
    return dict(rtol=1e-11, atol=1e-11) if loose else dict(rtol=1e-12, atol=1e-13)


UNARY_DOMAINS = {
    "asin": (-1, 1), "acos": (-1, 1), "atan": (-20, 20), "sinh": (-10, 10), "cosh": (-10, 10),
    "asinh": (-100, 100), "acosh": (1, 100), "atanh": (-0.99, 0.99), "log10": (1e-3, 1e3),
    "log1p": (-0.9, 10), "expm1": (-10, 10), "erf": (-5, 5), "erfc": (-3, 9), "erfinv": (-0.99, 0.99),
    "lgamma": (-9.9, 30), "digamma": (-9.9, 30), "sinc": (-10, 10), "frac": (-10, 10),
}
LOOSE = {"lgamma", "digamma", "erfinv", "sinh", "cosh"}


@pytest.mark.parametrize("name,dtype", [(n, d) for n in UNARY_DOMAINS for d in FLOATS])
def test_unary_math(name, dtype):
    x = uniform(4096, *UNARY_DOMAINS[name], dtype)
    op = getattr(torch, name)
    with on_gpu():
        got = op(x.to("vk"))
    want = op(x)
    if name in ("sinh", "cosh"):
        check(got, want, rtol=tolerance(dtype, True)["rtol"], atol=0)
    else:
        check(got, want, **tolerance(dtype, name in LOOSE))


ACTIVATIONS = {
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
    "silu": F.silu,
    "mish": F.mish,
    "softplus": lambda x: F.softplus(x, beta=2.0, threshold=10.0),
    "elu": lambda x: F.elu(x, alpha=0.7),
    "selu": F.selu,
    "logsigmoid": F.logsigmoid,
    "logit": lambda x: torch.logit(torch.sigmoid(x), eps=1e-3),
    "hardtanh": lambda x: F.hardtanh(x, -0.5, 0.75),
    "leaky_relu": lambda x: F.leaky_relu(x, 0.1),
    "softshrink": lambda x: F.softshrink(x, 0.4),
    "hardshrink": lambda x: F.hardshrink(x, 0.4),
    "threshold": lambda x: F.threshold(x, 0.2, -3.0),
}


@pytest.mark.parametrize("name,dtype", [(n, d) for n in ACTIVATIONS for d in FLOATS])
def test_activation(name, dtype):
    x = uniform(4096, -12, 12, dtype)
    op = ACTIVATIONS[name]
    with on_gpu():
        got = op(x.to("vk"))
    check(got, op(x), **tolerance(dtype, loose=True))


BINARY = {
    "pow": (lambda a, b: torch.pow(a.abs() + 0.1, b), True),
    "atan2": (torch.atan2, True),
    "logaddexp": (torch.logaddexp, False),
    "logaddexp2": (torch.logaddexp2, False),
    "xlogy": (lambda a, b: torch.xlogy(a, b.abs()), False),
    "xlog1py": (lambda a, b: torch.special.xlog1py(a, b.abs()), False),
    "hypot": (torch.hypot, False),
    "fmod": (torch.fmod, False),
    "remainder": (torch.remainder, False),
    "floor_divide": (torch.floor_divide, False),
    "copysign": (torch.copysign, False),
    "nextafter": (torch.nextafter, False),
}


@pytest.mark.parametrize("name,dtype", [(n, d) for n in BINARY for d in FLOATS])
def test_binary_math(name, dtype):
    op, loose = BINARY[name]
    a, b = uniform(4096, -8, 8, dtype, 1), uniform(4096, -8, 8, dtype, 2)
    with on_gpu():
        got = op(a.to("vk"), b.to("vk"))
    exact = name in ("fmod", "remainder", "floor_divide", "copysign", "nextafter")
    check(got, op(a, b), **(dict(rtol=0, atol=0) if exact else tolerance(dtype, loose)))


@pytest.mark.parametrize("dtype", FLOATS)
def test_pow_special_values(dtype):
    vals = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, float("inf"), float("-inf"), float("nan")], dtype=dtype)
    a, b = torch.meshgrid(vals, vals, indexing="ij")
    with on_gpu():
        got = torch.pow(a.to("vk"), b.to("vk"))
    check(got, torch.pow(a, b), equal_nan=True)


@pytest.mark.parametrize("dtype", FLOATS)
def test_atan2_special_values(dtype):
    vals = torch.tensor([0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan")], dtype=dtype)
    a, b = torch.meshgrid(vals, vals, indexing="ij")
    with on_gpu():
        got = torch.atan2(a.to("vk"), b.to("vk"))
    check(got, torch.atan2(a, b), equal_nan=True, **tolerance(dtype))

