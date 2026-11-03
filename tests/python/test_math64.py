"""float64 math library accuracy (ULP error vs the CPU)."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, check, make, make_complex, on_gpu

pytestmark = pytest.mark.float64


F64_DOMAINS = {
    "exp": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-745, 710, generator=g),
    "exp2": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1075, 1024, generator=g),
    "log": lambda g: torch.exp(torch.empty(200_000, dtype=torch.float64).uniform_(-740, 709, generator=g)),
    "log2": lambda g: torch.exp(torch.empty(200_000, dtype=torch.float64).uniform_(-740, 709, generator=g)),
    "sin": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e5, 1e5, generator=g),
    "cos": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e5, 1e5, generator=g),
    "tan": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e3, 1e3, generator=g),
    "tanh": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-25, 25, generator=g),
    "sigmoid": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-50, 50, generator=g),
}
F64_SPECIALS = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-310, 5e-324, 1e300, -1e300, 709.8, -745.2,
                             float("inf"), float("-inf"), float("nan")], dtype=torch.float64)


def ulp_error(got, want):
    """|got - want| in units of want's last place (0 where both are equal/NaN)."""
    same = (got == want) | (got.isnan() & want.isnan())
    ulp = torch.nextafter(want.abs(), torch.tensor(float("inf"), dtype=torch.float64)) - want.abs()
    err = (got - want).abs() / ulp
    return torch.where(same, torch.zeros_like(err), err)


@pytest.mark.parametrize("name", list(F64_DOMAINS))
def test_float64_math_accuracy(name):
    x = torch.cat([F64_DOMAINS[name](torch.Generator().manual_seed(0)), F64_SPECIALS])
    op = getattr(torch, name)
    with on_gpu():
        got = op(x.to("vk")).cpu()
    want = op(x)
    # Special values must match exactly (inf, nan, 0, overflow/underflow).
    special = ~torch.isfinite(want) | (want == 0) | ~torch.isfinite(x)
    assert torch.equal(got[special].isnan(), want[special].isnan())
    assert torch.equal(got[special & ~want.isnan()], want[special & ~want.isnan()])
    err = ulp_error(got, want)
    if name in ("sin", "cos", "tan"):
        # Near zeros of sin/cos (and poles of tan) the result is tiny/huge and its
        # ULP meaningless; bound the error relative to the function's scale instead.
        scale = want.abs().clamp(min=1.0) ** (2 if name == "tan" else 1)
        err = torch.minimum(err, (got - want).abs() / (scale * 2.0**-52))
    worst = err[torch.isfinite(err)].max().item()
    assert worst <= 4, f"{name}: max error {worst:.2f} ULP"
