"""Special functions on "vk" vs the CPU (M14): shaders/special.slang is a port
of the CPU's implementations, so results agree to a few ulp."""

import pytest
import torch

from _util import FLOATS, check, make, on_gpu

UNARY = ["i0", "special.i0e", "special.i1", "special.i1e", "special.ndtri", "special.erfcx", "special.log_ndtr",
         "special.bessel_j0", "special.bessel_j1", "special.bessel_y0", "special.bessel_y1",
         "special.modified_bessel_i0", "special.modified_bessel_i1", "special.modified_bessel_k0",
         "special.modified_bessel_k1", "special.scaled_modified_bessel_k0", "special.scaled_modified_bessel_k1",
         "special.spherical_bessel_j0", "special.airy_ai"]
POLYNOMIALS = ["chebyshev_polynomial_t", "chebyshev_polynomial_u", "chebyshev_polynomial_v", "chebyshev_polynomial_w",
               "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u", "shifted_chebyshev_polynomial_v",
               "shifted_chebyshev_polynomial_w", "hermite_polynomial_h", "hermite_polynomial_he",
               "laguerre_polynomial_l", "legendre_polynomial_p"]


def fn(name):
    f = torch
    for part in name.split("."):
        f = getattr(f, part)
    return f


def tol(dtype):
    return dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-13)


def inputs(dtype):
    x = torch.cat([make((200,), dtype) * 4, torch.linspace(-30, 30, 121, dtype=dtype),
                   torch.tensor([0.0, 1e-6, -1e-6, 0.5, 1.0, 8.0, 8.5, 25.0, 100.0, float("inf"), float("-inf"),
                                 float("nan")], dtype=dtype)])
    return x


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", UNARY)
def test_unary(dtype, name):
    x = inputs(dtype)
    if name == "special.ndtri":
        x = torch.cat([torch.rand(200, generator=torch.Generator().manual_seed(0), dtype=dtype),
                       torch.tensor([0.0, 1.0, 1e-30, 0.5, -0.1, 1.1], dtype=dtype)])
    f = fn(name)
    with on_gpu():
        got = f(x.to("vk"))
    t = tol(dtype)
    if name == "special.airy_ai" and dtype == torch.float32:
        # For large negative x, Ai oscillates with phase (2/3)|x|^1.5 (~110 at
        # x = -30): float32 rounding of the phase alone is ~1e-5 on any device.
        t = dict(rtol=1e-5, atol=1e-5)
    check(got, f(x), equal_nan=True, **t)


def test_unary_integer_input_and_inplace():
    x = torch.arange(-5, 6)
    with on_gpu():
        got = torch.special.bessel_j0(x.to("vk"))
        y = torch.linspace(0, 3, 7).to("vk")
        y.i0_()
    check(got, torch.special.bessel_j0(x))
    check(y, torch.linspace(0, 3, 7).i0())


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("name", POLYNOMIALS)
def test_polynomials(dtype, name):
    x = torch.cat([torch.linspace(-1.5, 1.5, 61, dtype=dtype), torch.tensor([float("nan")], dtype=dtype)])
    n = torch.arange(-1, 12, dtype=dtype).view(-1, 1)
    f = getattr(torch.special, name)
    with on_gpu():
        got = f(x.to("vk"), n.to("vk"))
    check(got, f(x, n), equal_nan=True, rtol=1e-4 if dtype == torch.float32 else 1e-10, atol=1e-4)


@pytest.mark.parametrize("dtype", FLOATS)
def test_zeta_igamma_polygamma(dtype):
    g = torch.Generator().manual_seed(0)
    a = torch.rand(300, generator=g, dtype=dtype) * 30
    x = torch.rand(300, generator=g, dtype=dtype) * 30
    a[:3] = torch.tensor([0.0, 25.0, 250.0], dtype=dtype)
    x[:3] = torch.tensor([1.0, 26.0, 249.0], dtype=dtype)
    q = torch.rand(300, generator=g, dtype=dtype) * 5 + 0.1
    s = torch.rand(300, generator=g, dtype=dtype) * 6 + 1.0
    va, vx, vq, vs = (t.to("vk") for t in (a, x, q, s))
    with on_gpu():
        got = [torch.igamma(va, vx), torch.igammac(va, vx), torch.special.zeta(vs, vq),
               *[torch.polygamma(n, vq) for n in range(5)], torch.special.zeta(vs, 2.0)]
    want = [torch.igamma(a, x), torch.igammac(a, x), torch.special.zeta(s, q), *[torch.polygamma(n, q) for n in range(5)],
            torch.special.zeta(s, 2.0)]
    for gt, w in zip(got, want):
        check(gt, w, equal_nan=True, rtol=1e-4 if dtype == torch.float32 else 1e-11, atol=1e-6)
