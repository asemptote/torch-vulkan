"""Complex numbers and FFTs on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, check, make, make_complex, on_gpu


def fft_tol(dtype, n):
    eps = 2**-23 if dtype in (torch.complex64, torch.float32) else 2**-52
    return dict(rtol=0, atol=8 * eps * math.log2(max(n, 2)) * n**0.5 * 4)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_arithmetic(dtype):
    a, b = make_complex((7, 9), dtype, 1), make_complex((9,), dtype, 2)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        results = [va + vb, va - 2 * vb, va * vb, va / vb, -va, va * (1 + 2j), va + 3.5, torch.add(va, vb, alpha=0.5)]
    expected = [a + b, a - 2 * b, a * b, a / b, -a, a * (1 + 2j), a + 3.5, torch.add(a, b, alpha=0.5)]
    for got, want in zip(results, expected):
        assert got.dtype == want.dtype
        check(got, want)
    with on_gpu():
        mag = va.abs()
        conj = va.conj().clone()
        filled = torch.full((3,), 1 - 2j, dtype=dtype, device="vk")
    check(mag, a.abs())
    check(conj, a.conj())
    check(filled, torch.full((3,), 1 - 2j, dtype=dtype))
    check(torch.view_as_real(va), torch.view_as_real(a))
    check(va.real + va.imag, a.real + a.imag)


def test_complex_casts():
    x = make((5, 4), torch.float64)
    c = make_complex((5, 4), torch.complex64)
    with on_gpu():
        results = [x.to("vk").to(torch.complex128), c.to("vk").to(torch.complex128), c.to("vk").to(torch.complex64)]
    for got, want in zip(results, [x.to(torch.complex128), c.to(torch.complex128), c]):
        check(got, want)


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("n", [1, 2, 8, 64, 1024, 3, 5, 12, 100, 1000, 4097])
def test_fft_sizes(dtype, n):
    x = make_complex((3, n), dtype)
    with on_gpu():
        got = torch.fft.fft(x.to("vk"))
        back = torch.fft.ifft(got)
    check(got, torch.fft.fft(x), **fft_tol(dtype, n))
    check(back, x, **fft_tol(dtype, n))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [16, 100, 257])
def test_rfft_irfft(dtype, n):
    x = make((4, n), dtype)
    with on_gpu():
        spec = torch.fft.rfft(x.to("vk"))
        back = torch.fft.irfft(spec, n=n)
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    check(spec, torch.fft.rfft(x), **fft_tol(cdtype, n))
    check(back, x, **fft_tol(cdtype, n))


@pytest.mark.parametrize("norm", ["backward", "ortho", "forward"])
def test_fftn_dims_and_norms(norm):
    x = make_complex((6, 10, 16), torch.complex128)
    v = x.to("vk")
    with on_gpu():
        results = [torch.fft.fftn(v, norm=norm), torch.fft.fft2(v, norm=norm), torch.fft.ifft(v, dim=0, norm=norm),
                   torch.fft.fft(v.transpose(0, 2), dim=1, norm=norm)]
    expected = [torch.fft.fftn(x, norm=norm), torch.fft.fft2(x, norm=norm), torch.fft.ifft(x, dim=0, norm=norm),
                torch.fft.fft(x.transpose(0, 2), dim=1, norm=norm)]
    for got, want in zip(results, expected):
        check(got, want, **fft_tol(torch.complex128, 960))
    r = make((6, 10, 16), torch.float64)
    with on_gpu():
        got = torch.fft.irfftn(torch.fft.rfftn(r.to("vk"), norm=norm), s=r.shape, norm=norm)
    check(got, r, **fft_tol(torch.complex128, 960))


def test_fft_large_float64_accuracy():
    x = make_complex((1 << 16,), torch.complex128)
    with on_gpu():
        got = torch.fft.fft(x.to("vk")).cpu()
    err = (got - torch.fft.fft(x)).abs().max().item()
    assert err < 1e-11, err


def test_complex_views_alias():
    x = make_complex((4, 5), torch.complex64)
    v = x.to("vk")
    with on_gpu():
        v.real.add_(1.0)
        v.imag.mul_(2.0)
        r = torch.view_as_real(v)
        back = torch.view_as_complex(r)
    x.real.add_(1.0)
    x.imag.mul_(2.0)
    check(v, x)
    check(back, x)
    u = make((10,), torch.float32)
    with on_gpu():
        w = u.to("vk").unfold(0, 4, 2) * 1
    check(w, u.unfold(0, 4, 2))
