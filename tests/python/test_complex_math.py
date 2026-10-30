"""Complex math, matrix products and reductions on "vk" vs the CPU (M14)."""

import pytest
import torch

from _util import COMPLEX, check, make, make_complex, on_gpu

UNARY = ["exp", "log", "log2", "log10", "log1p", "expm1", "sqrt", "rsqrt", "reciprocal", "square", "sin", "cos",
         "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "sigmoid"]


def tol(dtype, scale=1.0):
    return dict(rtol=2e-5 * scale, atol=2e-5 * scale) if dtype == torch.complex64 else dict(rtol=1e-12 * scale,
                                                                                            atol=1e-12 * scale)


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("name", UNARY)
def test_complex_unary(dtype, name):
    a = make_complex((5, 40), dtype, 3)
    a[0, :4] = torch.tensor([0, 1, -1, 1j], dtype=dtype)
    fn = getattr(torch, name)
    with on_gpu():
        got = fn(a.to("vk"))
    check(got, fn(a), **tol(dtype, 10), equal_nan=True)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_pow(dtype):
    a, b = make_complex((6, 7), dtype, 1), make_complex((7,), dtype, 2)
    with on_gpu():
        got = [torch.pow(a.to("vk"), b.to("vk")), a.to("vk") ** 2.5, a.to("vk") ** (1 - 1j)]
    for g, w in zip(got, [torch.pow(a, b), a**2.5, a ** (1 - 1j)]):
        check(g, w, **tol(dtype, 10))


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_matmul(dtype):
    a, b = make_complex((4, 9, 70), dtype, 1), make_complex((4, 70, 5), dtype, 2)
    c, v = make_complex((4, 9, 5), dtype, 3), make_complex((70,), dtype, 4)
    va, vb, vc, vv = (x.to("vk") for x in (a, b, c, v))
    with on_gpu():
        got = [va @ vb, torch.baddbmm(vc, va, vb, beta=0.5, alpha=2 - 1j), va[0] @ vv, va[0].conj() @ vb[0],
               va[1] @ vb[1].mH.mH]
    want = [a @ b, torch.baddbmm(c, a, b, beta=0.5, alpha=2 - 1j), a[0] @ v, a[0].conj() @ b[0], a[1] @ b[1]]
    for g, w in zip(got, want):
        check(g, w, **tol(dtype, 100))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_matmul(dtype):
    a, b = make((3, 17, 33), dtype, 1), make((3, 33, 6), dtype, 2)
    with on_gpu():
        got = a.to("vk") @ b.to("vk")
    check(got, a @ b)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_reductions(dtype):
    a = make_complex((6, 5, 7), dtype, 1)
    va = a.to("vk")
    with on_gpu():
        got = [va.sum(), va.sum(1), va.sum((0, 2), keepdim=True), va.mean(), va.mean(-1), va.prod(), va.prod(1),
               va.prod(2, keepdim=True), va.var(), va.std(1), va.var(0, correction=0), torch.linalg.vector_norm(va),
               torch.linalg.vector_norm(va, 1, dim=1), va.cumsum(1), va.cumprod(2), va.conj().sum(0)]
    want = [a.sum(), a.sum(1), a.sum((0, 2), keepdim=True), a.mean(), a.mean(-1), a.prod(), a.prod(1),
            a.prod(2, keepdim=True), a.var(), a.std(1), a.var(0, correction=0), torch.linalg.vector_norm(a),
            torch.linalg.vector_norm(a, 1, dim=1), a.cumsum(1), a.cumprod(2), a.conj().sum(0)]
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        check(g, w, **tol(dtype, 100))
