"""Dense linear algebra on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, WIDE, check, make, make_complex, on_gpu


LA_SIZES = [130, 200, 333]  # >= 128 takes the blocked GPU path; 333 is not a multiple of 64


def residual(a, x, b):
    """||A X - B|| / (||A|| ||X||): backward error of a solve."""
    return ((a @ x - b).norm() / (a.norm() * x.norm())).item()


def la_eps(dtype):
    return 1e-12 if dtype == torch.float64 else 1e-4


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("n", LA_SIZES)
def test_lu_factor_solve_inv(dtype, n):
    a, b = make((n, n), dtype, 1), make((n, 3), dtype, 2)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        lu, piv = torch.linalg.lu_factor(va)
        x = torch.linalg.solve(va, vb)
        xv = torch.linalg.solve(va, vb[:, 0])
        inv = torch.linalg.inv(va)
        x2 = torch.linalg.lu_solve(lu, piv, vb)
    wlu, wpiv = torch.linalg.lu_factor(a)
    assert torch.equal(piv.cpu(), wpiv), "same partial-pivoting decisions as LAPACK"
    check(lu, wlu, rtol=1e-8 if dtype == torch.float64 else 1e-3, atol=1e-8 if dtype == torch.float64 else 1e-3)
    assert residual(a, x.cpu(), b) < la_eps(dtype)
    assert residual(a, xv.cpu().unsqueeze(1), b[:, :1]) < la_eps(dtype)
    assert residual(a, x2.cpu(), b) < la_eps(dtype)
    assert residual(a, inv.cpu(), torch.eye(n, dtype=dtype)) < la_eps(dtype)


@pytest.mark.parametrize("dtype", FLOATS)
def test_det_slogdet(dtype):
    a = make((200, 200), dtype, 3) / 10 + torch.eye(200, dtype=dtype)
    with on_gpu():
        d = torch.linalg.det(a.to("vk"))
        sign, logabs = torch.linalg.slogdet(a.to("vk"))
    tol = dict(rtol=1e-9, atol=0) if dtype == torch.float64 else dict(rtol=1e-3, atol=0)
    check(d, torch.linalg.det(a), **tol)
    ws, wl = torch.linalg.slogdet(a)
    check(sign, ws)
    check(logabs, wl, **tol)


def test_singular_matrix_reports_info():
    a = make((150, 150), WIDE)
    a[:, 7] = 0  # singular
    with on_gpu():
        lu, piv, info = torch.linalg.lu_factor_ex(a.to("vk"))
    assert info.item() > 0
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.inv(a.to("vk"))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("n", LA_SIZES)
def test_cholesky(dtype, n):
    m = make((n, n), dtype, 4)
    spd = m @ m.T + n * torch.eye(n, dtype=dtype)
    with on_gpu():
        l = torch.linalg.cholesky(spd.to("vk"))
        u = torch.linalg.cholesky(spd.to("vk"), upper=True)
    tol = dict(rtol=1e-10, atol=1e-10) if dtype == torch.float64 else dict(rtol=1e-4, atol=1e-4)
    check(l, torch.linalg.cholesky(spd), **tol)
    check(u, torch.linalg.cholesky(spd, upper=True), **tol)
    bad = spd.clone()
    bad[100, 100] = -1e6
    with on_gpu():
        _, info = torch.linalg.cholesky_ex(bad.to("vk"))
    assert info.item() == torch.linalg.cholesky_ex(bad)[1].item() > 0


@pytest.mark.float64
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("unit", [False, True])
def test_solve_triangular(upper, unit):
    n = 333
    t = make((n, n), torch.float64, 5) / n + torch.eye(n, dtype=torch.float64) * 2
    t = t.triu() if upper else t.tril()
    b = make((n, 4), torch.float64, 6)
    with on_gpu():
        x = torch.linalg.solve_triangular(t.to("vk"), b.to("vk"), upper=upper, unitriangular=unit)
    check(x, torch.linalg.solve_triangular(t, b, upper=upper, unitriangular=unit), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", NUMERIC + [torch.bool, torch.complex64])
@pytest.mark.parametrize("diagonal", [0, 2, -3])
def test_tril_triu(dtype, diagonal):
    x = make((3, 7, 9), dtype) if not dtype.is_complex else make_complex((3, 7, 9), dtype)
    v = x.to("vk")
    with on_gpu():
        results = [v.tril(diagonal), v.triu(diagonal), v.transpose(1, 2).triu(diagonal)]
        w = v.clone()
        w.tril_(diagonal)
    for got, want in zip(results + [w], [x.tril(diagonal), x.triu(diagonal), x.transpose(1, 2).triu(diagonal), x.tril(diagonal)]):
        check(got, want)


@pytest.mark.float64
def test_small_and_batched_linalg_use_cpu_correctly():
    a = make((4, 50, 50), torch.float64) + 50 * torch.eye(50, dtype=torch.float64)
    check(torch.linalg.inv(a.to("vk")), torch.linalg.inv(a), rtol=1e-10, atol=1e-10)
    q, r = torch.linalg.qr(a[0].to("vk"))  # QR stays on the CPU in this milestone
    check(q @ r, a[0], rtol=1e-10, atol=1e-10)
