"""Complex linear algebra on the GPU (M14): the M13 algorithms with LAPACK's
z* conventions. Eigenvectors and singular vectors are unique only up to a
phase, so those are checked by reconstruction and orthonormality."""

import pytest
import torch

from _util import COMPLEX, check, make, make_complex, on_gpu


def tol(dtype, scale=1.0):
    return dict(rtol=1e-4 * scale, atol=1e-4 * scale) if dtype == torch.complex64 else dict(rtol=1e-10 * scale,
                                                                                            atol=1e-10 * scale)


def eye(n, dtype):
    return torch.eye(n, dtype=dtype)


def well_conditioned(shape, dtype, seed=0):
    return make_complex(shape, dtype, seed) + shape[-1] * eye(shape[-1], dtype)


def hpd(shape, dtype, seed=0):
    a = make_complex(shape, dtype, seed)
    return a @ a.mH + shape[-1] * eye(shape[-1], dtype)


def close(x, y, dtype, scale=10.0):
    torch.testing.assert_close(x, y, **tol(dtype, scale))


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("shape", [(3, 3), (4, 5, 5), (70, 70)])
def test_lu_family(dtype, shape):
    a = well_conditioned(shape, dtype)
    b = make_complex(shape[:-1] + (3,), dtype, 1)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        got = [torch.linalg.solve(va, vb), torch.linalg.inv(va), torch.linalg.det(va), torch.linalg.slogdet(va),
               torch.linalg.lu_factor(va), torch.linalg.lu_solve(*torch.linalg.lu_factor(va), vb, adjoint=True),
               torch.linalg.lu_solve(*torch.linalg.lu_factor(va), vb.mT, left=False),
               torch.linalg.lu_solve(*torch.linalg.lu_factor(va), vb.mT, left=False, adjoint=True),
               torch.linalg.solve(va, vb.mT, left=False), torch.linalg.lu(va)]
    want = [torch.linalg.solve(a, b), torch.linalg.inv(a), torch.linalg.det(a), torch.linalg.slogdet(a),
            torch.linalg.lu_factor(a), torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b, adjoint=True),
            torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b.mT, left=False),
            torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b.mT, left=False, adjoint=True),
            torch.linalg.solve(a, b.mT, left=False), torch.linalg.lu(a)]
    for g, w in zip(got, want):
        for x, y in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(x, y, equal_nan=True, **tol(dtype, 100))  # det overflows for n = 70


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("shape", [(4, 4), (3, 6, 6), (100, 100)])
def test_cholesky_family_and_triangular(dtype, shape):
    a = hpd(shape, dtype)
    b = make_complex(shape[:-1] + (2,), dtype, 1)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        L = torch.linalg.cholesky(va)
        U = torch.linalg.cholesky(va, upper=True)
        got = [L, U, torch.cholesky_solve(vb, L), torch.cholesky_solve(vb, U, upper=True), torch.cholesky_inverse(L),
               torch.linalg.solve_triangular(L, vb, upper=False),
               torch.linalg.solve_triangular(U, vb.mT, upper=True, left=False)]
    wL, wU = torch.linalg.cholesky(a), torch.linalg.cholesky(a, upper=True)
    want = [wL, wU, torch.cholesky_solve(b, wL), torch.cholesky_solve(b, wU, upper=True), torch.cholesky_inverse(wL),
            torch.linalg.solve_triangular(wL, b, upper=False),
            torch.linalg.solve_triangular(wU, b.mT, upper=True, left=False)]
    for g, w in zip(got, want):
        check(g, w, **tol(dtype, 100))


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (3, 5, 5), (90, 70)])
def test_qr_family(dtype, shape):
    a = make_complex(shape, dtype)
    va = a.to("vk")
    with on_gpu():
        got = [torch.linalg.qr(va), torch.linalg.qr(va, mode="complete"), torch.geqrf(va)]
        qr, tau = got[-1]
        k = min(shape[-2:])
        hp = torch.linalg.householder_product(qr[..., :k], tau)
        c = make_complex(shape[:-2] + (shape[-2], 3), dtype, 2).to("vk")
        om = [torch.ormqr(qr, tau, c), torch.ormqr(qr, tau, c, transpose=True),
              torch.ormqr(qr, tau, c.mT, left=False)]
    want = [torch.linalg.qr(a), torch.linalg.qr(a, mode="complete"), torch.geqrf(a)]
    wqr, wtau = want[-1]
    wc = c.cpu()
    for g, w in zip(got, want):
        for x, y in zip(g, w):
            check(x, y, **tol(dtype, 100))
    check(hp, torch.linalg.householder_product(wqr[..., :k], wtau), **tol(dtype, 100))
    check(om[0], torch.ormqr(wqr, wtau, wc), **tol(dtype, 100))
    check(om[1], torch.ormqr(wqr, wtau, wc, transpose=True), **tol(dtype, 100))
    check(om[2], torch.ormqr(wqr, wtau, wc.mT, left=False), **tol(dtype, 100))


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("shape", [(5, 5), (2, 9, 9), (40, 40)])
def test_eigh(dtype, shape):
    a = make_complex(shape, dtype)
    a = a + a.mH
    va = a.to("vk")
    with on_gpu():
        w, v = torch.linalg.eigh(va)
        wu = torch.linalg.eigvalsh(va, UPLO="U")
    ww = torch.linalg.eigvalsh(a)
    check(w, ww, **tol(dtype, 10))
    check(wu, ww, **tol(dtype, 10))
    w, v = w.cpu(), v.cpu()
    close(v.mH @ v, eye(shape[-1], dtype).expand_as(a), dtype)
    close(a @ v, v * w.unsqueeze(-2).to(dtype), dtype, 100)


@pytest.mark.parametrize("dtype", COMPLEX)
@pytest.mark.parametrize("shape,full", [((7, 4), False), ((4, 7), True), ((3, 6, 6), False), ((50, 30), False)])
def test_svd(dtype, shape, full):
    a = make_complex(shape, dtype)
    with on_gpu():
        U, S, Vh = torch.linalg.svd(a.to("vk"), full_matrices=full)
    check(S, torch.linalg.svdvals(a), **tol(dtype, 10))
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    k = S.shape[-1]
    close(U[..., :, :k] @ torch.diag_embed(S).to(dtype) @ Vh[..., :k, :], a, dtype)
    close(U.mH @ U, eye(U.shape[-1], dtype).expand(U.shape[:-2] + (U.shape[-1],) * 2), dtype)
    close(Vh @ Vh.mH, eye(Vh.shape[-2], dtype).expand(Vh.shape[:-2] + (Vh.shape[-2],) * 2), dtype)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_lstsq_pinv(dtype):
    a = make_complex((8, 3), dtype)
    b = make_complex((8, 2), dtype, 1)
    with on_gpu():
        got = torch.linalg.lstsq(a.to("vk"), b.to("vk")).solution
        pinv = torch.linalg.pinv(a.to("vk"))
    check(got, torch.linalg.lstsq(a, b).solution, **tol(dtype, 100))
    check(pinv, torch.linalg.pinv(a), **tol(dtype, 100))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (3, 6, 6), (40, 40)])
def test_eig(dtype, shape):
    # Eigenvalues in no particular order and eigenvectors up to a phase:
    # compare sorted eigenvalues and check A V = V diag(w) with unit columns.
    a = make_complex(shape, dtype) if dtype.is_complex else torch.randn(shape, dtype=dtype,
                                                                         generator=torch.Generator().manual_seed(0))
    with on_gpu():
        w, v = torch.linalg.eig(a.to("vk"))
        wv = torch.linalg.eigvals(a.to("vk"))
    ww = torch.linalg.eigvals(a)
    assert w.dtype == ww.dtype and v.dtype == ww.dtype
    key = lambda x: torch.sort(x.real * 1000 + x.imag, -1).values  # noqa: E731
    real = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    scale = 1e-3 if real == torch.float32 else 1e-9
    torch.testing.assert_close(key(w.cpu()), key(ww), rtol=scale, atol=scale * shape[-1])
    torch.testing.assert_close(key(wv.cpu()), key(ww), rtol=scale, atol=scale * shape[-1])
    w, v = w.cpu(), v.cpu()
    ac = a.to(w.dtype)
    torch.testing.assert_close(ac @ v, v * w.unsqueeze(-2), rtol=scale, atol=scale * shape[-1] * 10)
    torch.testing.assert_close(torch.linalg.vector_norm(v, dim=-2), torch.ones(v.shape[:-2] + v.shape[-1:], dtype=real))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
@pytest.mark.parametrize("hermitian", [False, True])
def test_ldl(dtype, hermitian):
    for shape in [(5, 5), (2, 7, 7), (30, 30)]:
        a = make_complex(shape, dtype) if dtype.is_complex else make(shape, dtype)
        a = a + (a.mH if hermitian else a.mT)
        a[..., 0, 0] = 1e-3  # forces 2 x 2 pivots
        b = make_complex(shape[:-1] + (3,), dtype, 1) if dtype.is_complex else make(shape[:-1] + (3,), dtype, 1)
        with on_gpu():
            LD, piv, info = torch.linalg.ldl_factor_ex(a.to("vk"), hermitian=hermitian)
            x = torch.linalg.ldl_solve(LD, piv, b.to("vk"), hermitian=hermitian)
        wLD, wpiv, winfo = torch.linalg.ldl_factor_ex(a, hermitian=hermitian)
        t = dict(rtol=1e-4, atol=1e-4) if dtype in (torch.float32, torch.complex64) else dict(rtol=1e-9, atol=1e-9)
        check(piv, wpiv)
        check(info, winfo)
        check(LD, wLD, **t)
        check(x, torch.linalg.ldl_solve(wLD, wpiv, b, hermitian=hermitian), **t)


def test_complex_logcumsumexp_and_chalf():
    a = make_complex((4, 9), torch.complex64)
    with on_gpu():
        got = torch.logcumsumexp(a.to("vk"), 1)
        h = a.to("vk").chalf()
        back = h.to(torch.complex64)
        hr = make((5,), torch.float32).to("vk").chalf()
    check(got, torch.logcumsumexp(a, 1), rtol=1e-4, atol=1e-5)
    check(back, a.chalf().to(torch.complex64))
    assert h.dtype == torch.complex32
    check(hr.to(torch.complex64), make((5,), torch.float32).chalf().to(torch.complex64))
