"""Linear algebra entirely on the GPU (M13): every size, batches, QR, eigh, SVD,
least squares. Eigenvectors and singular vectors are unique only up to sign,
so those are checked by reconstruction and orthonormality."""

import pytest
import torch

from _util import COMPLEX, check, make, make_complex, on_gpu

FLOATS = [torch.float32, torch.float64]


def tol(dtype, scale=1.0):
    return dict(rtol=1e-4 * scale, atol=1e-4 * scale) if dtype == torch.float32 else dict(rtol=1e-10 * scale,
                                                                                          atol=1e-10 * scale)


def well_conditioned(shape, dtype, seed=0):
    a = make(shape, dtype, seed)
    return a + shape[-1] * torch.eye(shape[-1], dtype=dtype)


def spd(shape, dtype, seed=0):
    a = make(shape, dtype, seed)
    return a @ a.transpose(-2, -1) + shape[-1] * torch.eye(shape[-1], dtype=dtype)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(3, 3), (1, 1), (4, 5, 5), (2, 3, 70, 70), (130, 130)])
def test_lu_family(dtype, shape):
    a = well_conditioned(shape, dtype)
    b = make(shape[:-1] + (3,), dtype, 1)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        got = [torch.linalg.solve(va, vb), torch.linalg.inv(va), torch.linalg.det(va), torch.linalg.slogdet(va),
               torch.linalg.lu_factor(va), torch.linalg.lu_solve(*torch.linalg.lu_factor(va), vb),
               torch.linalg.lu_solve(*torch.linalg.lu_factor(va), vb, adjoint=True),
               torch.linalg.solve(va, vb.transpose(-2, -1), left=False) if shape[-1] == 3 else None]
    want = [torch.linalg.solve(a, b), torch.linalg.inv(a), torch.linalg.det(a), torch.linalg.slogdet(a),
            torch.linalg.lu_factor(a), torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b),
            torch.linalg.lu_solve(*torch.linalg.lu_factor(a), b, adjoint=True),
            torch.linalg.solve(a, b.transpose(-2, -1), left=False) if shape[-1] == 3 else None]
    for g, w in zip(got, want):
        if w is None:
            continue
        for x, y in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(x, y, **tol(dtype, 10))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(5, 3), (3, 6), (2, 7, 7), (80, 70)])
def test_rectangular_lu_and_unpack(dtype, shape):
    a = make(shape, dtype)
    with on_gpu():
        P, L, U = torch.linalg.lu(a.to("vk"))
        lu, piv = torch.linalg.lu_factor(a.to("vk")) if shape[-1] == shape[-2] else (None, None)
    wP, wL, wU = torch.linalg.lu(a)
    for g, w in [(P, wP), (L, wL), (U, wU)]:
        check(g, w, **tol(dtype, 10))
    if lu is not None:
        with on_gpu():
            up = torch.lu_unpack(lu, piv)
        for g, w in zip(up, torch.lu_unpack(*torch.linalg.lu_factor(a))):
            check(g, w, **tol(dtype, 10))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(4, 4), (3, 6, 6), (100, 100)])
def test_cholesky_family_and_triangular(dtype, shape):
    a = spd(shape, dtype)
    b = make(shape[:-1] + (2,), dtype, 1)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        L = torch.linalg.cholesky(va)
        U = torch.linalg.cholesky(va, upper=True)
        got = [L, U, torch.cholesky_solve(vb, L), torch.cholesky_inverse(L), torch.cholesky_inverse(U, upper=True),
               torch.linalg.solve_triangular(L, vb, upper=False),
               torch.linalg.solve_triangular(U, vb.transpose(-2, -1), upper=True, left=False),
               torch.linalg.solve_triangular(L, vb, upper=False, unitriangular=True)]
    wL, wU = torch.linalg.cholesky(a), torch.linalg.cholesky(a, upper=True)
    want = [wL, wU, torch.cholesky_solve(b, wL), torch.cholesky_inverse(wL), torch.cholesky_inverse(wU, upper=True),
            torch.linalg.solve_triangular(wL, b, upper=False),
            torch.linalg.solve_triangular(wU, b.transpose(-2, -1), upper=True, left=False),
            torch.linalg.solve_triangular(wL, b, upper=False, unitriangular=True)]
    for g, w in zip(got, want):
        check(g, w, **tol(dtype, 10))
    bad = a.clone()
    bad[..., 1, 1] = -1
    with on_gpu():
        _, info = torch.linalg.cholesky_ex(bad.to("vk"))
    check(info, torch.linalg.cholesky_ex(bad)[1])


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (3, 5, 5), (90, 70), (70, 130)])
def test_qr_family(dtype, shape):
    a = make(shape, dtype)
    va = a.to("vk")
    with on_gpu():
        got = [torch.linalg.qr(va), torch.linalg.qr(va, mode="complete"), torch.linalg.qr(va, mode="r")[1],
               torch.geqrf(va)]
        qr, tau = got[-1]
        k = min(shape[-2:])
        hp = torch.linalg.householder_product(qr[..., :k], tau)
        c = make(shape[:-2] + (shape[-2], 3), dtype, 2).to("vk")
        om = [torch.ormqr(qr, tau, c), torch.ormqr(qr, tau, c, transpose=True)]
    want = [torch.linalg.qr(a), torch.linalg.qr(a, mode="complete"), torch.linalg.qr(a, mode="r")[1], torch.geqrf(a)]
    wqr, wtau = want[-1]
    wc = c.cpu()
    for g, w in zip(got, want):
        for x, y in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(x, y, **tol(dtype, 10))
    check(hp, torch.linalg.householder_product(wqr[..., :k], wtau), **tol(dtype, 10))
    check(om[0], torch.ormqr(wqr, wtau, wc), **tol(dtype, 10))
    check(om[1], torch.ormqr(wqr, wtau, wc, transpose=True), **tol(dtype, 10))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape", [(5, 5), (2, 9, 9), (64, 64), (1, 1)])
def test_eigh(dtype, shape):
    a = make(shape, dtype)
    a = a + a.transpose(-2, -1)
    va = a.to("vk")
    with on_gpu():
        w, v = torch.linalg.eigh(va)
        wu, _ = torch.linalg.eigh(va, UPLO="U")
        vals = torch.linalg.eigvalsh(va)
    ww = torch.linalg.eigvalsh(a)
    for x in (w, wu, vals):
        check(x, ww, **tol(dtype, 10))
    w, v = w.cpu(), v.cpu()
    eye = torch.eye(shape[-1], dtype=dtype).expand_as(a)
    check_close(v.transpose(-2, -1) @ v, eye, dtype)
    check_close(a @ v, v * w.unsqueeze(-2), dtype)


def hermitian(a, upper):
    t = a.triu(1) if upper else a.tril(-1)
    return t + t.mH + torch.diag_embed(a.diagonal(0, -2, -1).real.to(a.dtype))


@pytest.mark.parametrize("dtype", FLOATS + COMPLEX)
@pytest.mark.parametrize("shape", [(4, 4), (33, 33), (3, 40, 40), (130, 130)])
def test_eigh_tridiagonal(dtype, shape):
    # n >= 4: Householder reduction (shaders/tridiag.slang, gemv.slang) and
    # implicit QL with its rotations logged and applied to Z^T, then Q Z
    a = make_complex(shape, dtype, 1) if dtype.is_complex else make(shape, dtype, 1)
    real = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    n = shape[-1]
    va = a.to("vk")
    for upper in (False, True):
        uplo = "U" if upper else "L"
        with on_gpu():
            w, v = torch.linalg.eigh(va, UPLO=uplo)
            vals = torch.linalg.eigvalsh(va, UPLO=uplo)
        h = hermitian(a, upper)
        want = torch.linalg.eigvalsh(h)
        check(w, want, **tol(real, n))
        check(vals, want, **tol(real, n))
        w, v = w.cpu(), v.cpu()
        eye = torch.eye(n, dtype=dtype).expand_as(h)
        check_close(v.mH @ v, eye, real, n)
        check_close(h @ v, v * w.unsqueeze(-2).to(dtype), real, n)


@pytest.mark.parametrize("dtype,n", [(torch.float64, 520), (torch.float32, 1030)])
def test_eigh_tridiagonal_large(dtype, n):
    # past shaders/tridiag.slang's QL_SHARED: QL iterates on d and e in global memory
    a = make((n, n), dtype, 2)
    h = hermitian(a, False)
    with on_gpu():
        vals = torch.linalg.eigvalsh(a.to("vk"))
    check(vals, torch.linalg.eigvalsh(h), **tol(dtype, n))
    if dtype == torch.float64:
        with on_gpu():
            w, v = torch.linalg.eigh(a.to("vk"))
        w, v = w.cpu(), v.cpu()
        check_close(v.mT @ v, torch.eye(n, dtype=dtype), dtype, n)
        check_close(h @ v, v * w.unsqueeze(-2), dtype, n)


@pytest.mark.parametrize("dtype", FLOATS)
def test_eigh_tridiagonal_degenerate(dtype):
    # repeated, zero and widely scaled eigenvalues, and already (tri)diagonal input
    n = 40
    q, _ = torch.linalg.qr(make((n, n), dtype, 3))
    lam = torch.cat([torch.full((10,), 2.0), torch.zeros(10), torch.linspace(-1, 1, 10), torch.logspace(-6, 6, 10)])
    cases = [torch.eye(n, dtype=dtype), torch.zeros(n, n, dtype=dtype), (q * lam.to(dtype)) @ q.mT,
             torch.diag(torch.arange(n, dtype=dtype)) + torch.diag(torch.ones(n - 1, dtype=dtype), 1) +
             torch.diag(torch.ones(n - 1, dtype=dtype), -1)]
    for a in cases:
        a = (a + a.mT) / 2
        with on_gpu():
            w, v = torch.linalg.eigh(a.to("vk"))
        scale = max(1.0, float(a.abs().max()))
        check(w, torch.linalg.eigvalsh(a), **tol(dtype, n * scale))
        w, v = w.cpu(), v.cpu()
        check_close(v.mT @ v, torch.eye(n, dtype=dtype), dtype, n)
        check_close(a @ v, v * w.unsqueeze(-2), dtype, n * scale)


def check_close(x, y, dtype, scale=10.0):
    torch.testing.assert_close(x, y, **tol(dtype, scale))


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("shape,full", [((7, 4), False), ((4, 7), False), ((7, 4), True), ((4, 7), True),
                                        ((3, 6, 6), False), ((50, 30), False), ((1, 1), True)])
def test_svd(dtype, shape, full):
    a = make(shape, dtype)
    va = a.to("vk")
    with on_gpu():
        U, S, Vh = torch.linalg.svd(va, full_matrices=full)
        sv = torch.linalg.svdvals(va)
    wS = torch.linalg.svdvals(a)
    check(S, wS, **tol(dtype, 10))
    check(sv, wS, **tol(dtype, 10))
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    k = S.shape[-1]
    check_close(U[..., :, :k] @ torch.diag_embed(S) @ Vh[..., :k, :], a, dtype)
    check_close(U.transpose(-2, -1) @ U, torch.eye(U.shape[-1], dtype=dtype).expand(U.shape[:-2] + (U.shape[-1],) * 2), dtype)
    check_close(Vh @ Vh.transpose(-2, -1), torch.eye(Vh.shape[-2], dtype=dtype).expand(Vh.shape[:-2] + (Vh.shape[-2],) * 2), dtype)


@pytest.mark.parametrize("dtype", FLOATS)
def test_svd_rank_deficient(dtype):
    a = make((6, 2), dtype) @ make((2, 5), dtype, 1)  # rank 2
    with on_gpu():
        U, S, Vh = torch.linalg.svd(a.to("vk"))
    U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
    check_close(S, torch.linalg.svdvals(a), dtype, 100)
    check_close(U.T @ U, torch.eye(6, dtype=dtype), dtype)
    check_close(U[:, :5] @ torch.diag(S) @ Vh, a, dtype, 100)


@pytest.mark.parametrize("dtype", FLOATS + COMPLEX)
@pytest.mark.parametrize("shape,full", [((70, 40), False), ((40, 70), True), ((2, 64, 64), False), ((90, 50), True)])
def test_svd_bidiagonal(dtype, shape, full):
    # from 32 columns up: bidiagonal reduction, the Golub-Kahan tridiagonal's
    # divide and conquer, back-transforms; random, rank-deficient and zero
    # matrices
    m, n = shape[-2:]
    r = min(m, n) // 4
    lead = shape[:-2]
    gen = make_complex if dtype.is_complex else make
    cases = [gen(shape, dtype, 4),
             gen(lead + (m, r), dtype, 5) @ gen(lead + (r, n), dtype, 6),
             torch.zeros(shape, dtype=dtype)]
    real = torch.float32 if dtype in (torch.float32, torch.complex64) else torch.float64
    for a in cases:
        with on_gpu():
            U, S, Vh = torch.linalg.svd(a.to("vk"), full_matrices=full)
        U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()
        scale = max(1.0, float(torch.linalg.svdvals(a).max()))
        check_close(S, torch.linalg.svdvals(a), real, 10 * scale)
        k = S.shape[-1]
        check_close(U[..., :, :k] @ torch.diag_embed(S).to(dtype) @ Vh[..., :k, :], a, real, 10 * scale)
        eye_u = torch.eye(U.shape[-1], dtype=dtype).expand(U.shape[:-2] + (U.shape[-1],) * 2)
        eye_v = torch.eye(Vh.shape[-2], dtype=dtype).expand(Vh.shape[:-2] + (Vh.shape[-2],) * 2)
        check_close(U.mH @ U, eye_u, real)
        check_close(Vh @ Vh.mH, eye_v, real)


@pytest.mark.parametrize("dtype", FLOATS)
def test_lstsq(dtype):
    for shape in [(8, 3), (3, 8), (2, 6, 4)]:
        a = make(shape, dtype)
        b = make(shape[:-1] + (2,), dtype, 1)
        with on_gpu():
            got = torch.linalg.lstsq(a.to("vk"), b.to("vk"))
            gd = torch.linalg.lstsq(a.to("vk"), b.to("vk"), driver="gelsd")
        want = torch.linalg.lstsq(a, b)
        wd = torch.linalg.lstsq(a, b, driver="gelsd")
        check(got.solution, want.solution, **tol(dtype, 100))
        check(gd.solution, wd.solution, **tol(dtype, 100))
        check(gd.singular_values, wd.singular_values, **tol(dtype, 100))
        check(gd.residuals, wd.residuals, **tol(dtype, 100))
        check(gd.rank, wd.rank)


@pytest.mark.parametrize("dtype", FLOATS)
def test_matrix_exp(dtype):
    a = make((3, 6, 6), dtype) * torch.tensor([0.01, 1.0, 8.0], dtype=dtype).view(3, 1, 1)
    with on_gpu():
        got = torch.linalg.matrix_exp(a.to("vk"))
    check(got, torch.linalg.matrix_exp(a), **tol(dtype, 100))
