"""Sparse COO and CSR tensors on "vk" (M18) against the CPU."""

import pytest
import torch

import torch_vulkan as tv
from _util import WIDE, on_gpu


def dense_of(t):
    return t.cpu().to_dense() if t.layout != torch.strided else t.cpu()


def same(got, want, tol=1e-5):
    assert got.device.type == "vk"
    assert got.layout == want.layout, (got.layout, want.layout)
    g, w = dense_of(got), dense_of(want)
    assert g.dtype == w.dtype and g.shape == w.shape
    if g.is_floating_point() or g.is_complex():
        torch.testing.assert_close(g, w, rtol=tol, atol=tol)
    else:
        assert torch.equal(g, w)


def vk(*ts):
    return [t.to("vk") for t in ts]


INDICES = torch.tensor([[0, 1, 1, 3, 0], [2, 0, 2, 1, 2]])  # (0, 2) twice: uncoalesced
VALUES = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


def coo():
    return torch.sparse_coo_tensor(INDICES, VALUES, (4, 3))


def test_coo_construction_coalesce_and_conversion():
    s = coo()
    d = torch.randn(4, 3).relu()
    h = torch.randn(3, 4, 5).relu().to_sparse(2)  # hybrid: one dense dim
    with on_gpu():
        sv, dv, hv = vk(s, d, h)
        got = [sv, sv.coalesce(), sv.to_dense(), dv.to_sparse(), dv.to_sparse(1), hv.coalesce(), hv.to_dense(),
               torch.sparse_coo_tensor(*vk(INDICES, VALUES), (4, 3))]
        flags = [sv.is_coalesced(), sv.coalesce().is_coalesced(), sv._nnz(), sv.coalesce()._nnz()]
    want = [s, s.coalesce(), s.to_dense(), d.to_sparse(), d.to_sparse(1), h.coalesce(), h.to_dense(),
            torch.sparse_coo_tensor(INDICES, VALUES, (4, 3))]
    for g, w in zip(got, want):
        same(g, w)
    assert flags == [False, True, 5, 4]
    assert torch.equal(sv.coalesce().indices().cpu(), s.coalesce().indices())
    with pytest.raises(RuntimeError):
        torch.sparse_coo_tensor(*vk(INDICES, VALUES), (2, 3), check_invariants=True)  # an index out of range


def test_coo_arithmetic_and_unary():
    s, d = coo(), torch.randn(4, 3)
    ops = [lambda a, b: b + a, lambda a, b: b.clone().add_(a, alpha=2.5), lambda a, b: a + a.coalesce() * 2,
           lambda a, b: a - a.coalesce() * 2, lambda a, b: a * 3, lambda a, b: a / 2, lambda a, b: a * a.coalesce().neg(),
           lambda a, b: a * b, lambda a, b: -a, lambda a, b: a.abs().sin() + a.abs().sqrt(), lambda a, b: a.pow(2),
           lambda a, b: a.sum(), lambda a, b: torch.sparse.sum(a, 0), lambda a, b: a.t(), lambda a, b: a.clone().zero_(),
           lambda a, b: torch.cat([a, a], 0), lambda a, b: a.unsqueeze(0), lambda a, b: a.narrow_copy(0, 1, 2),
           lambda a, b: b.sparse_mask(a.coalesce()), lambda a, b: a.norm(), lambda a, b: torch.isnan(a)]
    for f in ops:
        with on_gpu():
            got = f(*vk(s, d))
        same(got, f(s, d))
    with pytest.raises(RuntimeError, match="add\\(sparse, dense\\) is not supported"):
        coo().to("vk") + d.to("vk")


@pytest.mark.parametrize("dtype", [torch.float32, WIDE, torch.int64, torch.complex64])
def test_coo_dtypes(dtype):
    x = (torch.randn(5, 6) * 3).relu().to(dtype) if dtype != torch.complex64 else torch.randn(5, 6, dtype=dtype)
    s = x.to_sparse()
    with on_gpu():
        got = [(s.to("vk") * 2).to_dense(), (s.to("vk") + s.to("vk")).coalesce(), x.to("vk").to_sparse()]
    for g, w in zip(got, [(s * 2).to_dense(), (s + s).coalesce(), s]):
        same(g, w)


def test_sparse_matrix_products_and_autograd():
    s = coo()
    m = torch.randn(3, 6)
    big = torch.sparse_coo_tensor(torch.randint(0, 300, (2, 5000)), torch.randn(5000), (300, 400))
    bm = torch.randn(400, 30)
    with on_gpu():
        got = [torch.mm(*vk(s, m)), torch.sparse.mm(*vk(s, m)), torch.addmm(*vk(torch.ones(4, 6), s, m), beta=0.5, alpha=2),
               torch.mv(*vk(s, torch.ones(3))), torch.ones(5, 4, device="vk") @ s.to("vk"), big.to("vk") @ bm.to("vk")]
    want = [torch.mm(s, m), torch.sparse.mm(s, m), torch.addmm(torch.ones(4, 6), s, m, beta=0.5, alpha=2),
            torch.mv(s, torch.ones(3)), torch.ones(5, 4) @ s, big @ bm]
    for g, w in zip(got, want):
        same(g, w, tol=1e-4)
    a, b = s.coalesce().requires_grad_(), m.clone().requires_grad_()
    torch.sparse.mm(a, b).sum().backward()
    av, bv = s.to("vk").coalesce().requires_grad_(), m.to("vk").requires_grad_()
    with on_gpu():
        torch.sparse.mm(av, bv).sum().backward()
    same(av.grad, a.grad)
    same(bv.grad, b.grad)


def test_csr():
    d = torch.randn(5, 4).relu()
    c = d.to_sparse_csr()
    m = torch.randn(4, 3)
    with on_gpu():
        cv, dv, mv = vk(c, d, m)
        got = [cv, cv.to_dense(), dv.to_sparse_csr(), dv.to_sparse().to_sparse_csr(), cv.to_sparse(), cv @ mv,
               torch.addmm(torch.ones(5, 3, device="vk"), cv, mv, beta=0.5, alpha=3), cv.neg().sin(), cv * 2,
               cv.sum(), cv.to_sparse_csc(), torch.sparse_csr_tensor(*vk(c.crow_indices(), c.col_indices(), c.values()),
                                                                     (5, 4))]
        idx = [cv.crow_indices(), cv.col_indices(), torch._convert_indices_from_csr_to_coo(cv.crow_indices(), cv.col_indices()),
               torch._convert_indices_from_coo_to_csr(torch.tensor([0, 0, 2, 3, 3, 5], device="vk"), 6)]
    want = [c, c.to_dense(), d.to_sparse_csr(), d.to_sparse().to_sparse_csr(), c.to_sparse(), c @ m,
            torch.addmm(torch.ones(5, 3), c, m, beta=0.5, alpha=3), c.neg().sin(), c * 2, c.sum(), c.to_sparse_csc(), c]
    for g, w in zip(got, want):
        same(g, w, tol=1e-5)
    for g, w in zip(idx, [c.crow_indices(), c.col_indices(), torch._convert_indices_from_csr_to_coo(c.crow_indices(), c.col_indices()),
                          torch._convert_indices_from_coo_to_csr(torch.tensor([0, 0, 2, 3, 3, 5]), 6)]):
        assert torch.equal(g.cpu(), w)


def test_coo_softmax_sum_backward_bmm_hspmm():
    torch.manual_seed(0)
    s = torch.sparse_coo_tensor(INDICES, VALUES, (4, 3)).coalesce()
    h = torch.randn(3, 4, 5).relu().to_sparse(2)  # (one dense dim)
    sv, hv = vk(s, h)
    with on_gpu():
        for dim in (0, 1):
            same(torch.sparse.softmax(sv, dim), torch.sparse.softmax(s, dim))
            same(torch.sparse.log_softmax(sv, dim), torch.sparse.log_softmax(s, dim))
        same(torch.sparse.softmax(hv, 2), torch.sparse.softmax(h, 2))
        same(torch.sparse.softmax(hv, 0), torch.sparse.softmax(h, 0))
    # gradients: softmax, and sum over sparse and dense dims
    for fn in (lambda t: torch.sparse.softmax(t, 1), lambda t: torch.sparse.log_softmax(t, 0),
               lambda t: torch.sparse.sum(t, [0]), lambda t: torch.sparse.sum(t, [1, 2]),
               lambda t: torch.sparse.sum(t, [0, 1])):
        grads = []
        for t in (h, hv):
            x = t.detach().clone().requires_grad_(True)
            y = fn(x)
            (y.to_dense() if y.layout != torch.strided else y).pow(2).sum().backward()
            grads.append(x.grad)
        same(grads[1], grads[0])
    # bmm, hspmm, and the vector norm
    b3 = torch.randn(2, 4, 3).relu().to_sparse()
    m = torch.randn(2, 3, 5)
    same(torch.bmm(*vk(b3, m)), torch.bmm(b3, m))
    d = torch.randn(3, 6)
    same(torch.hspmm(sv, d.to("vk")), torch.hspmm(s, d))
    assert abs(torch.linalg.vector_norm(sv).item() - torch.linalg.vector_norm(s).item()) < 1e-5


def test_broadcasting_mul_and_explicit_zeros():
    torch.manual_seed(0)
    s = torch.randn(3, 4, 5).relu().to_sparse(2)
    for d in (torch.randn(5), torch.randn(4, 5), torch.randn(2, 3, 4, 5), torch.tensor(2.5)):
        same(s.to("vk") * d.to("vk"), s * d)
        same(d.to("vk") * s.to("vk"), d * s)
    y = torch.randn(3, 4, 1).relu().to_sparse(2)
    same(s.to("vk") * y.to("vk"), s * y)
    # blocked layouts keep explicit zeros through conversions and gradients
    x = torch.tensor([[1.0, 2, 0], [3, 0, 4]])
    for layout, bs in ((torch.sparse_bsc, (2, 1)), (torch.sparse_bsr, (1, 3)), (torch.sparse_csc, None)):
        c = x.to_sparse(layout=layout, blocksize=bs) if bs else x.to_sparse(layout=layout)
        compressed = c.ccol_indices() if layout in (torch.sparse_csc, torch.sparse_bsc) else c.crow_indices()
        plain = c.row_indices() if layout in (torch.sparse_csc, torch.sparse_bsc) else c.col_indices()
        g = torch.arange(6.0).reshape(2, 3) + 1
        grads = []
        for dev in ("cpu", "vk"):
            for f in (lambda t: t.to_dense(masked_grad=True), lambda t: t.to_sparse().to_dense(masked_grad=True)):
                v = c.values().to(dev).clone().requires_grad_(True)
                t = torch.sparse_compressed_tensor(compressed.to(dev), plain.to(dev), v, c.shape, layout=layout)
                grads.append(torch.autograd.grad(f(t), v, g.to(dev))[0].cpu())
        for a, b in zip(grads[:2], grads[2:]):
            assert torch.equal(a, b), (layout, a, b)
        same(c.to("vk").to_sparse(), c.to_sparse())


def test_csr_products_and_sampled_addmm():
    torch.manual_seed(0)
    a, b, v = torch.randn(6, 5).relu(), torch.randn(5, 4), torch.randn(5)
    bias = torch.randn(6, 4)
    for layout in (torch.sparse_csr, torch.sparse_csc):
        ca = a.to_sparse(layout=layout)
        same(torch.addmm(*vk(bias, ca, b), beta=0.5, alpha=2), torch.addmm(bias, ca, b, beta=0.5, alpha=2), 1e-4)
        same(torch.addmv(*vk(bias[:, 0], ca, v), beta=0.5), torch.addmv(bias[:, 0], ca, v, beta=0.5), 1e-4)
        cb = b.to_sparse(layout=layout)
        # (CSR from either layout, as PyTorch's; the reference through dense,
        # as CPU builds without MKL have no sparse @ sparse)
        same(torch.mm(*vk(ca, cb)), torch.mm(ca.to_dense(), cb.to_dense()).to_sparse_csr(), 1e-4)
    # batched sampled_addmm, and the dense op's errors for bad shapes
    m = (torch.randn(2, 4, 4) * (torch.rand(4, 4) > 0.5)).to_sparse_csr()  # (the same pattern per batch)
    x, y = torch.randn(2, 4, 3), torch.randn(2, 3, 4)
    same(torch.sparse.sampled_addmm(*vk(m, x, y), beta=0.5), torch.sparse.sampled_addmm(m, x, y, beta=0.5), 1e-4)
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        torch.mm(*vk(a.to_sparse_csr(), a))
