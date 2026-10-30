"""The remaining dtypes and ops of M14 on "vk" vs the CPU: integer, bool and
complex variants of element-wise ops, reductions and accumulation, and the
composite kernels (hash_tensor, frexp, histograms, segment_reduce, CTC loss,
matrix square root and polar decomposition)."""

import pytest
import torch
import torch.nn.functional as F

from _util import COMPLEX, check, make, make_complex, on_gpu


def vk(*ts):
    return [t.to("vk") for t in ts]


def test_integer_and_bool_elementwise():
    i = make((40,), torch.int64)
    j = make((40,), torch.int64, 1)
    b = make((40,), torch.bool)
    c = make((40,), torch.bool, 1)
    vi, vj, vb, vc = vk(i, j, b, c)
    with on_gpu():
        got = [torch.ceil(vi), torch.floor(vi), torch.round(vi), torch.trunc(vi), torch.sign(vb), torch.sgn(vb),
               torch.fmax(vi, vj), torch.fmin(vi, vj), torch.fmax(vb, vc), torch.heaviside(vi, vj),
               torch.heaviside(vb, vc), F.hardtanh(vi, -10, 10), F.relu6(vi), F.threshold(vi, 3, -7),
               torch.clamp_min(vb, vc), torch.clamp_max(vb, vc), vb.max(0), vb.min(0)]
    want = [torch.ceil(i), torch.floor(i), torch.round(i), torch.trunc(i), torch.sign(b), torch.sgn(b), torch.fmax(i, j),
            torch.fmin(i, j), torch.fmax(b, c), torch.heaviside(i, j), torch.heaviside(b, c), F.hardtanh(i, -10, 10),
            F.relu6(i), F.threshold(i, 3, -7), torch.clamp_min(b, c), torch.clamp_max(b, c), b.max(0), b.min(0)]
    for g, w in zip(got, want):
        for x, y in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(x, y)


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_elementwise(dtype):
    a, b, w = make_complex((30,), dtype, 1), make_complex((30,), dtype, 2), make_complex((30,), dtype, 3) * 0.4
    a[0] = 0
    va, vb, vw = vk(a, b, w)
    m = make((30,), torch.bool)
    with on_gpu():
        got = [torch.logaddexp(va, vb), torch.sinc(va), F.silu(va), torch.lerp(va, vb, 0.3),
               torch.lerp(va, vb, vw), torch.addcmul(va, vb, vw, value=2 - 1j), torch.addcdiv(va, vb, vw, value=0.5),
               va.masked_fill(m.to("vk"), 2 + 3j), va.any(), (va * 0).all(), va.all()]
    want = [torch.logaddexp(a, b), torch.sinc(a), F.silu(a), torch.lerp(a, b, 0.3), torch.lerp(a, b, w),
            torch.addcmul(a, b, w, value=2 - 1j), torch.addcdiv(a, b, w, value=0.5), a.masked_fill(m, 2 + 3j), a.any(),
            (a * 0).all(), a.all()]
    for g, w_ in zip(got, want):
        check(g, w_, rtol=1e-5, atol=1e-5)


def test_reductions_edge_cases():
    x = torch.tensor(3.5)
    v = make((5, 6), torch.float32)
    with on_gpu():
        got = [x.to("vk").cumsum(0), x.to("vk").max(0), torch.linalg.vector_norm(v.to("vk"), 4),
               torch.linalg.vector_norm(v.to("vk"), 0.5, dim=1), torch.linalg.vector_norm(v.to("vk"), 0),
               torch.dist(v.to("vk"), v.to("vk") * 2, 3)]
    want = [x.cumsum(0), x.max(0), torch.linalg.vector_norm(v, 4), torch.linalg.vector_norm(v, 0.5, dim=1),
            torch.linalg.vector_norm(v, 0), torch.dist(v, v * 2, 3)]
    for g, w in zip(got, want):
        for a, b in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64, torch.int8, torch.float16])
def test_accumulation_dtypes(dtype):
    def data(shape, seed):
        return make_complex(shape, dtype, seed) if dtype.is_complex else make(shape, dtype, seed) if dtype in (
            torch.bool, torch.float16) else torch.randint(-5, 5, shape, generator=torch.Generator().manual_seed(seed),
                                                          dtype=dtype)

    if dtype == torch.float16:
        def data(shape, seed):
            return make(shape, torch.float32, seed).half()
    self_, src = data((6, 4), 0), data((5, 4), 1)
    index = torch.tensor([0, 2, 2, 5, 0])
    sidx = torch.randint(0, 6, (5, 4), generator=torch.Generator().manual_seed(2))
    vs, vsrc, vi, vsi = vk(self_, src, index, sidx)
    with on_gpu():
        got = [vs.index_add(0, vi, vsrc), vs.scatter_add(0, vsi, vsrc), vs.index_put((vi,), vsrc, accumulate=True),
               vs.clone().put_(vi, vsrc[:, 0].contiguous(), accumulate=True)]
    want = [self_.index_add(0, index, src), self_.scatter_add(0, sidx, src),
            self_.index_put((index,), src, accumulate=True), self_.clone().put_(index, src[:, 0].contiguous(),
                                                                                accumulate=True)]
    for g, w in zip(got, want):
        check(g, w)
    if dtype == torch.bool:
        with on_gpu():
            r = vs.scatter_reduce(0, vsi, vsrc, "amax")
        check(r, self_.scatter_reduce(0, sidx, src, "amax"))


def test_gather_int32_index():
    x = make((10, 5), torch.float32)
    idx = torch.randint(0, 10, (5, 5), generator=torch.Generator().manual_seed(0), dtype=torch.int32)
    with on_gpu():
        got = torch.gather(x.to("vk"), 0, idx.to("vk"))
    check(got, torch.gather(x, 0, idx))


def test_addbmm_integer_conv_and_pool():
    s, a, b = make((5, 7), torch.float32), make((4, 5, 6), torch.float32, 1), make((4, 6, 7), torch.float32, 2)
    x = torch.randint(-9, 9, (2, 3, 9, 9), generator=torch.Generator().manual_seed(0))
    w = torch.randint(-3, 3, (4, 3, 3, 3), generator=torch.Generator().manual_seed(1))
    with on_gpu():
        got = [torch.addbmm(*vk(s, a, b), beta=0.5, alpha=2), F.conv2d(*vk(x, w), padding=1),
               F.conv_transpose2d(*vk(x, w.transpose(0, 1).contiguous()), stride=2),
               F.avg_pool2d(x.to("vk"), 3, 2, 1), F.avg_pool2d(-x.to("vk"), 2, ceil_mode=True)]
    want = [torch.addbmm(s, a, b, beta=0.5, alpha=2), F.conv2d(x, w, padding=1),
            F.conv_transpose2d(x, w.transpose(0, 1).contiguous(), stride=2), F.avg_pool2d(x, 3, 2, 1),
            F.avg_pool2d(-x, 2, ceil_mode=True)]
    for g, w_ in zip(got, want):
        check(g, w_, rtol=1e-5, atol=1e-5)


def test_complex_matrix_exp():
    a = make_complex((2, 5, 5), torch.complex64)
    with on_gpu():
        got = torch.linalg.matrix_exp(a.to("vk"))
    check(got, torch.linalg.matrix_exp(a), rtol=1e-4, atol=1e-4)


def test_hash_tensor_and_frexp():
    x = make((4, 5), torch.float32)
    sub = torch.tensor([1e-40, -3e-39, 0.0, -0.0, float("inf"), float("nan"), 6.0, -0.375])
    with on_gpu():
        got = [torch.hash_tensor(x.to("vk")), torch.hash_tensor(x.to("vk"), dim=1, keepdim=True),
               torch.hash_tensor(torch.arange(7).to("vk"), dim=0), *torch.frexp(x.to("vk")),
               *torch.frexp(sub.to("vk")), *torch.frexp(sub.double().to("vk"))]
    want = [torch.hash_tensor(x), torch.hash_tensor(x, dim=1, keepdim=True), torch.hash_tensor(torch.arange(7), dim=0),
            *torch.frexp(x), *torch.frexp(sub), *torch.frexp(sub.double())]
    for g, w in zip(got, want):
        assert g.dtype == w.dtype
        if w.dtype == torch.uint64:
            assert torch.equal(g.cpu().view(torch.int64), w.view(torch.int64))
        else:
            check(g, w, equal_nan=True)


def test_histograms():
    x = make((200,), torch.float32)
    wts = make((200,), torch.float32, 1).abs()
    pts = make((100, 2), torch.float32, 2)
    with on_gpu():
        got = [torch.histogram(x.to("vk"), 7), torch.histogram(x.to("vk"), 5, range=(-1.0, 1.0), weight=wts.to("vk"),
                                                               density=True),
               torch.histogram(x.to("vk"), torch.tensor([-2.0, -0.5, 0.0, 0.1, 3.0]).to("vk")),
               torch.histogramdd(pts.to("vk"), [3, 4]), torch.histogramdd(pts.to("vk"), [2, 2], density=True)]
    want = [torch.histogram(x, 7), torch.histogram(x, 5, range=(-1.0, 1.0), weight=wts, density=True),
            torch.histogram(x, torch.tensor([-2.0, -0.5, 0.0, 0.1, 3.0])), torch.histogramdd(pts, [3, 4]),
            torch.histogramdd(pts, [2, 2], density=True)]
    for g, w in zip(got, want):
        check(g.hist, w.hist, rtol=1e-5, atol=1e-6)
        for a, b in zip(g.bin_edges if isinstance(g.bin_edges, (list, tuple)) else [g.bin_edges],
                        w.bin_edges if isinstance(w.bin_edges, (list, tuple)) else [w.bin_edges]):
            check(a, b)


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min", "prod"])
def test_segment_reduce(reduce):
    d = make((6, 3), torch.float32)
    lengths = torch.tensor([2, 0, 3, 1])
    with on_gpu():
        got = [torch.segment_reduce(d.to("vk"), reduce, lengths=lengths.to("vk")),
               torch.segment_reduce(d.to("vk"), reduce, lengths=lengths.to("vk"), initial=0.5),
               torch.segment_reduce(d.to("vk"), reduce, offsets=torch.tensor([0, 1, 4, 6]).to("vk"))]
    want = [torch.segment_reduce(d, reduce, lengths=lengths), torch.segment_reduce(d, reduce, lengths=lengths, initial=0.5),
            torch.segment_reduce(d, reduce, offsets=torch.tensor([0, 1, 4, 6]))]
    for g, w in zip(got, want):
        check(g, w, equal_nan=True, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ctc_loss(dtype):
    g = torch.Generator().manual_seed(0)
    T, N, C = 12, 3, 5
    lp = torch.randn(T, N, C, generator=g, dtype=dtype).log_softmax(2)
    targets = torch.tensor([[1, 2, 2], [3, 1, 0], [4, 0, 0]])
    il, tl = [12, 9, 6], [3, 2, 1]
    vlp = lp.to("vk").requires_grad_()
    clp = lp.clone().requires_grad_()
    with on_gpu():
        loss = F.ctc_loss(vlp, targets.to("vk"), il, tl, reduction="sum")
        loss.backward()
        nll, alpha = torch.ops.aten._ctc_loss(lp.to("vk"), targets.to("vk"), il, tl)
        flat = F.ctc_loss(lp.to("vk"), torch.tensor([1, 2, 2, 3, 1, 4]).to("vk"), il, tl, reduction="none")
    want = F.ctc_loss(clp, targets, il, tl, reduction="sum")
    want.backward()
    wnll, walpha = torch.ops.aten._ctc_loss(lp, targets, il, tl)
    tl_ = dict(rtol=1e-4, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
    check(loss.detach(), want.detach(), **tl_)
    check(vlp.grad, clp.grad, **tl_)
    check(nll, wnll, **tl_)
    check(alpha, walpha, **tl_)
    check(flat, F.ctc_loss(lp, torch.tensor([1, 2, 2, 3, 1, 4]), il, tl, reduction="none"), **tl_)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_matrix_sqrth_and_polar(dtype):
    a = make_complex((3, 4, 4), dtype) if dtype.is_complex else make((3, 4, 4), dtype)
    h = a @ a.mH
    with on_gpu():
        got = [torch.ops.aten.linalg_matrix_sqrth(h.to("vk")), *torch.ops.aten.linalg_polar(a.to("vk"))]
    want = [torch.ops.aten.linalg_matrix_sqrth(h), *torch.ops.aten.linalg_polar(a)]
    for g, w in zip(got, want):
        check(g, w, rtol=1e-8, atol=1e-8)
