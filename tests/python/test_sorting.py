"""Sorting, searching, order statistics and accumulation on "vk" vs the CPU (M10)."""

import pytest
import torch

from _util import HAS_FLOAT64, check, make, on_gpu

SORTABLE = [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool, torch.uint8, torch.int8,
            torch.int16, torch.float16, torch.bfloat16]


def sample(shape, dtype, seed=0, ties=False):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if ties or not (dtype.is_floating_point):
        lo = 0 if dtype == torch.uint8 else -5
        return torch.randint(lo, 6, shape, generator=g).to(dtype)
    return torch.randn(shape, generator=g).to(dtype)


@pytest.mark.parametrize("dtype", SORTABLE)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_stable_matches_cpu(dtype, descending):
    for shape, dim in [((7, 33), 1), ((5, 6, 4), 1), ((3000,), 0), ((4, 2500), -1), ((1,), 0), ((2, 0), 1)]:
        x = sample(shape, dtype, ties=True)
        with on_gpu():
            v, i = torch.sort(x.to("vk"), dim=dim, descending=descending, stable=True)
        wv, wi = torch.sort(x, dim=dim, descending=descending, stable=True)
        check(v, wv)
        check(i, wi)


def test_sort_floats_nan_and_signed_zero():
    x = torch.tensor([3.0, float("nan"), -0.0, 0.0, -float("inf"), 1.0, float("nan"), float("inf"), -2.0])
    for desc in (False, True):
        with on_gpu():
            v, i = torch.sort(x.to("vk"), descending=desc, stable=True)
        wv, wi = torch.sort(x, descending=desc, stable=True)
        check(v, wv, equal_nan=True)
        check(i, wi)
        assert torch.equal(torch.signbit(v.cpu()), torch.signbit(wv))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_topk_kthvalue_median_mode(dtype):
    x = sample((6, 41), dtype, ties=dtype == torch.int64)
    v = x.to("vk")
    with on_gpu():
        got = [torch.topk(v, 5, dim=1), torch.topk(v, 3, dim=0, largest=False), torch.kthvalue(v, 7, dim=1),
               torch.kthvalue(v, 2, dim=0, keepdim=True), torch.median(v, dim=1), torch.median(v),
               torch.mode(v, dim=1), torch.argsort(v, dim=1, stable=True), torch.msort(v)]
    want = [torch.topk(x, 5, dim=1), torch.topk(x, 3, dim=0, largest=False), torch.kthvalue(x, 7, dim=1),
            torch.kthvalue(x, 2, dim=0, keepdim=True), torch.median(x, dim=1), torch.median(x),
            torch.mode(x, dim=1), torch.argsort(x, dim=1, stable=True), torch.msort(x)]
    for k, (g, w) in enumerate(zip(got, want)):
        if k in (0, 1, 2, 3, 4, 6):
            # Which of several equal elements topk, kthvalue, median and mode
            # report is unspecified (the CPU uses unstable partial sorts):
            # compare values, and check the indices point at them.
            check(g.values, w.values)
            dim = 0 if k in (1, 3) else 1
            idx, vals = g.indices, w.values
            if idx.dim() < v.dim():
                idx, vals = idx.unsqueeze(dim), vals.unsqueeze(dim)
            check(torch.gather(v, dim, idx), vals)
        elif isinstance(w, tuple):
            for a, b in zip(g, w):
                check(a, b)
        else:
            check(g, w)


def test_median_nanmedian_with_nan():
    x = make((5, 9), torch.float32)
    x[1, 3] = x[1, 7] = float("nan")
    x[4] = float("nan")
    v = x.to("vk")
    with on_gpu():
        got = [torch.median(v, 1), torch.nanmedian(v, 1), torch.nanmedian(v)]
    want = [torch.median(x, 1), torch.nanmedian(x, 1), torch.nanmedian(x)]
    for g, w in zip(got[:2], want[:2]):
        check(g.values, w.values, equal_nan=True)
        check(g.indices, w.indices)
    check(got[2], want[2], equal_nan=True)


@pytest.mark.parametrize("n", [1, 8, 9, 17, 33])
def test_nanmedian_all_nan_row_index(n):
    # (which NaN the CPU picks depends on its C++ standard library and n)
    x = make((3, n), torch.float32)
    x[1] = float("nan")
    with on_gpu():
        got = torch.nanmedian(x.to("vk"), 1)
    want = torch.nanmedian(x, 1)
    check(got.values, want.values, equal_nan=True)
    check(got.indices, want.indices)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_unique(dtype):
    x = sample((7, 9), dtype, ties=True)
    v = x.to("vk")
    with on_gpu():
        got = [torch.unique(v, return_inverse=True, return_counts=True), torch.unique(v, sorted=False),
               torch.unique_consecutive(v.flatten().sort().values, return_inverse=True, return_counts=True),
               torch.unique_consecutive(v, return_counts=True)]
    want = [torch.unique(x, return_inverse=True, return_counts=True), torch.unique(x, sorted=False),
            torch.unique_consecutive(x.flatten().sort().values, return_inverse=True, return_counts=True),
            torch.unique_consecutive(x, return_counts=True)]
    for g, w in zip(got, want):
        for a, b in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_searchsorted_bucketize(dtype):
    seq = sample((40,), dtype, ties=True).sort().values
    seq2 = sample((3, 40), dtype, 1, ties=True).sort(dim=1).values
    vals = sample((3, 17), dtype, 2, ties=True)
    sorter = torch.randperm(40)
    unsorted = seq[sorter.argsort()]
    with on_gpu():
        got = [torch.searchsorted(seq.to("vk"), vals.to("vk")), torch.searchsorted(seq.to("vk"), vals.to("vk"), right=True),
               torch.searchsorted(seq2.to("vk"), vals.to("vk"), side="right", out_int32=True),
               torch.searchsorted(unsorted.to("vk"), vals.to("vk"), sorter=sorter.argsort().to("vk")),
               torch.bucketize(vals.to("vk"), seq.to("vk")), torch.bucketize(vals.to("vk"), seq.to("vk"), right=True)]
    want = [torch.searchsorted(seq, vals), torch.searchsorted(seq, vals, right=True),
            torch.searchsorted(seq2, vals, side="right", out_int32=True),
            torch.searchsorted(unsorted, vals, sorter=sorter.argsort()),
            torch.bucketize(vals, seq), torch.bucketize(vals, seq, right=True)]
    for g, w in zip(got, want):
        check(g, w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
def test_cummax_cummin_logcumsumexp(dtype):
    x = sample((4, 50), dtype, ties=True)
    if dtype.is_floating_point:
        x[1, 10] = float("nan")
    v = x.to("vk")
    with on_gpu():
        got = [torch.cummax(v, 1), torch.cummin(v, 0)]
    want = [torch.cummax(x, 1), torch.cummin(x, 0)]
    for g, w in zip(got, want):
        check(g.values, w.values, equal_nan=True)
        check(g.indices, w.indices)
    if dtype.is_floating_point:
        y = make((3, 60), dtype) * 5
        y[0, 5] = -float("inf")
        with on_gpu():
            got = torch.logcumsumexp(y.to("vk"), 1)
        check(got, torch.logcumsumexp(y, 1), rtol=1e-5 if dtype == torch.float32 else 1e-12, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_accumulating_index_ops(dtype):
    x = sample((5, 6), dtype)
    src = sample((5, 6), dtype, 1)
    idx = torch.randint(0, 6, (5, 6))  # many repeats
    rows = torch.tensor([0, 2, 2, 4, 0])
    v = x.to("vk")
    ops = [
        lambda t, d: torch.scatter_add(t, 1, d(idx), d(src)),
        lambda t, d: torch.scatter_reduce(t, 1, d(idx), d(src), "sum", include_self=False),
        lambda t, d: torch.scatter_reduce(t, 1, d(idx), d(src), "prod"),
        lambda t, d: torch.scatter_reduce(t, 1, d(idx), d(src), "amax", include_self=False),
        lambda t, d: torch.scatter_reduce(t, 1, d(idx), d(src), "amin"),
        lambda t, d: torch.scatter_reduce(t, 1, d(idx), d(src), "mean"),
        lambda t, d: torch.scatter(t, 1, d(idx), d(src), reduce="add"),
        lambda t, d: t.index_add(0, d(rows), d(src), alpha=2),
        lambda t, d: t.index_reduce(0, d(rows), d(src), "amax", include_self=False),
        lambda t, d: t.index_reduce(0, d(rows), d(src), "mean"),
        lambda t, d: t.clone().index_put_((d(rows),), d(src), accumulate=True),
        lambda t, d: t.clone().put_(d(torch.tensor([3, 3, -1, 3])), d(src[0, :4]), accumulate=True),
    ]
    for op in ops:
        with on_gpu():
            got = op(v, lambda t: t.to("vk"))
        want = op(x, lambda t: t)
        if dtype.is_floating_point:
            check(got, want, rtol=1e-6, atol=1e-6)
        else:
            check(got, want)


def test_accumulation_order_matches_cpu_sum():
    # Many float32 values into one bin: the CPU adds them in order, and so do we.
    vals = torch.randn(10000) * 1e3
    idx = torch.zeros(10000, dtype=torch.int64)
    with on_gpu():
        got = torch.zeros(1).to("vk").index_add(0, idx.to("vk"), vals.to("vk"))
    assert torch.equal(got.cpu(), torch.zeros(1).index_add(0, idx, vals))


def test_bincount_histc():
    x = torch.randint(0, 9, (200,))
    w = torch.rand(200)
    f = torch.randn(300)
    with on_gpu():
        got = [torch.bincount(x.to("vk")), torch.bincount(x.to("vk"), w.to("vk"), minlength=12),
               *([torch.bincount(x.to("vk"), x.to("vk").int())] if HAS_FLOAT64 else []),  # (float64 counts)
               torch.histc(f.to("vk"), 10, -2, 2), torch.histc(f.to("vk"), 7)]
    want = [torch.bincount(x), torch.bincount(x, w, minlength=12), *([torch.bincount(x, x.int())] if HAS_FLOAT64 else []),
            torch.histc(f, 10, -2, 2), torch.histc(f, 7)]
    for g, k in zip(got, want):
        check(g, k, rtol=1e-6, atol=1e-6)


def test_small_reductions():
    x = make((4, 7), torch.float32)
    x[0, 1] = float("nan")
    v = x.to("vk")
    with on_gpu():
        got = [torch.aminmax(v[1:], dim=1), torch.var_mean(v[1:], 1), torch.std_mean(v[1:], correction=0),
               torch.nansum(v), torch.nansum(v, 1), torch.trace(v[:, :4])]
    want = [torch.aminmax(x[1:], dim=1), torch.var_mean(x[1:], 1), torch.std_mean(x[1:], correction=0),
            torch.nansum(x), torch.nansum(x, 1), torch.trace(x[:, :4])]
    for g, w in zip(got, want):
        for a, b in zip(g if isinstance(g, tuple) else (g,), w if isinstance(w, tuple) else (w,)):
            check(a, b, equal_nan=True, rtol=1e-5, atol=1e-5)


def test_large_sort():
    x = torch.randn(1 << 18)
    with on_gpu():
        v, i = torch.sort(x.to("vk"))
    wv, wi = torch.sort(x, stable=True)
    check(v, wv)
    check(i, wi)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool, torch.float16, torch.int32])
def test_unique_along_dim(dtype):
    g = torch.Generator().manual_seed(4)
    xs = [torch.randint(0, 3, (50, 4), generator=g), torch.randint(0, 2, (7, 3, 5), generator=g),
          torch.randint(0, 3, (2000, 3), generator=g), torch.randint(0, 2, (13, 37), generator=g)]
    if dtype.is_floating_point:
        xs.append(torch.tensor([[1.0, 3.0], [1.0, 3.0], [0.0, 2.0], [0.0, -0.0], [0.0, 0.0]]))
        # (with NaNs the CPU's sort is not an order: those run on the CPU, for its result)
        nan = torch.tensor([float("nan"), 0, 0, float("nan"), float("nan"), 1], dtype=dtype)
        got = torch.unique(nan.to("vk"), dim=0, return_inverse=True, return_counts=True)
        for g_, w_ in zip(got, torch.unique(nan, dim=0, return_inverse=True, return_counts=True)):
            check(g_, w_, equal_nan=True)
    for x in [t.to(dtype) for t in xs] + [torch.zeros(0, 3, dtype=dtype)]:
        for dim in range(1 if x.numel() == 0 else x.dim()):
            with on_gpu():
                got = [torch.unique(x.to("vk"), dim=dim, return_inverse=True, return_counts=True),
                       torch.unique_consecutive(x.to("vk"), dim=dim, return_inverse=True, return_counts=True)]
            want = [torch.unique(x, dim=dim, return_inverse=True, return_counts=True),
                    torch.unique_consecutive(x, dim=dim, return_inverse=True, return_counts=True)]
            for gs, ws in zip(got, want):
                for g_, w_ in zip(gs, ws):
                    check(g_, w_, equal_nan=True)
    with pytest.raises(RuntimeError, match="0 sized dimensions"):
        torch.unique(torch.zeros(3, 0, device="vk"), dim=0)
