"""Reductions and scans on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, check, make, make_complex, on_gpu


def red_tol(dtype, n=1000):
    """float32 sums are compared with a tolerance that grows with the length."""
    # Both the GPU and the CPU round; their difference grows like sqrt(n).
    return dict(rtol=1e-5, atol=max(1e-5, 1e-6 * n**0.5)) if dtype == torch.float32 else {}


DIM_CASES = [(None, False), ((0,), False), ((1,), True), ((0, 2), False), ((-1,), True), ((0, 1, 2), False)]


@pytest.mark.parametrize("dims,keepdim", DIM_CASES)
@pytest.mark.parametrize("dtype", NUMERIC + [torch.bool])
def test_sum(dtype, dims, keepdim):
    x = make((7, 33, 5), dtype)
    with on_gpu():
        got = x.to("vk").sum() if dims is None else x.to("vk").sum(dims, keepdim=keepdim)
    want = x.sum() if dims is None else x.sum(dims, keepdim=keepdim)
    assert got.dtype == want.dtype
    check(got, want, **red_tol(dtype))


def test_large_and_noncontiguous_sums():
    x = torch.randn(1 << 22, dtype=torch.float64)
    with on_gpu():
        got64 = x.to("vk").sum()  # multi-pass: 4M elements
        got32 = x.float().to("vk").sum()
    check(got64, x.sum(), rtol=1e-12, atol=1e-9)
    check(got32, x.sum().float(), rtol=1e-4, atol=1e-2)
    y = make((40, 50, 60), torch.float64).permute(2, 0, 1)[:, ::2]
    with on_gpu():
        got = y.to("vk").sum((0, 2))
    check(got, y.sum((0, 2)))
    with on_gpu():
        got = y.to("vk").sum(1)
    check(got, y.sum(1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mean_var_std(dtype):
    x = make((9, 70, 3), dtype)
    v = x.to("vk")
    with on_gpu():
        results = [v.mean(), v.mean(1), v.var(), v.var(1, correction=0), v.std((0, 2), keepdim=True), v.std(1)]
    expected = [x.mean(), x.mean(1), x.var(), x.var(1, correction=0), x.std((0, 2), keepdim=True), x.std(1)]
    for got, want in zip(results, expected):
        check(got, want, **red_tol(dtype))


@pytest.mark.parametrize("dtype", NUMERIC)
def test_prod_amax_amin_max_min(dtype):
    x = make((6, 7, 8), dtype)
    if not dtype.is_floating_point:
        x = x % 3 - 1  # keep integer products small
    v = x.to("vk")
    with on_gpu():
        results = [v.prod(1), v.prod(), v.amax(), v.amax((0, 2)), v.amin(1, keepdim=True), v.max(), v.min()]
    expected = [x.prod(1), x.prod(), x.amax(), x.amax((0, 2)), x.amin(1, keepdim=True), x.max(), x.min()]
    for got, want in zip(results, expected):
        assert got.dtype == want.dtype
        check(got, want, **red_tol(dtype))


@pytest.mark.parametrize("dtype", NUMERIC)
def test_max_dim_and_argmax_with_ties(dtype):
    x = (make((5, 300, 4), dtype) % 4) if not dtype.is_floating_point else make((5, 300, 4), dtype).round()
    v = x.to("vk")
    with on_gpu():
        mv, mi = v.max(1)
        nv, ni = v.min(2, keepdim=True)
        am, an, af = v.argmax(1), v.argmin(0), v.argmax()
    wmv, wmi = x.max(1)
    wnv, wni = x.min(2, keepdim=True)
    check(mv, wmv)
    check(mi, wmi)  # ties resolve to the first index, as on the CPU
    check(nv, wnv)
    check(ni, wni)
    check(am, x.argmax(1))
    check(an, x.argmin(0))
    check(af, x.argmax())


def test_nan_propagation_in_max():
    x = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, 6.0]])
    v = x.to("vk")
    with on_gpu():
        results = [v.max(), v.amax(1), v.argmax(1), v.max(1).values, v.max(1).indices]
    for got, want in zip(results, [x.max(), x.amax(1), x.argmax(1), x.max(1).values, x.max(1).indices]):
        torch.testing.assert_close(got.cpu(), want, equal_nan=True)


@pytest.mark.parametrize("ord", [1, 2, float("inf"), float("-inf")])
def test_vector_norm(ord):
    x = make((30, 40), torch.float64)
    v = x.to("vk")
    with on_gpu():
        got_all, got_dim = torch.linalg.vector_norm(v, ord), torch.linalg.vector_norm(v, ord, dim=1)
    check(got_all, torch.linalg.vector_norm(x, ord))
    check(got_dim, torch.linalg.vector_norm(x, ord, dim=1))


@pytest.mark.parametrize("dtype", [torch.bool, torch.float32, torch.int64])
def test_any_all(dtype):
    x = make((4, 50), dtype)
    x[1] = 0
    v = x.to("vk")
    with on_gpu():
        results = [v.any(), v.all(), v.any(1), v.all(1)]
    for got, want in zip(results, [x.any(), x.all(), x.any(1), x.all(1)]):
        check(got, want)


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize("shape,dim", [((5000,), 0), ((3, 2500), 1), ((70, 6), 0), ((4, 3, 1100), -1), ((4, 3, 1100), 1)])
def test_cumsum(dtype, shape, dim):
    x = make(shape, dtype)
    with on_gpu():
        got = x.to("vk").cumsum(dim)
    want = x.cumsum(dim)
    assert got.dtype == want.dtype  # int32 -> int64, as on the CPU
    check(got, want, **red_tol(dtype, x.size(dim)))


def test_cumprod():
    x = 1 + 0.001 * make((4, 3000), torch.float64)
    with on_gpu():
        got = x.to("vk").cumprod(1)
    check(got, x.cumprod(1))
    i = make((50, 6), torch.int64) % 3 - 1
    with on_gpu():
        got_i = i.to("vk").cumprod(0)
    check(got_i, i.cumprod(0))


def test_empty_reductions():
    e = torch.empty(0, 3, device="vk")
    check(e.sum(0), torch.zeros(3))
    check(e.sum(), torch.tensor(0.0))
    assert e.mean().cpu().isnan()
    check(e.prod(0), torch.ones(3))
    with pytest.raises(RuntimeError):
        e.max()
    with pytest.raises((RuntimeError, IndexError)):  # IndexError on the CPU too
        e.argmax(0)


@pytest.mark.parametrize("dtype", NUMERIC)
def test_column_reductions_per_thread_mode(dtype):
    # Many adjacent outputs with a strided reduced dim: the thread-per-output path.
    x = make((3000, 300), dtype)
    if not dtype.is_floating_point:
        x = x % 5 - 2
    v = x.to("vk")
    with on_gpu():
        results = [v.sum(0), v.amax(0), v.amin(0, keepdim=True), v.any(0), v.prod(0) if not dtype.is_floating_point else v.mean(0)]
    expected = [x.sum(0), x.amax(0), x.amin(0, keepdim=True), x.any(0), x.prod(0) if not dtype.is_floating_point else x.mean(0)]
    for got, want in zip(results, expected):
        check(got, want, **red_tol(dtype, 3000))
    if dtype.is_floating_point:
        with on_gpu():
            got = v.var(0)
        check(got, x.var(0), **red_tol(dtype, 3000))
