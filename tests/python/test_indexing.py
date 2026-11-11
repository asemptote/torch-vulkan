"""Data movement and indexing on "vk" vs the CPU (M9)."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from _util import ALL, ARM32, COMPLEX, FLOATS, NUMERIC, check, make, make_complex, on_gpu

import torch_vulkan as tv

SMALL = [torch.uint8, torch.int8, torch.int16, torch.float16, torch.bfloat16]
EVERY = ALL + COMPLEX + SMALL


def sample(shape, dtype, seed=0):
    if dtype in COMPLEX:
        return make_complex(shape, dtype, seed)
    if dtype in SMALL:
        return make(shape, torch.float32, seed).mul(20).to(dtype)
    return make(shape, dtype, seed)


@pytest.mark.parametrize("src,dst", list(itertools.product(ALL + SMALL, ALL + SMALL)))
def test_casts_small_dtypes(src, dst):
    x = sample((5, 7), src)
    with on_gpu():
        got = x.to("vk").to(dst)
    check(got, x.to(dst))


@pytest.mark.parametrize("dtype", EVERY)
def test_strided_copy_and_fill(dtype):
    x = sample((6, 8), dtype)
    v = x.to("vk")
    with on_gpu():
        t = v.t().contiguous()
        v[::2, 1::3] = v[1::2, ::3].clone()
        v[:, 0].fill_(1)
    check(t, x.t().contiguous())
    x[::2, 1::3] = x[1::2, ::3].clone()
    x[:, 0].fill_(1)
    check(v, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.complex64, torch.float16])
def test_index_select_gather_take(dtype):
    x = sample((5, 6, 4), dtype)
    idx = torch.tensor([4, 0, 3, 2, 2])
    idx2 = torch.tensor([3, 0, 1])
    g = torch.randint(0, 6, (5, 3, 4))
    t = torch.tensor([[0, -1], [7, 100]])
    v = x.to("vk")
    with on_gpu():
        got = [v.index_select(0, idx.to("vk")), v.index_select(2, idx2.to("vk")), torch.gather(v, 1, g.to("vk")),
               v.transpose(0, 2).take(t.to("vk"))]
    want = [x.index_select(0, idx), x.index_select(2, idx2), torch.gather(x, 1, g), x.transpose(0, 2).take(t)]
    for a, b in zip(got, want):
        check(a, b)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32, torch.bool])
def test_advanced_indexing(dtype):
    x = sample((5, 6, 7), dtype)
    v = x.to("vk")
    i0, i2 = torch.tensor([[0, 4], [3, -2]]), torch.tensor([1, 6])
    mask = make((5, 6), torch.float32) > 0
    cases = [
        lambda t, d: t[d(i0)],
        lambda t, d: t[:, d(i0)],
        lambda t, d: t[d(i0), :, d(i2)],  # non-adjacent: indexed dims go first
        lambda t, d: t[d(mask)],
        lambda t, d: t[1:4, d(torch.tensor([0, 5])), ::2],
        lambda t, d: t[d(torch.tensor([True, False, True, True, False]))],
    ]
    for case in cases:
        with on_gpu():
            got = case(v, lambda i: i.to("vk"))
        check(got, case(x, lambda i: i))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.complex128])
def test_index_put(dtype):
    x = sample((5, 6), dtype)
    vals = sample((2, 6), dtype, 3)
    v = x.to("vk")
    with on_gpu():
        v[torch.tensor([3, 0]).to("vk")] = vals.to("vk")
        v[:, torch.tensor([1, -1]).to("vk")] = sample((), dtype, 4).to("vk")
        v[v.real > 0.5 if dtype.is_complex else v > 0.5] = 0
    x[torch.tensor([3, 0])] = vals
    x[:, torch.tensor([1, -1])] = sample((), dtype, 4)
    x[x.real > 0.5 if dtype.is_complex else x > 0.5] = 0
    check(v, x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_scatter_index_copy_fill_put(dtype):
    x = sample((4, 5), dtype)
    src = sample((4, 5), dtype, 2)
    perm = torch.stack([torch.randperm(5) for _ in range(3)])  # unique per row: deterministic
    v = x.to("vk")
    with on_gpu():
        got = [
            torch.scatter(v, 1, perm.to("vk"), src.to("vk")),
            torch.scatter(v, 1, perm[:, :2].to("vk"), 7),
            v.index_copy(0, torch.tensor([3, 1]).to("vk"), src[:2].to("vk")),
            v.clone().index_fill_(1, torch.tensor([0, -1]).to("vk"), -2),
            v.t().clone().put_(torch.tensor([0, 19, -3]).to("vk"), src[0, :3].to("vk")),
        ]
    want = [
        torch.scatter(x, 1, perm, src),
        torch.scatter(x, 1, perm[:, :2], 7),
        x.index_copy(0, torch.tensor([3, 1]), src[:2]),
        x.clone().index_fill_(1, torch.tensor([0, -1]), -2),
        x.t().clone().put_(torch.tensor([0, 19, -3]), src[0, :3]),
    ]
    for a, b in zip(got, want):
        check(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
def test_nonzero_masked(dtype):
    x = sample((4, 3, 5), dtype)
    mask = make((3, 5), torch.float32) > 0
    src = sample((60,), dtype, 5)
    v = x.to("vk")
    with on_gpu():
        got = [torch.nonzero(v), torch.argwhere(v), torch.masked_select(v, mask.to("vk")),
               v.clone().masked_scatter_(mask.to("vk"), src.to("vk")), torch.count_nonzero(v, dim=(0, 2)),
               torch.nonzero(v.flatten()), torch.nonzero(v, as_tuple=True)[1]]
    want = [torch.nonzero(x), torch.argwhere(x), torch.masked_select(x, mask), x.clone().masked_scatter_(mask, src),
            torch.count_nonzero(x, dim=(0, 2)), torch.nonzero(x.flatten()), torch.nonzero(x, as_tuple=True)[1]]
    for a, b in zip(got, want):
        check(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.complex64, torch.float16])
def test_cat_flip_roll(dtype):
    a, b, c = sample((3, 4), dtype), sample((2, 4), dtype, 1), sample((3, 2), dtype, 2)
    va, vb, vc = a.to("vk"), b.to("vk"), c.to("vk")
    with on_gpu():
        got = [torch.cat([va, vb]), torch.cat([va, vc], 1), torch.stack([va, va.flip(0)]), va.flip(0, 1),
               va.t().flip(1), va.roll(1, 0), va.roll((-5, 2), (1, 0)), va.roll(3), torch.cat([va, va.new_empty(0)])]
    want = [torch.cat([a, b]), torch.cat([a, c], 1), torch.stack([a, a.flip(0)]), a.flip(0, 1), a.t().flip(1),
            a.roll(1, 0), a.roll((-5, 2), (1, 0)), a.roll(3), torch.cat([a, a.new_empty(0)])]
    for g, w in zip(got, want):
        check(g, w)


@pytest.mark.parametrize("mode,dims", [("reflect", 1), ("reflect", 2), ("reflect", 3), ("replicate", 1),
                                       ("replicate", 2), ("replicate", 3)])
def test_padding(mode, dims):
    x = make((2, 3) + (5,) * dims, torch.float32)
    pad = (2, 1, 0, 3, 1, 1)[: 2 * dims]
    with on_gpu():
        got = F.pad(x.to("vk"), pad, mode=mode)
    check(got, F.pad(x, pad, mode=mode))


def test_channel_shuffle():
    x = make((2, 6, 3, 2), torch.float32)
    with on_gpu():
        got = F.channel_shuffle(x.to("vk"), 3)
    check(got, F.channel_shuffle(x, 3))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.int64, torch.float16, torch.uint8])
def test_sequences(dtype):
    if ARM32 and dtype == torch.float16:
        pytest.skip("a 32-bit Arm CPU's float16 linspace runs in native float16 arithmetic, x86's in float32")
    with on_gpu():
        got = [torch.arange(3, 50, 4, dtype=dtype, device="vk"), torch.arange(10, dtype=dtype, device="vk"),
               torch.linspace(-3, 40, 37, dtype=dtype, device="vk"), torch.eye(4, 6, dtype=dtype, device="vk")]
    want = [torch.arange(3, 50, 4, dtype=dtype), torch.arange(10, dtype=dtype),
            torch.linspace(-3, 40, 37, dtype=dtype), torch.eye(4, 6, dtype=dtype)]
    for g, w in zip(got, want):
        check(g, w)
    if dtype.is_floating_point:
        with on_gpu():
            got = [torch.arange(-1.5, 2.25, 0.25, dtype=dtype, device="vk"),
                   torch.logspace(-2, 2, 9, dtype=dtype, device="vk")]
        # float16: PyTorch rounds the exponents to half before pow; we round once.
        tol = dict(rtol=1e-2, atol=1e-3) if dtype == torch.float16 else {}
        check(got[0], torch.arange(-1.5, 2.25, 0.25, dtype=dtype))
        check(got[1], torch.logspace(-2, 2, 9, dtype=dtype), **tol)


def test_equal_isin():
    a = make((5, 4), torch.int64)
    b = a.clone()
    b[2, 1] += 1
    test = torch.tensor([3, -7, 11, 0])
    va = a.to("vk")
    with on_gpu():
        results = [torch.equal(va, a.to("vk")), torch.equal(va, b.to("vk")), torch.equal(va, va[:2])]
        isin = [torch.isin(va, test.to("vk")), torch.isin(va, test.to("vk"), invert=True)]
    assert results == [True, False, False]
    check(isin[0], torch.isin(a, test))
    check(isin[1], torch.isin(a, test, invert=True))


def test_out_of_range_index_raises_on_sync():
    v = torch.arange(10.0).to("vk")
    with on_gpu():
        bad = v[torch.tensor([1, 10]).to("vk")]  # queued; checked when the host next waits
    with pytest.raises(IndexError, match="index out of range"):
        bad.cpu()
    with pytest.raises(IndexError):
        torch.gather(v, 0, torch.tensor([-11]).to("vk"))
        tv.synchronize()
    with pytest.raises(IndexError):  # gather and index_select take no negative indices, as on the CPU
        v.index_select(0, torch.tensor([-1]).to("vk")).cpu()
    check(v[torch.tensor([1, -1]).to("vk")], torch.tensor([1.0, 9.0]))  # the error is cleared
