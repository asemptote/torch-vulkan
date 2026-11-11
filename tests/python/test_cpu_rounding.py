"""Results that round as the CPU's (M25): correctly rounded sqrt and log1p,
scatter means, 16-bit ops whose intermediates the CPU rounds, conjugate views
inside a TorchDispatchMode, pinv's own derivative, rrelu under vmap; and the
allocator hooks (torch.accelerator's memory functions, pinned memory)."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from _util import HAS_FLOAT64, check, make, make_complex, on_gpu


def wide_range(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    lim = 80 if dtype == torch.float32 else 600
    x = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(-lim, lim, generator=g)).to(dtype)
    tiny = torch.finfo(dtype).tiny
    specials = [0.0, -0.0, math.inf, -1.0, math.nan, tiny / 4, tiny / 1024, 1.0, 2.0, 4.0, 0.25]
    return torch.cat([x, torch.tensor(specials, dtype=dtype)])


def assert_bitwise(got, want):
    got = got.cpu()
    same = (got == want) | (got.isnan() & want.isnan())
    assert bool(same.all()), f"{int((~same).sum())} of {same.numel()} differ"
    assert bool((torch.signbit(got) == torch.signbit(want))[~want.isnan()].all())


def test_sqrt_rsqrt_float32_bitwise():
    x = wide_range(1 << 16, torch.float32)
    vx = x.to("vk")
    with on_gpu():
        got = [torch.sqrt(vx), torch.rsqrt(vx), torch.pow(vx, 0.5)]
    for g, w in zip(got, [torch.sqrt(x), torch.rsqrt(x), torch.pow(x, 0.5)]):
        assert_bitwise(g, w)


@pytest.mark.skipif(not HAS_FLOAT64, reason="no float64 on this GPU")
def test_sqrt_float64_correctly_rounded():
    # (against math.sqrt: the CPU's vectorised float64 sqrt (MKL) is not
    # always correctly rounded)
    x = wide_range(4096, torch.float64)
    with on_gpu():
        got = torch.sqrt(x.to("vk")).cpu()
    want = torch.tensor([math.sqrt(v) if v >= 0 or math.isnan(v) else math.nan for v in x.tolist()],
                        dtype=torch.float64)
    want[x == 0] = x[x == 0]  # (sqrt(-0) is -0)
    assert_bitwise(got, want)


def test_log1p_float32_accuracy():
    g = torch.Generator().manual_seed(1)
    y = torch.cat([torch.empty(1 << 15).uniform_(-1, 1, generator=g) * s for s in (1e-7, 1e-3, 1.0)]
                  + [torch.exp(torch.empty(1 << 15).uniform_(-30, 30, generator=g))])
    with on_gpu():
        got = torch.log1p(y.to("vk")).cpu().double()
    ref = torch.log1p(y.double())
    rel = (got - ref).abs() / ref.abs().clamp(min=1e-38) / torch.finfo(torch.float32).eps
    assert rel.max().item() < 0.52, rel.max().item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scatter_reduce_mean_bitwise(dtype):
    if dtype == torch.float64 and not HAS_FLOAT64:
        pytest.skip("no float64 on this GPU")
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(0, 50, (20000,), generator=g)
    src, base = make((20000,), dtype, 3), make((50,), dtype, 4)
    for include_self in (True, False):
        with on_gpu():
            got = base.to("vk").scatter_reduce(0, idx.to("vk"), src.to("vk"), "mean", include_self=include_self)
        assert_bitwise(got, base.scatter_reduce(0, idx, src, "mean", include_self=include_self))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("reduce", ["sum", "mean", "amax"])
def test_scatter_whole_rows(dtype, reduce):
    # (an index expanded along the other dims: one sort key per row)
    m, n, k = 6, 40, 33
    self_ = make((m, k), dtype, 5)
    src = make((n + 2, k), dtype, 6)
    idx = torch.randint(0, m, (n,), generator=torch.Generator().manual_seed(7)).unsqueeze(1).expand(n, k)
    for include_self in (True, False):
        with on_gpu():
            got = self_.to("vk").scatter_reduce(0, idx.to("vk"), src.to("vk"), reduce, include_self=include_self)
            added = self_.to("vk").scatter_add(0, idx.to("vk"), src.to("vk"))
        tol = dict(rtol=1e-2, atol=1e-2) if dtype == torch.bfloat16 else {}
        check(got, self_.scatter_reduce(0, idx, src, reduce, include_self=include_self), **tol)
        check(added, self_.scatter_add(0, idx, src), **tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_16bit_rounding_as_the_cpu(dtype):
    g = torch.Generator().manual_seed(8)
    x = torch.randn(4096, generator=g).to(dtype)
    e = torch.randint(-20, 20, (4096,), generator=g)
    with on_gpu():
        got = torch.ldexp(x.to("vk"), e.to("vk"))
    assert_bitwise(got, torch.ldexp(x, e))
    s = torch.randn(64, 10, generator=g).to(dtype)
    t = torch.randint(0, 10, (64,), generator=g)
    w = torch.rand(10, generator=g).to(dtype)
    for reduction in ("mean", "sum", "none"):
        grads = []
        for dev in ("cpu", "vk"):
            xx = s.detach().to(dev).clone().requires_grad_()
            loss = F.nll_loss(xx, t.to(dev), weight=w.to(dev), reduction=reduction)
            loss.backward(torch.ones_like(loss))
            grads.append(xx.grad)
        assert_bitwise(grads[1], grads[0])
    grads = []
    for dev in ("cpu", "vk"):
        xx = s.detach().to(dev).clone().requires_grad_()
        F.logsigmoid(xx).sum().backward()
        grads.append(xx.grad)
    assert_bitwise(grads[1], grads[0])


class PassThrough(TorchDispatchMode):
    """A mode that only runs each op (Conjugate is excluded while it does)."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_conjugates_inside_a_dispatch_mode(dtype):
    a, b = make_complex((7,), dtype, 9), make_complex((7,), dtype, 10)
    y = torch.tanh(make_complex((7,), dtype, 11))
    va, vb, vy = a.to("vk"), b.to("vk"), y.to("vk")
    with on_gpu(), PassThrough():
        got = [torch.ops.aten.tanh_backward(va, vy), torch.ops.aten.sigmoid_backward(va, vy),
               torch.dot(va, vb.conj()), torch.dot(va.conj(), vb.conj())]
    want = [torch.ops.aten.tanh_backward(a, y), torch.ops.aten.sigmoid_backward(a, y), torch.dot(a, b.conj()),
            torch.dot(a.conj(), b.conj())]
    for g, w in zip(got, want):
        check(g, w)


def test_pinv_uses_its_own_derivative():
    a = make((4, 3), torch.float32, 12).to("vk").requires_grad_()
    with on_gpu():
        p = torch.linalg.pinv(a)
    assert type(p.grad_fn).__name__ == "LinalgPinvBackward0", type(p.grad_fn).__name__


def test_rrelu_noise_inside_vmap():
    from torch._vmap_internals import _vmap

    x = make((5,), torch.float32, 13).to("vk")
    ts = make((3, 5), torch.float32, 14).to("vk")
    out = _vmap(lambda t: F.rrelu(x, 0.1, 0.3, training=True) * t)(ts)  # (no "random operations" error)
    assert out.shape == (3, 5)


def test_accelerator_memory_functions():
    x = torch.ones(1 << 20, device="vk")
    assert torch.accelerator.memory_allocated() >= x.numel() * 4
    del x
    torch.accelerator.empty_cache()


def test_pinned_memory():
    p = torch.randn(16).pin_memory()
    assert p.is_pinned()
    assert not torch.randn(16).is_pinned()
    check(p.to("vk"), p)
