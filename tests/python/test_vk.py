"""torch_vulkan M2 tests: results on device "vk" must match the CPU.

Run with TORCH_VK_VALIDATE=1 to also fail on any Vulkan validation message.
"""

import contextlib
import itertools

import pytest
import torch

import torch_vulkan as tv
from torch_vulkan import _native

NUMERIC = [torch.float32, torch.float64, torch.int32, torch.int64]
ALL = NUMERIC + [torch.bool]


@pytest.fixture(scope="session", autouse=True)
def validation_clean():
    yield
    _native.synchronize()
    assert _native.validation_issues() == 0, "Vulkan validation layer reported problems (see stderr)"


@contextlib.contextmanager
def on_gpu():
    """Fails if anything inside falls back to the CPU."""
    before = tv.fallback_counts()
    yield
    after = tv.fallback_counts()
    new = {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    assert not new, f"ran on the CPU: {new}"


def make(shape, dtype, seed=0, positive=False):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=g).bool()
    if dtype.is_floating_point:
        x = torch.randn(shape, generator=g, dtype=dtype)
        return x.abs() + 0.1 if positive else x
    lo = 1 if positive else -50
    return torch.randint(lo, 50, shape, generator=g, dtype=dtype)


def check(vk_result, cpu_result, **tol):
    assert isinstance(vk_result, tv.VkTensor), type(vk_result)
    assert vk_result.device.type == "vk"
    torch.testing.assert_close(vk_result.cpu(), cpu_result, **tol)


# ---------------------------------------------------------------- transfers


@pytest.mark.parametrize("dtype", ALL + [torch.float16, torch.uint8, torch.complex64])
def test_roundtrip(dtype):
    x = torch.arange(24).reshape(4, 6).to(dtype)
    v = x.to("vk")
    assert isinstance(v, tv.VkTensor) and v.device == torch.device("vk", 0) and v.dtype == dtype
    assert torch.equal(v.cpu(), x)


def test_noncontiguous_upload_and_download():
    x = make((5, 7), torch.float32)
    v = x.t().to("vk")
    assert torch.equal(v.cpu(), x.t())
    assert torch.equal(x.to("vk").t().cpu(), x.t())


# ---------------------------------------------------------------- element-wise

UNARY_CASES = [
    (name, dtype)
    for name, (_, kind) in tv._tensor.UNARY.items()
    for dtype in NUMERIC
    if tv._kernels.kind_allows(kind, dtype)
]


@pytest.mark.parametrize("name,dtype", UNARY_CASES)
def test_unary(name, dtype):
    positive = name in ("sqrt", "rsqrt", "log", "log2", "reciprocal")
    x = make((33, 17), dtype, positive=positive)
    if name == "tan":
        x = x.clamp(-1.2, 1.2)
    op = getattr(torch, name)
    with on_gpu():
        got = op(x.to("vk"))
    check(got, op(x), **f32_tolerance(name, dtype))


def f32_tolerance(name, dtype):
    """Vulkan's precision rules for float32 built-ins are looser than libm's:
    sin/cos/tan have an absolute error bound of 2^-11 on [-pi, pi]; exp/exp2,
    log/log2, pow, tanh and atan2 allow a few ULP or derived bounds."""
    if dtype != torch.float32:
        return {}
    if name in ("sin", "cos", "tan"):
        return dict(rtol=0, atol=2**-11 * (4 if name == "tan" else 1))
    if name in ("exp", "exp2", "log", "log2", "tanh", "sigmoid", "pow", "atan2", "rsqrt"):
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["neg", "abs", "sqrt", "exp"])
def test_unary_inplace_on_view(name):
    x = make((8, 8), torch.float32, positive=True)
    v = x.to("vk")
    with on_gpu():
        getattr(v[:, 2:5], name + "_")()
    getattr(x[:, 2:5], name + "_")()
    check(v, x, **f32_tolerance(name, torch.float32))


BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}


@pytest.mark.parametrize("name,dtype", list(itertools.product(BINARY_OPS, NUMERIC)))
def test_binary_broadcast(name, dtype):
    a, b = make((6, 1, 5), dtype, 1), make((4, 1), dtype, 2)
    op = BINARY_OPS[name]
    with on_gpu():
        got = op(a.to("vk"), b.to("vk"))
    check(got, op(a, b))


@pytest.mark.parametrize("dtype", NUMERIC)
def test_scalar_operands_and_alpha(dtype):
    a, b = make((10, 3), dtype, 1), make((10, 3), dtype, 2)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        results = [va + 3, va * 2, va - 7, torch.add(va, vb, alpha=2), torch.sub(va, vb, alpha=3), 5 - va]
    expected = [a + 3, a * 2, a - 7, torch.add(a, b, alpha=2), torch.sub(a, b, alpha=3), 5 - a]
    for got, want in zip(results, expected):
        check(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_div(dtype):
    a, b = make((50,), dtype, 1), make((50,), dtype, 2, positive=True)
    with on_gpu():
        got = [a.to("vk") / b.to("vk"), a.to("vk") / 4.0]
    check(got[0], a / b)
    check(got[1], a / 4.0)


def test_float32_only_binary():
    a, b = make((40,), torch.float32, 1, positive=True), make((40,), torch.float32, 2)
    with on_gpu():
        pw, p2, at = torch.pow(a.to("vk"), b.to("vk")), a.to("vk") ** 2.0, torch.atan2(a.to("vk"), b.to("vk"))
    check(pw, torch.pow(a, b), **f32_tolerance("pow", torch.float32))
    check(p2, a**2.0, **f32_tolerance("pow", torch.float32))
    check(at, torch.atan2(a, b), **f32_tolerance("atan2", torch.float32))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitwise(dtype):
    a, b = make((30,), dtype, 1), make((30,), dtype, 2)
    with on_gpu():
        got = [a.to("vk") & b.to("vk"), a.to("vk") | b.to("vk"), a.to("vk") ^ 5]
    for g, w in zip(got, [a & b, a | b, a ^ 5]):
        check(g, w)


@pytest.mark.parametrize("op,dtype", list(itertools.product(["eq", "ne", "lt", "le", "gt", "ge"], NUMERIC)))
def test_compare(op, dtype):
    a, b = make((7, 9), dtype, 1), make((9,), dtype, 2)
    a[0, :] = b  # make sure some elements compare equal
    f = getattr(torch, op)
    with on_gpu():
        got_t, got_s = f(a.to("vk"), b.to("vk")), f(a.to("vk"), 0)
    check(got_t, f(a, b))
    check(got_s, f(a, 0))


def test_type_promotion():
    i32, f64, i64 = make((5,), torch.int32), make((5,), torch.float64), make((5,), torch.int64, positive=True)
    with on_gpu():
        results = [i32.to("vk") + 0.5, i32.to("vk") + f64.to("vk"), i32.to("vk") * i64.to("vk"), i32.to("vk") / 2, i64.to("vk").sqrt()]
    expected = [i32 + 0.5, i32 + f64, i32 * i64, i32 / 2, i64.sqrt()]
    for got, want in zip(results, expected):
        assert got.dtype == want.dtype
        check(got, want)


def test_inplace_binary_and_slices():
    x = make((6, 6), torch.float64)
    v = x.to("vk")
    with on_gpu():
        v[:, 1] += 1
        v[::2] *= 3
        v.sub_(v[0], alpha=0.5)
    x[:, 1] += 1
    x[::2] *= 3
    x.sub_(x[0].clone(), alpha=0.5)
    check(v, x)


@pytest.mark.parametrize("src,dst", list(itertools.product(ALL, ALL)))
def test_casts(src, dst):
    x = make((4, 5), src)
    with on_gpu():
        got = x.to("vk").to(dst)
    check(got, x.to(dst))


def test_strided_views_share_memory():
    x = make((4, 6), torch.float32)
    v = x.to("vk")
    t = v.t()
    with on_gpu():
        t[1, 2] = 42.0
        e = v[:, :1].expand(4, 3) * 2
    x.t()[1, 2] = 42.0
    check(v, x)
    check(e, x[:, :1].expand(4, 3) * 2)


def test_many_dims_fall_back_correctly():
    x = make((2, 3, 2, 3, 2), torch.float32)
    p = x.permute(4, 2, 0, 3, 1)  # 5 dims that cannot be merged
    check(x.to("vk").permute(4, 2, 0, 3, 1) + 1, p + 1)
    assert tv.fallback_counts().get("aten.add.Tensor", 0) > 0


# ---------------------------------------------------------------- factories, fallback, autograd


def test_factories():
    with on_gpu():
        z = torch.zeros(3, 4, device="vk")
        o = torch.ones(2, dtype=torch.int64, device="vk")
        f = torch.full((2, 2), 7.5, device="vk", dtype=torch.float64)
    check(z, torch.zeros(3, 4))
    check(o, torch.ones(2, dtype=torch.int64))
    check(f, torch.full((2, 2), 7.5, dtype=torch.float64))
    check(torch.arange(2, 11, 3, device="vk"), torch.arange(2, 11, 3))
    check(torch.eye(3, device="vk"), torch.eye(3))
    r = torch.randn(100, device="vk")
    assert r.shape == (100,) and r.device.type == "vk" and r.std().item() > 0.5


def test_cpu_fallback_matches_and_is_counted():
    x = make((20,), torch.float64)
    before = tv.fallback_counts().get("aten.exp.default", 0)
    with pytest.warns(UserWarning, match="no Vulkan kernel") if before == 0 else contextlib.nullcontext():
        got = x.to("vk").exp()  # float64 exp has no kernel yet
    check(got, x.exp())
    assert tv.fallback_counts()["aten.exp.default"] == before + 1
    check(x.to("vk").sum(), x.sum())


def test_autograd():
    x = make((5,), torch.float32).requires_grad_()
    xv = x.detach().to("vk").requires_grad_()
    (xv * xv + 3 * xv).relu().sum().backward()
    (x * x + 3 * x).relu().sum().backward()
    assert isinstance(xv.grad, tv.VkTensor)
    check(xv.grad, x.grad)


def test_item_repr_and_memory_reuse():
    v = torch.tensor([1.5, 2.5], device="vk")
    assert v[1].item() == 2.5
    assert "device='vk:0'" in repr(v)
    tv.synchronize()
    start = tv.memory_stats()["allocated_bytes"]
    for _ in range(100):
        w = torch.ones(1000, device="vk") * 2
        del w
    tv.synchronize()
    assert tv.memory_stats()["allocated_bytes"] == start
