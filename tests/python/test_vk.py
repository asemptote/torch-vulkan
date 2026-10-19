"""torch_vulkan tests: results on device "vk" must match the CPU.

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
    assert vk_result.device.type == "vk", vk_result.device
    torch.testing.assert_close(vk_result.cpu(), cpu_result, **tol)


# ---------------------------------------------------------------- transfers


@pytest.mark.parametrize("dtype", ALL + [torch.float16, torch.uint8, torch.complex64])
def test_roundtrip(dtype):
    x = torch.arange(24).reshape(4, 6).to(dtype)
    v = x.to("vk")
    assert v.device == torch.device("vk", 0) and v.dtype == dtype
    assert torch.equal(v.cpu(), x)


def test_noncontiguous_upload_and_download():
    x = make((5, 7), torch.float32)
    v = x.t().to("vk")
    assert torch.equal(v.cpu(), x.t())
    assert torch.equal(x.to("vk").t().cpu(), x.t())


# ---------------------------------------------------------------- element-wise

F32, FLOAT = [torch.float32], [torch.float32, torch.float64]
UNARY_KINDS = {
    **{name: NUMERIC for name in ["neg", "abs", "relu", "sign"]},
    **{name: FLOAT for name in ["reciprocal", "sqrt", "rsqrt", "floor", "ceil", "trunc", "round"]},
    **{name: FLOAT for name in ["exp", "log", "sin", "cos", "tanh", "sigmoid", "exp2", "log2", "tan"]},
}

UNARY_CASES = [
    (name, dtype)
    for name, kind in UNARY_KINDS.items()
    for dtype in NUMERIC
    if dtype in kind
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
        v.sub_(v[0].clone(), alpha=0.5)  # (an overlapping input is rejected, as on the CPU)
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


def test_more_than_four_dims_on_gpu():
    x = make((2, 3, 2, 3, 2, 2), torch.float32)
    p = x.permute(5, 4, 2, 0, 3, 1)  # 6 dims that cannot be merged: outer ones loop on the host
    with on_gpu():
        got = x.to("vk").permute(5, 4, 2, 0, 3, 1) + 1
    check(got, p + 1)


def test_out_argument_of_wrong_size_is_resized():
    # torch-directml and pytorch_ocl both return an empty tensor here (eye.m_out
    # through their CPU fallback); the result must be resized and filled.
    check(torch.eye(4, device="vk"), torch.eye(4))
    out = torch.empty(0, device="vk")
    torch.add(torch.ones(3, device="vk"), 1, out=out)
    check(out, torch.full((3,), 2.0))
    r = torch.empty(0, device="vk")
    torch.arange(5, out=r)
    check(r, torch.arange(5).float())


def test_inplace_metadata_ops():
    v = make((3, 4), torch.float32).to("vk")
    x = v.cpu()
    v.transpose_(0, 1).unsqueeze_(0)
    check(v, x.t().unsqueeze(0))
    v.resize_(2, 6)
    assert v.shape == (2, 6)


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
    before = tv.fallback_counts().get("aten::erf.out", 0)
    with pytest.warns(UserWarning, match="no Vulkan kernel") if before == 0 else contextlib.nullcontext():
        got = x.to("vk").erf()  # no erf kernel yet
    check(got, x.erf())
    assert tv.fallback_counts()["aten::erf.out"] == before + 1
    check(x.to("vk").sum(), x.sum())


def test_autograd():
    x = make((5,), torch.float32).requires_grad_()
    xv = x.detach().to("vk").requires_grad_()
    (xv * xv + 3 * xv).relu().sum().backward()
    (x * x + 3 * x).relu().sum().backward()
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


# ---------------------------------------------------------------- float64 math library

F64_DOMAINS = {
    "exp": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-745, 710, generator=g),
    "exp2": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1075, 1024, generator=g),
    "log": lambda g: torch.exp(torch.empty(200_000, dtype=torch.float64).uniform_(-740, 709, generator=g)),
    "log2": lambda g: torch.exp(torch.empty(200_000, dtype=torch.float64).uniform_(-740, 709, generator=g)),
    "sin": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e5, 1e5, generator=g),
    "cos": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e5, 1e5, generator=g),
    "tan": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-1e3, 1e3, generator=g),
    "tanh": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-25, 25, generator=g),
    "sigmoid": lambda g: torch.empty(200_000, dtype=torch.float64).uniform_(-50, 50, generator=g),
}
F64_SPECIALS = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-310, 5e-324, 1e300, -1e300, 709.8, -745.2,
                             float("inf"), float("-inf"), float("nan")], dtype=torch.float64)


def ulp_error(got, want):
    """|got - want| in units of want's last place (0 where both are equal/NaN)."""
    same = (got == want) | (got.isnan() & want.isnan())
    ulp = torch.nextafter(want.abs(), torch.tensor(float("inf"), dtype=torch.float64)) - want.abs()
    err = (got - want).abs() / ulp
    return torch.where(same, torch.zeros_like(err), err)


@pytest.mark.parametrize("name", list(F64_DOMAINS))
def test_float64_math_accuracy(name):
    x = torch.cat([F64_DOMAINS[name](torch.Generator().manual_seed(0)), F64_SPECIALS])
    op = getattr(torch, name)
    with on_gpu():
        got = op(x.to("vk")).cpu()
    want = op(x)
    # Special values must match exactly (inf, nan, 0, overflow/underflow).
    special = ~torch.isfinite(want) | (want == 0) | ~torch.isfinite(x)
    assert torch.equal(got[special].isnan(), want[special].isnan())
    assert torch.equal(got[special & ~want.isnan()], want[special & ~want.isnan()])
    err = ulp_error(got, want)
    if name in ("sin", "cos", "tan"):
        # Near zeros of sin/cos (and poles of tan) the result is tiny/huge and its
        # ULP meaningless; bound the error relative to the function's scale instead.
        scale = want.abs().clamp(min=1.0) ** (2 if name == "tan" else 1)
        err = torch.minimum(err, (got - want).abs() / (scale * 2.0**-52))
    worst = err[torch.isfinite(err)].max().item()
    assert worst <= 4, f"{name}: max error {worst:.2f} ULP"


# ---------------------------------------------------------------- reductions and scans

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
