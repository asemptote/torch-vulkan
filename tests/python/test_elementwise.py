"""Element-wise ops on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, WIDE, check, make, make_complex, on_gpu


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


@pytest.mark.float64
def test_type_promotion():
    i32, f64, i64 = make((5,), torch.int32), make((5,), torch.float64), make((5,), torch.int64, positive=True)
    with on_gpu():
        results = [i32.to("vk") + 0.5, i32.to("vk") + f64.to("vk"), i32.to("vk") * i64.to("vk"), i32.to("vk") / 2, i64.to("vk").sqrt()]
    expected = [i32 + 0.5, i32 + f64, i32 * i64, i32 / 2, i64.sqrt()]
    for got, want in zip(results, expected):
        assert got.dtype == want.dtype
        check(got, want)


def test_inplace_binary_and_slices():
    x = make((6, 6), WIDE)
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


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.int16, torch.float16, torch.bfloat16])
def test_small_dtypes_compute_widened(dtype):
    """8/16-bit types compute in int32/float32 and cast back: integers wrap,
    16-bit floats round to nearest even, as on the CPU."""
    g = torch.Generator().manual_seed(0)
    if dtype.is_floating_point:
        a, b = (torch.randn(50, generator=g) * 30).to(dtype), (torch.randn(50, generator=g) * 30).to(dtype)
    else:
        lo = 0 if dtype == torch.uint8 else -100
        a, b = torch.randint(lo, 100, (50,), generator=g).to(dtype), torch.randint(1, 100, (50,), generator=g).to(dtype)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        got = [va + vb, va * vb, va - vb, -va, va.abs(), torch.maximum(va, vb), va == vb, va < vb]
        if not dtype.is_floating_point:
            got += [va & vb, va ^ vb, ~va, va // vb, va % vb]
        else:
            got += [va / vb, va.exp(), torch.sqrt(va.abs())]
    want = [a + b, a * b, a - b, -a, a.abs(), torch.maximum(a, b), a == b, a < b]
    if not dtype.is_floating_point:
        want += [a & b, a ^ b, ~a, a // b, a % b]
    else:
        want += [a / b, a.exp(), torch.sqrt(a.abs())]
    for gv, w in zip(got, want):
        check(gv, w)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
def test_small_integer_ops_without_their_own_kernels(dtype):
    # (computed in int32 on the GPU: wrapping arithmetic gives the narrow results)
    g = torch.Generator().manual_seed(5)
    lo = 0 if dtype == torch.uint8 else -100
    a, b = (torch.randint(lo, 100, (6, 7), generator=g).to(dtype) for _ in range(2))
    ops = [lambda x, y: torch.clamp(x, -5 if lo else 5, 50), lambda x, y: torch.amax(x, 1), lambda x, y: x.argmin(0),
           lambda x, y: torch.cummax(x, 1).values, lambda x, y: torch.cumsum(x, 0, dtype=dtype),
           lambda x, y: x @ y.t(), lambda x, y: torch.addcmul(x, x, y, value=3), lambda x, y: torch.where(x > y, x, y),
           lambda x, y: x.tril(1), lambda x, y: torch.isin(x, y), lambda x, y: torch.logical_xor(x, y),
           lambda x, y: torch.equal(x, x.clone()), lambda x, y: x.all(1), lambda x, y: x.prod(1)]
    for f in ops:
        with on_gpu():
            got = f(a.to("vk"), b.to("vk"))
        want = f(a, b)
        if isinstance(want, bool):
            assert got == want
        else:
            check(got, want)


def test_unsigned_casts_and_unique():
    # uint16/32/64 convert through their signed twins, as the CPU's casts
    unsigned = [torch.uint16, torch.uint32, torch.uint64]
    vals = torch.tensor([0, 1, 255, 65535, 65536, 2**31, 2**32 - 1, 2**32, 2**53, 2**63 - 1])
    floats = torch.tensor([0.0, 1.5, 255.9, 65535.0, 3e9, 2.0**40, 9.3e18])
    for u, other in itertools.product(unsigned, [torch.float32, torch.int64, torch.int16, torch.uint8, torch.bool] + unsigned):
        if u == other:
            continue
        src = vals.to(u) if u != torch.uint64 else torch.cat([vals, torch.tensor([-1, -2**63])]).view(torch.uint64)
        with on_gpu():
            got = src.to("vk").to(other)
        want = src.to(other)
        if other.is_floating_point:
            torch.testing.assert_close(got.cpu(), want, rtol=1e-6, atol=0)
        else:
            assert torch.equal(got.cpu().view(torch.uint8), want.view(torch.uint8)), (u, other)
        f = floats[floats < {torch.uint16: 65536, torch.uint32: 2**32, torch.uint64: 1.8e19}[u]]
        with on_gpu():
            got = f.to("vk").to(u)
        assert torch.equal(got.cpu().view(torch.uint8), f.to(u).view(torch.uint8)), u
    x = torch.cat([vals, vals[:3]]).to(torch.uint64)
    with on_gpu():
        got = torch.unique(x.to("vk"), return_inverse=True, return_counts=True)
    for g, w in zip(got, torch.unique(x, return_inverse=True, return_counts=True)):
        assert torch.equal(g.cpu(), w)


def test_complex_nan_to_num():
    nan, inf = float("nan"), float("inf")
    x = torch.tensor([complex(1, nan), complex(inf, -inf), complex(nan, 2), complex(-inf, 3)], dtype=torch.complex64)
    for t in (x, x.conj()):
        for kw in ({}, dict(nan=1.0, posinf=5.0, neginf=-5.0)):
            with on_gpu():
                got = torch.nan_to_num(t.to("vk"), **kw)
            assert torch.equal(got.cpu().resolve_conj(), torch.nan_to_num(t, **kw).resolve_conj())


def test_float32_division_correctly_rounded():
    # _fmath.slang's m_div32: x / y and 1 / x equal to the CPU's bit for bit
    # (Vulkan allows 2.5 ulp; the FMA correction it replaced needed a driver
    # that fuses FMAs), over random bit patterns (subnormals included), huge
    # and tiny quotients, and zeros, infinities and NaNs
    g = torch.Generator().manual_seed(5)
    n = 1 << 18

    def bits(count):
        u = torch.randint(0, 0x7F800000, (count,), generator=g, dtype=torch.int64)
        u |= torch.randint(0, 2, (count,), generator=g, dtype=torch.int64) << 31
        return u.to(torch.int32).view(torch.float32)  # (the low 32 bits: no numpy view, which 32-bit builds refuse)

    special = torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1.0, -3.0, 2.0**-126, 2.0**-149,
                            -(2.0**-140), 2.0**-128, 1.1754942e-38, 3.4028235e38, 2.0**127, 7.0, 0.1])
    xs = [bits(n), torch.randn(n, generator=g) * 1e30, torch.randn(n, generator=g) * 1e-30,
          special.repeat_interleave(len(special))]
    ys = [bits(n), torch.randn(n, generator=g) * 1e-8, torch.randn(n, generator=g) * 1e10, special.repeat(len(special))]
    for x, y in zip(xs, ys):
        with on_gpu():
            got = [(x.to("vk") / y.to("vk")).cpu(), torch.reciprocal(x.to("vk")).cpu()]
        for g_, w in zip(got, [x / y, torch.reciprocal(x)]):
            same = (g_.view(torch.int32) == w.view(torch.int32)) | (g_.isnan() & w.isnan())
            assert same.all(), (g_[~same][:4], w[~same][:4])
