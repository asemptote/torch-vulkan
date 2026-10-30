"""Element-wise ops on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, check, make, make_complex, on_gpu


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
