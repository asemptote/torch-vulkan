"""Matrix products on "vk" vs the CPU."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, WIDE, check, make, make_complex, on_gpu


def mm_tol(dtype, k):
    return dict(rtol=1e-4, atol=1e-5 * max(1, k) ** 0.5) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (3, 5, 7), (64, 64, 64), (65, 17, 130), (200, 300, 150), (5, 0, 4)])
def test_mm_shapes(dtype, m, k, n):
    a, b = make((m, k), dtype, 1), make((k, n), dtype, 2)
    with on_gpu():
        got = a.to("vk") @ b.to("vk")
    check(got, a @ b, **mm_tol(dtype, k))


@pytest.mark.parametrize("dtype", FLOATS)
def test_mm_transposed_and_strided_operands(dtype):
    a, b = make((90, 70), dtype, 1), make((110, 90), dtype, 2)
    va, vb = a.to("vk"), b.to("vk")
    with on_gpu():
        results = [va.t() @ vb.t(), (vb @ va)[::2, 1::3], va[10:50].t() @ vb[:, 5:45].t()]
    expected = [a.t() @ b.t(), (b @ a)[::2, 1::3], a[10:50].t() @ b[:, 5:45].t()]
    for got, want in zip(results, expected):
        check(got, want, **mm_tol(dtype, 110))


@pytest.mark.parametrize("dtype", FLOATS)
def test_bmm_and_broadcast_matmul(dtype):
    a, b = make((6, 33, 20), dtype, 1), make((6, 20, 45), dtype, 2)
    c = make((2, 1, 33, 20), dtype, 3)
    with on_gpu():
        got_bmm = torch.bmm(a.to("vk"), b.to("vk"))
        got_mat = c.to("vk") @ b.to("vk")  # (2, 1, 33, 20) @ (6, 20, 45) -> (2, 6, 33, 45)
        got_t = torch.bmm(b.to("vk").transpose(1, 2), a.to("vk").transpose(1, 2))
    check(got_bmm, torch.bmm(a, b), **mm_tol(dtype, 20))
    check(got_mat, c @ b, **mm_tol(dtype, 20))
    check(got_t, torch.bmm(b.transpose(1, 2), a.transpose(1, 2)), **mm_tol(dtype, 20))


@pytest.mark.parametrize("dtype", FLOATS)
def test_addmm_baddbmm_addmv(dtype):
    bias, a, b = make((40,), dtype, 1), make((30, 25), dtype, 2), make((25, 40), dtype, 3)
    with on_gpu():
        got = torch.addmm(bias.to("vk"), a.to("vk"), b.to("vk"), beta=0.5, alpha=2.0)
    check(got, torch.addmm(bias, a, b, beta=0.5, alpha=2.0), **mm_tol(dtype, 25))

    nan_bias = torch.full((30, 40), float("nan"), dtype=dtype)
    with on_gpu():
        got = torch.addmm(nan_bias.to("vk"), a.to("vk"), b.to("vk"), beta=0)  # beta=0 ignores NaN input
    check(got, torch.addmm(nan_bias, a, b, beta=0), **mm_tol(dtype, 25))

    s, b1, b2 = make((4, 30, 40), dtype, 4), make((4, 30, 25), dtype, 5), make((4, 25, 40), dtype, 6)
    with on_gpu():
        got = torch.baddbmm(s.to("vk"), b1.to("vk"), b2.to("vk"), beta=-1.0, alpha=0.25)
    check(got, torch.baddbmm(s, b1, b2, beta=-1.0, alpha=0.25), **mm_tol(dtype, 25))

    v, mat, vec = make((30,), dtype, 7), make((30, 25), dtype, 8), make((25,), dtype, 9)
    with on_gpu():
        results = [torch.addmv(v.to("vk"), mat.to("vk"), vec.to("vk"), beta=2.0), mat.to("vk") @ vec.to("vk"),
                   vec.to("vk") @ mat.to("vk").t(), torch.dot(vec.to("vk"), vec.to("vk"))]
    expected = [torch.addmv(v, mat, vec, beta=2.0), mat @ vec, vec @ mat.t(), torch.dot(vec, vec)]
    for got, want in zip(results, expected):
        check(got, want, **mm_tol(dtype, 25))


def test_linear_layer_autograd():
    torch.manual_seed(0)
    layer = torch.nn.Linear(50, 20, dtype=WIDE)
    x = make((8, 50), WIDE)
    layer_vk = torch.nn.Linear(50, 20, dtype=WIDE, device="vk")
    layer_vk.load_state_dict(layer.state_dict())
    layer(x).tanh().sum().backward()
    layer_vk(x.to("vk")).tanh().sum().backward()
    check(layer_vk.weight.grad, layer.weight.grad)
    check(layer_vk.bias.grad, layer.bias.grad)


def test_integer_mm_falls_back_correctly():
    a, b = make((7, 9), torch.int64, 1), make((9, 4), torch.int64, 2)
    check(a.to("vk") @ b.to("vk"), a @ b)
