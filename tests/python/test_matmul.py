"""Matrix products on "vk" vs the CPU."""

import contextlib
import itertools
import math
import os
import subprocess
import sys

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, HAS_FLOAT64, NUMERIC, WIDE, check, make, make_complex, on_gpu


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


def exact_or_mm_tol(dtype, k):
    return dict(rtol=0, atol=0) if not dtype.is_floating_point else mm_tol(dtype, k)


@pytest.mark.parametrize("dtype", NUMERIC)
@pytest.mark.parametrize("rows,k", [(1, 1), (7, 3), (64, 64), (130, 257), (1000, 37), (9, 0), (3, 2000)])
def test_matrix_vector_products(dtype, rows, k):
    # shaders/gemv.slang: one output column (mat @ vec, threads along k for a
    # row-major mat, along rows for a transposed one) or one output row
    # (vec @ mat), each with addmv's beta / alpha
    mat, vec, bias = make((rows, k), dtype, 1), make((k,), dtype, 2), make((rows,), dtype, 3)
    mat_t = make((k, rows), dtype, 4)
    vm, vv, vb, vt = mat.to("vk"), vec.to("vk"), bias.to("vk"), mat_t.to("vk")
    alpha, beta = (2, -1) if not dtype.is_floating_point else (0.5, 2.0)
    with on_gpu():
        results = [vm @ vv, vt.t() @ vv, vv @ vt, vv @ vm.t(), torch.addmv(vb, vm, vv, beta=beta, alpha=alpha),
                   torch.addmv(vb, vt.t(), vv, beta=beta, alpha=alpha)]
    expected = [mat @ vec, mat_t.t() @ vec, vec @ mat_t, vec @ mat.t(), torch.addmv(bias, mat, vec, beta=beta, alpha=alpha),
                torch.addmv(bias, mat_t.t(), vec, beta=beta, alpha=alpha)]
    for got, want in zip(results, expected):
        check(got, want, **exact_or_mm_tol(dtype, k))


@pytest.mark.parametrize("dtype", FLOATS)
def test_matrix_vector_products_strided_batched_and_nan_bias(dtype):
    a = make((5, 40, 70), dtype, 1)
    x = make((5, 70, 3), dtype, 2)
    va, vx = a.to("vk"), x.to("vk")
    nan_bias = torch.full((40,), float("nan"), dtype=dtype)
    with on_gpu():
        results = [
            torch.bmm(va, vx[:, :, 1:2]),  # strided column of x
            torch.bmm(vx[:, :, 2:3].transpose(1, 2), va.transpose(1, 2)),  # one row: [5, 1, 70] @ [5, 70, 40]
            va[:, ::3, ::2] @ vx[:, ::2, 0:1],  # strided rows and k
            torch.addmv(nan_bias.to("vk"), va[0], vx[0, :, 0], beta=0),  # beta = 0 ignores NaN
            torch.addmv(torch.tensor(1.5, dtype=dtype).to("vk"), va[1], vx[1, :, 0]),  # broadcast bias
            va[2, 7] @ vx[2],  # [70] @ [70, 3]: one row, 3 columns
        ]
    expected = [torch.bmm(a, x[:, :, 1:2]), torch.bmm(x[:, :, 2:3].transpose(1, 2), a.transpose(1, 2)),
                a[:, ::3, ::2] @ x[:, ::2, 0:1], torch.addmv(nan_bias, a[0], x[0, :, 0], beta=0),
                torch.addmv(torch.tensor(1.5, dtype=dtype), a[1], x[1, :, 0]), a[2, 7] @ x[2]]
    for got, want in zip(results, expected):
        check(got, want, **mm_tol(dtype, 70))


@pytest.mark.parametrize("dtype", COMPLEX)
def test_complex_matrix_vector_products(dtype):
    mat, vec, bias = make_complex((50, 33), dtype, 1), make_complex((33,), dtype, 2), make_complex((50,), dtype, 3)
    vm, vv, vb = mat.to("vk"), vec.to("vk"), bias.to("vk")
    with on_gpu():
        results = [vm @ vv, vv.conj() @ vm.t(), torch.addmv(vb, vm, vv, beta=0.5, alpha=2.0), vm.mH @ vb]
    expected = [mat @ vec, vec.conj() @ mat.t(), torch.addmv(bias, mat, vec, beta=0.5, alpha=2.0), mat.mH @ bias]
    tol = mm_tol(torch.float32 if dtype == torch.complex64 else torch.float64, 50)
    for got, want in zip(results, expected):
        check(got, want, **tol)


def test_int64_matrix_vector_product_wraps():
    mat = make((20, 30), torch.int64, 1) * 3_000_000_007
    vec = make((30,), torch.int64, 2) * 1_000_003
    with on_gpu():
        got = mat.to("vk") @ vec.to("vk")
    check(got, mat @ vec, rtol=0, atol=0)


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


GEMM4_SCRIPT = r"""
import torch, torch_vulkan
torch.manual_seed(0)
cases = []
for m, k, n in [(1, 4, 4), (9, 8, 132), (130, 64, 256), (33, 12, 20), (5, 0, 8)]:
    a, b = torch.randn(m, k), torch.randn(k, n)
    cases.append(((a.to("vk") @ b.to("vk")).cpu(), a @ b))
a, b, c = torch.randn(3, 20, 16), torch.randn(3, 16, 36), torch.randn(20, 36)
cases.append(((a.to("vk") @ b.to("vk")).cpu(), a @ b))
cases.append((torch.addmm(c.to("vk"), a[1].to("vk"), b[1].to("vk"), beta=0.5, alpha=2).cpu(),
              torch.addmm(c, a[1], b[1], beta=0.5, alpha=2)))
for m, k, n in [(64, 30, 50), (40, 33, 66), (70, 64, 61)]:  # K or N not multiples of 4: zero-padded copies
    a, b = torch.randn(m, k), torch.randn(k, n)
    cases.append(((a.to("vk") @ b.to("vk")).cpu(), a @ b))
    ai, bi = torch.randint(-50, 50, (m, k), dtype=torch.int32), torch.randint(-50, 50, (k, n), dtype=torch.int32)
    assert torch.equal((ai.to("vk") @ bi.to("vk")).cpu(), ai @ bi)  # (int32: exact)
    al, bl = ai.long() * 3_000_000_007, bi.long() * 1_000_003  # (int64, wrapping past 2^63)
    assert torch.equal((al.to("vk") @ bl.to("vk")).cpu(), al @ bl)
x, w = torch.randn(64, 32), torch.randn(48, 32)  # x @ W.t() and x.t() @ y: copied row-major first
cases.append(((x.to("vk") @ w.to("vk").t()).cpu(), x @ w.t()))
cases.append(((x.to("vk").t() @ x.to("vk")).cpu(), x.t() @ x))
x = torch.randn(65, 40)  # (views 16-byte aligned or not)
for v in (x[1:, 4:], x[:, 1:37], x[3:, 2:34]):
    w = torch.randn(v.shape[1], 24)
    cases.append(((v.to("vk") @ w.to("vk")).cpu(), v @ w))
print(max(float((g - w).abs().max() / w.abs().max().clamp_min(1)) for g, w in cases))
"""


SUBGROUP_GEMM_SCRIPT = r"""
import torch, torch_vulkan
torch.manual_seed(0)
worst = 0.0
for dt in (torch.float32, torch.float64):
    cases = []
    for m, k, n in [(2, 3, 2), (8, 16, 32), (65, 17, 130), (130, 257, 70), (33, 1, 40), (5, 0, 8), (200, 64, 129)]:
        a, b = torch.randn(m, k, dtype=dt), torch.randn(k, n, dtype=dt)
        cases.append(((a.to("vk") @ b.to("vk")).cpu(), a @ b))
        cases.append(((a.to("vk") @ b.t().contiguous().to("vk").t()).cpu(), a @ b))  # column-major B
        cases.append(((a.t().contiguous().to("vk").t() @ b.to("vk")).cpu(), a @ b))  # column-major A
    a, b, c = torch.randn(3, 40, 24, dtype=dt), torch.randn(3, 24, 50, dtype=dt), torch.randn(40, 50, dtype=dt)
    cases.append(((a.to("vk") @ b.to("vk")).cpu(), a @ b))
    cases.append((torch.baddbmm(c.to("vk"), a.to("vk"), b.to("vk"), beta=0.5, alpha=2).cpu(), torch.baddbmm(c, a, b, beta=0.5, alpha=2)))
    nan = torch.full((40, 50), float("nan"), dtype=dt)
    cases.append((torch.addmm(nan.to("vk"), a[0].to("vk"), b[0].to("vk"), beta=0).cpu(), torch.addmm(nan, a[0], b[0], beta=0)))
    x = torch.randn(70, 90, dtype=dt)
    cases.append(((x.to("vk")[3:, 5:61] @ x.to("vk")[:56, 10:80]).cpu(), x[3:, 5:61] @ x[:56, 10:80]))  # strided views
    for g, w in cases:
        assert not torch.isnan(g).any(), "NaN"
        worst = max(worst, float((g - w).abs().max() / w.abs().max().clamp_min(1)) / (1e-5 if dt == torch.float32 else 1e-13))
print(worst)
"""


@pytest.mark.parametrize("env", [dict(TORCH_VK_SUBGROUP="8"), dict(TORCH_VK_SUBGROUP="16"), dict(TORCH_VK_SUBGROUP="32"),
                                 dict(TORCH_VK_GEMM="tiled")])
def test_subgroup_gemm_kernel(env):
    # shaders/gemmsg.slang (the default where the device has subgroup size
    # control) at each subgroup size it is instantiated for (a size outside the
    # device's range falls back to one inside it), and gemm.slang's tiles
    out = subprocess.run([sys.executable, "-c", SUBGROUP_GEMM_SCRIPT], env=dict(os.environ, **env), capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip().splitlines()[-1]) < 1.0  # (errors in units of 1e-5 / 1e-13 of the largest entry)


def test_subgroup_gemm_coverage():
    # which kernel float products take: a subgroup size the kernel is built for
    # (0 where the device cannot run it); on lavapipe (llvmpipe, 256-bit
    # vectors, the CI's driver) gemmsg must run, with 8 lanes, so CI covers it
    sizes = [tv._native.kernel_subgroup_size(f"gemmsg_{t}") for t in ("f32", "f64")]
    assert all(s in (0, 8, 16, 32) for s in sizes), sizes
    if "llvmpipe" in tv.device_name() and "256 bits" in tv.device_name():
        assert sizes[0] == 8 and (sizes[1] == 8 or not HAS_FLOAT64), sizes


def test_float4_gemm_kernel():
    # shaders/gemm4.slang (the Mali default) on any GPU, through its env switch
    env = dict(os.environ, TORCH_VK_GEMM="float4")
    out = subprocess.run([sys.executable, "-c", GEMM4_SCRIPT], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip().splitlines()[-1]) < 1e-5
