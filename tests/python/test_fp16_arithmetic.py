"""torch_vulkan.set_fp16_arithmetic: float16 matrix products and convolutions
in float16 arithmetic (opt-in; the default matches the CPU)."""

import contextlib

import pytest
import torch
import torch.nn.functional as F

import torch_vulkan as tv

# Where the float16-arithmetic kernels run (shaders/gemm4s.slang and
# conv4s.slang need 16-lane subgroups); elsewhere the flag changes nothing.
RUNS = tv._native.kernel_subgroup_size("gemm4s_f32") == 16 and bool(tv._native.capabilities() & 4)


@contextlib.contextmanager
def fp16_arithmetic(on=True):
    before = tv.get_fp16_arithmetic()
    tv.set_fp16_arithmetic(on)
    try:
        yield
    finally:
        tv.set_fp16_arithmetic(before)


def within_float16_sums(got, want, mag, k):
    # recursive float16 summation of k products: |error| <= about k eps
    # times the sum of the products' magnitudes (a worst-case bound), plus
    # one rounding of the result
    eps = torch.finfo(torch.float16).eps
    err = (got.cpu().double() - want).abs()
    tol = (k + 2) * eps * mag + eps * want.abs() + 2**-24
    assert (err <= tol).all(), float((err / tol).max())


def test_flag():
    assert tv.get_fp16_arithmetic() in (True, False)
    with fp16_arithmetic(True):
        assert tv.get_fp16_arithmetic()
    with fp16_arithmetic(False):
        assert not tv.get_fp16_arithmetic()


@pytest.mark.parametrize("m,k,n", [(64, 32, 64), (65, 36, 63), (17, 200, 129), (300, 1408, 257), (40, 12, 9)])
def test_matmul(m, k, n):
    g = torch.Generator().manual_seed(m + k + n)
    a = torch.randn(m, k, generator=g).div(k**0.5).half()
    b = torch.randn(k, n, generator=g).half()
    c = torch.randn(m, n, generator=g).half()
    want, mag = a.double() @ b.double(), a.double().abs() @ b.double().abs()
    with fp16_arithmetic(False):
        off = a.to("vk") @ b.to("vk")
    with fp16_arithmetic(True):
        on = a.to("vk") @ b.to("vk")
        on_t = a.to("vk") @ b.t().contiguous().to("vk").t()
        add = torch.addmm(c.to("vk"), a.to("vk"), b.to("vk"), beta=0.5, alpha=2)
    with fp16_arithmetic(False):
        assert torch.equal((a.to("vk") @ b.to("vk")).cpu(), off.cpu())  # (the default is back)
    within_float16_sums(on, want, mag, k)
    within_float16_sums(on_t, want, mag, k)
    within_float16_sums(add, 2 * want + 0.5 * c.double(), 2 * mag + 0.5 * c.double().abs(), k)
    if RUNS and k >= 200:
        assert not torch.equal(on.cpu(), off.cpu())  # (float16 sums did run)


@pytest.mark.parametrize("dims,n,cin,cout,k,pad,dil,groups,spatial", [
    (1, 1, 128, 64, 11, 5, 1, 1, (300,)),
    (1, 2, 16, 24, 7, 9, 3, 2, (101,)),
    (2, 1, 8, 16, 3, 1, 1, 1, (20, 21)),
])
def test_conv(dims, n, cin, cout, k, pad, dil, groups, spatial):
    g = torch.Generator().manual_seed(cin + k)
    x = torch.randn((n, cin) + spatial, generator=g).half()
    w = torch.randn((cout, cin // groups) + (k,) * dims, generator=g).div((cin * k) ** 0.5).half()
    b = torch.randn(cout, generator=g).half()
    conv = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
    want = conv(x.double(), w.double(), b.double(), 1, pad, dil, groups)
    mag = conv(x.double().abs(), w.double().abs(), b.double().abs(), 1, pad, dil, groups)
    with fp16_arithmetic(True):
        on = conv(x.to("vk"), w.to("vk"), b.to("vk"), 1, pad, dil, groups)
    within_float16_sums(on, want, mag, cin // groups * k**dims)


def test_overflow():
    # float16 sums overflow to inf past 65504, as the products' sums would
    a = torch.full((16, 64), 40.0).half()
    with fp16_arithmetic(True):
        got = (a.to("vk") @ a.t().contiguous().to("vk")).cpu()
    assert torch.isinf(got).all()
