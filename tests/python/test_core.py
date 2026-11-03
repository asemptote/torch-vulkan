"""Transfers, factories, views, out= resizing, CPU fallback and autograd on device "vk"."""

import contextlib
import itertools
import math

import pytest
import torch

import torch_vulkan as tv
from _util import ALL, COMPLEX, FLOATS, NUMERIC, WIDE, check, make, make_complex, on_gpu


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





def test_factories():
    with on_gpu():
        z = torch.zeros(3, 4, device="vk")
        o = torch.ones(2, dtype=torch.int64, device="vk")
        f = torch.full((2, 2), 7.5, device="vk", dtype=WIDE)
    check(z, torch.zeros(3, 4))
    check(o, torch.ones(2, dtype=torch.int64))
    check(f, torch.full((2, 2), 7.5, dtype=WIDE))
    check(torch.arange(2, 11, 3, device="vk"), torch.arange(2, 11, 3))
    check(torch.eye(3, device="vk"), torch.eye(3))
    r = torch.randn(100, device="vk")
    assert r.shape == (100,) and r.device.type == "vk" and r.std().item() > 0.5


def test_cpu_fallback_matches_and_is_counted():
    x = make((20,), WIDE)
    x = x.abs() + 0.5
    x = x.view(4, 5)
    before = tv.fallback_counts().get("aten::unique_dim", 0)
    with pytest.warns(UserWarning, match="no Vulkan kernel") if before == 0 else contextlib.nullcontext():
        got = torch.unique(x.to("vk"), dim=0)  # no unique(dim=...) kernel yet
    check(got, torch.unique(x, dim=0))
    assert tv.fallback_counts()["aten::unique_dim"] == before + 1
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


def test_fallback_with_several_outputs_writes_them_back():
    # frexp has no Vulkan kernel; its CPU results must reach the vk outputs.
    x = torch.tensor([0.0, 1.0, 0.75, -3.5, 1e-40, float("inf"), 1024.0])
    m, e = torch.frexp(x.to("vk"))
    wm, we = torch.frexp(x)
    assert torch.equal(m.cpu(), wm) and torch.equal(e.cpu(), we)
    mo, eo = torch.empty(7, device="vk"), torch.empty(7, dtype=torch.int32, device="vk")
    torch.frexp(x.to("vk"), out=(mo, eo))
    assert torch.equal(mo.cpu(), wm) and torch.equal(eo.cpu(), we)
