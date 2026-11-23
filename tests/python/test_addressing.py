"""Tensors past 4 GiB and 2^31 elements (docs/log/M56, #142).

Kernels index tensors with 32-bit integers from the addresses they are
given. A dispatch that can reach more than 4 GiB past one of them runs its
kernel with 64-bit address arithmetic (NVIDIA's driver wrapped those offsets
at 4 GiB); offsets past 2^31 elements are run in pieces or refused."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import torch_vulkan as tv
from _util import on_gpu

GIB = 2**30


def big_tensor_or_skip(nbytes):
    """Skips on lavapipe (the host's memory, more than CI has) and where the
    device cannot hold `nbytes` more."""
    if "llvmpipe" in tv.device_name():
        pytest.skip("lavapipe: tensors past 4 GiB take the host's memory")
    try:
        probe = torch.empty(nbytes // 4, device="vk")
    except RuntimeError as e:
        pytest.skip(f"cannot allocate {nbytes / GIB:.1f} GiB: {str(e)[:200]}")
    del probe
    tv.empty_cache()


def test_tensor_past_4_gib():
    # (one float32 tensor of 4 GiB + 4 MiB, each kernel given its start:
    # elements past 2^30 are 4 GiB or more past that address)
    n = 4 * 1024 * 262400  # 2^30 + 2^20
    big_tensor_or_skip(n * 4 + GIB)
    x = torch.empty(n, device="vk")
    with on_gpu():
        x.fill_(1.0)
        assert x[2**30:].sum().item() == 2**20  # (a wrapped fill left these unwritten)
        x.mul_(3.0)
        assert x[-3:].tolist() == [3.0, 3.0, 3.0]
        assert x.sum().item() == pytest.approx(3.0 * n, rel=1e-6)  # (a wrapped sum read the start twice)
        assert x.max().item() == 3.0
        # matrices k + 1, the last one past 4 GiB
        b = x.view(4, 1024, 262400)
        for k in range(4):
            b[k].fill_(k + 1.0)  # (each through its own address)
        y = torch.bmm(torch.ones(4, 2, 1024, device="vk"), b)
        want = [1024.0 * (k + 1) for k in range(4)]
        assert y.amin(dim=(1, 2)).tolist() == want and y.amax(dim=(1, 2)).tolist() == want
        torch.arange(n, device="vk", dtype=torch.float32, out=x)
        for i in (2**30 - 1000, 2**30 + 1000, n - 1):
            assert x[i].item() == torch.tensor(i, dtype=torch.float32).item()  # (float32's rounding of i)


@pytest.mark.parametrize("N", [2, 4])
def test_convolution_with_columns_past_4_gib(N):
    # (#142: pytorch_unet's Conv2d(128, 64, 3, padding=1) on [2, 128, 640,
    # 959] makes im2col columns of 5.27 GiB, 1.4e9 elements; at batch 4 they
    # pass 2^31 elements and run in pieces of the batch. Sample n, channel c
    # holds (n + 1)(c + 1) / 128, so a column read from the wrong place shows.)
    big_tensor_or_skip((N * 4 + 1) * GIB)
    C, H, W = 128, 640, 959
    scale = torch.arange(1, N + 1).view(N, 1, 1, 1) * torch.arange(1, C + 1).view(1, C, 1, 1) / 128.0
    x = scale.to("vk", torch.float32).expand(N, C, H, W).contiguous()
    w = torch.ones(64, C, 3, 3, device="vk")
    # (the gradient: ones every 64th output row, so that every sum is exact)
    g = torch.zeros(N, 64, H, W, device="vk")
    g[:, :, ::64] = 1
    with on_gpu():
        y = F.conv2d(x, w, padding=1)
        gw = torch.ops.aten.convolution_backward(g, x, w, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1,
                                                 [False, True, False])[1]
    # out[n, o, i, j] = (n + 1) (C + 1) / 2 * (taps inside the image at i, j)
    taps = F.conv2d(torch.ones(1, 1, H, W), torch.ones(1, 1, 3, 3), padding=1)[0, 0]
    for n in range(N):
        want = (n + 1) * (C + 1) / 2 * taps
        for o in (0, 63):
            torch.testing.assert_close(y[n, o].cpu(), want, rtol=1e-6, atol=0)
    # grad_weight[o, c] = sum_n (n + 1)(c + 1) / 128 * (positions the gradient's ones reach through each tap)
    reach = torch.ops.aten.convolution_backward(g[:1, :1].cpu().double(), torch.ones(1, 1, H, W, dtype=torch.float64),
                                                torch.ones(1, 1, 3, 3, dtype=torch.float64), None, [1, 1], [1, 1],
                                                [1, 1], False, [0, 0], 1, [False, True, False])[1][0, 0]
    want = N * (N + 1) / 2 * (torch.arange(1, C + 1, dtype=torch.float64) / 128).view(C, 1, 1) * reach
    for o in (0, 63):
        torch.testing.assert_close(gw[o].cpu().double(), want, rtol=1e-6, atol=0)


def test_wide_kernels_need_int64():
    # (a device without shaderInt64, TORCH_VK_DISABLE=int64, with every
    # dispatch taking the 64-bit addresses, TORCH_VK_WIDE_ADDRESSES=1: refused
    # by name rather than run with 32-bit offsets)
    script = "import torch, torch_vulkan\nprint(torch.ones(3, device='vk').cpu())\n"
    env = {**os.environ, "TORCH_VK_DISABLE": "int64", "TORCH_VK_WIDE_ADDRESSES": "1"}
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode != 0
    assert "more than 4 GiB past a tensor's address, which needs shaderInt64" in out.stderr, out.stderr[-2000:]


def test_wide_kernels():
    # (TORCH_VK_WIDE_ADDRESSES=1: every kernel with 64-bit address arithmetic,
    # as dispatches reaching past 4 GiB get them; tests of element-wise ops,
    # copies, indexing, reductions, sorting and products, in a fresh process)
    if os.environ.get("TORCH_VK_WIDE_ADDRESSES"):
        pytest.skip("TORCH_VK_WIDE_ADDRESSES is set: every test takes those kernels already")
    here = os.path.dirname(__file__)
    files = [os.path.join(here, f) for f in ("test_core.py", "test_indexing.py", "test_sorting.py")]
    sel = "not test_wide and not past_4_gib and not subprocess and not environment"
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *files, "-k", sel],
                         env={**os.environ, "TORCH_VK_WIDE_ADDRESSES": "1"}, capture_output=True, text=True,
                         timeout=1800)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
