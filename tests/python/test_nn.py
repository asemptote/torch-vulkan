"""Neural-network ops on "vk" vs the CPU (M12), forward and backward."""

import contextlib
import itertools
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from _util import MAX_NUMEL, check, make, on_gpu

import torch_vulkan as tv

TOL = dict(rtol=2e-4, atol=2e-4)


def grads(fn, *tensors):
    """fn's output and the gradients of sum(output * r) w.r.t. tensors (r fixed)."""
    leaves = [t.detach().requires_grad_() for t in tensors]
    out = fn(*leaves)
    r = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(out.device, out.dtype)
    (out * r).sum().backward()
    return [out.detach()] + [t.grad for t in leaves]


def compare_with_cpu(fn, *tensors, tol=TOL):
    want = grads(fn, *tensors)
    with on_gpu():
        got = grads(fn, *(t.to("vk") for t in tensors))
    for g, w in zip(got, want):
        check(g, w, **tol)


CONV_CASES = [
    # dims, in_ch, out_ch, kernel, stride, padding, dilation, groups
    (1, 3, 5, 3, 1, 1, 1, 1),
    (1, 4, 6, 4, 2, 0, 2, 2),
    (2, 3, 8, 3, 1, 1, 1, 1),
    (2, 4, 4, (3, 2), (2, 1), (1, 0), (1, 2), 4),
    (2, 6, 9, 5, 3, 2, 1, 3),
    (3, 2, 4, 3, 2, 1, 1, 1),
    # depthwise (one input channel per group: multiply-adds per tap)
    (1, 6, 12, 5, 2, 2, 2, 6),
    (3, 3, 3, 2, 1, 1, 1, 3),
]


@pytest.mark.parametrize("dims,cin,cout,k,stride,pad,dil,groups", CONV_CASES)
def test_conv(dims, cin, cout, k, stride, pad, dil, groups):
    spatial = (11, 9, 7)[:dims]
    x = make((2, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cout, cin // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, dil, groups), x, w, b)


@pytest.mark.parametrize("dims,cin,cout,k,stride,pad,dil,groups", [
    (1, 6, 9, 3, 1, 1, 1, 3), (1, 16, 12, 7, 1, 9, 3, 1), (2, 4, 6, (3, 2), (2, 1), 1, 1, 2)])
def test_conv_one_sample(dims, cin, cout, k, stride, pad, dil, groups):
    # batches of one: the products run per group, the columns as rows along
    # the output positions (convolution.cpp), here with lengths and K not
    # multiples of 4
    spatial = (301, 9, 7)[:dims]
    x = make((1, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cout, cin // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, dil, groups), x, w, b)
    compare_with_cpu(lambda x, w: conv(x, w, None, stride, pad, dil, groups), x, w)


@pytest.mark.parametrize("dims,cin,cout,k,stride,pad,dil,groups,outpad", [
    (1, 3, 5, 3, 2, 1, 1, 1, 1),
    (2, 4, 6, 3, 2, 1, 1, 2, (1, 0)),
    (2, 3, 4, (2, 4), (3, 1), (0, 2), (2, 1), 1, (2, 0)),
    (3, 2, 2, 3, 1, 0, 1, 2, 0),
])
def test_conv_transpose(dims, cin, cout, k, stride, pad, dil, groups, outpad):
    spatial = (6, 5, 4)[:dims]
    x = make((2, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cin, cout // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, outpad, groups, dil), x, w, b)


@pytest.mark.parametrize("dims,n,cin,cout,k,stride,pad,dil,groups,outpad,length", [
    # Kokoro's upsampling convolutions (scaled down): K a multiple of the stride
    (1, 1, 8, 4, 20, 10, 5, 1, 1, 0, 9),
    (1, 1, 6, 4, 12, 6, 3, 1, 1, 0, 9),
    # depthwise stride-2 upsampling, K not a multiple of the stride
    (1, 2, 6, 6, 3, 2, 1, 1, 6, 1, 13),
    # padding past T - 1 (the phase convolution crops its input), output_padding
    (1, 2, 3, 5, 5, 2, 7, 1, 1, 1, 17),
    # dilation coprime to the stride (phases with different taps), output_padding >= stride
    (1, 2, 4, 3, 4, (3,), 2, 2, 1, 1, 10),
    (1, 1, 4, 3, 5, 2, 0, 3, 1, 2, 7),
    # 2-D: strided and stride-1 dims mixed, dilation on the stride-1 dim, groups
    (2, 2, 4, 6, (4, 3), (2, 1), (1, 2), (1, 2), 2, (1, 0), 6),
    (2, 1, 3, 4, 4, 2, 1, 1, 1, 0, 7),
    (2, 2, 4, 4, (5, 3), (3, 2), (2, 0), 1, 4, (2, 1), 5),
    (3, 1, 2, 3, 3, 2, 1, 1, 1, 1, 4),
])
def test_conv_transpose_phases(dims, n, cin, cout, k, stride, pad, dil, groups, outpad, length):
    # strided transposed convolutions run as one convolution per phase
    # (convolution.cpp: conv_transpose_phases)
    spatial = (length, length - 1, length - 2)[:dims]
    x = make((n, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cin, cout // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, outpad, groups, dil), x, w, b)
    compare_with_cpu(lambda x, w: conv(x, w, None, stride, pad, outpad, groups, dil), x, w)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32])
def test_conv_transpose_phases_dtypes(dtype):
    x = make((1, 8, 23), torch.float32)
    w = make((8, 4, 12), torch.float32, 1)
    b = make((4,), torch.float32, 2)
    if dtype == torch.int32:
        x, w, b = (t.mul(4).round().to(dtype) for t in (x, w, b))
        tol = dict(rtol=0, atol=0)
    else:
        x, w, b = (t.to(dtype) for t in (x, w, b))
        tol = dict(rtol=1e-2, atol=1e-2)
    want = F.conv_transpose1d(x.float(), w.float(), b.float(), 6, 3).to(dtype)
    with on_gpu():
        got = F.conv_transpose1d(x.to("vk"), w.to("vk"), b.to("vk"), 6, 3)
    check(got, want, **tol)


def depthwise_case(dims, mult, k, stride, pad, dil, n, length):
    """A depthwise convolution (5 channels, mult outputs each) and its inputs."""
    c = 5
    spatial = (length, length - 2, length - 1)[:dims]
    x = make((n, c) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((c * mult, 1) + ks, torch.float32, 1)
    b = make((c * mult,), torch.float32, 2)
    conv = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
    return (lambda x, w, b: conv(x, w, b, stride, pad, dil, c)), x, w, b


@pytest.mark.parametrize("dims,mult,k,stride,pad,dil,n,length", [
    (1, 1, 3, 1, 1, 1, 2, 30), (1, 3, 31, 1, 15, 1, 1, 64), (1, 2, 5, 3, 0, 2, 2, 41),
    (2, 1, 3, 1, 1, 1, 2, 12), (2, 1, 3, 2, 1, 1, 3, 13), (2, 2, (7, 5), 1, (3, 2), 1, 1, 15),
    (2, 1, (3, 1), (1, 2), (0, 0), (2, 1), 2, 9), (3, 1, 3, 1, 1, 1, 1, 7), (3, 2, (2, 3, 1), 2, 1, (1, 2, 1), 2, 8),
])
def test_conv_depthwise(dims, mult, k, stride, pad, dil, n, length):
    # one input channel per group, mult outputs each: shaders/dwconv.slang,
    # forward (one launch) and backward (the weight's gradient in one launch);
    # taps past 32, strides, dilations
    fn, x, w, b = depthwise_case(dims, mult, k, stride, pad, dil, n, length)
    compare_with_cpu(fn, x, w, b)
    compare_with_cpu(lambda x, w: fn(x, w, None), x, w)
    with on_gpu():
        before = tv._native.dispatch_count()
        fn(x.to("vk"), w.to("vk"), b.to("vk"))
        assert tv._native.dispatch_count() - before == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int32, torch.int64,
                                   pytest.param(torch.float64, marks=pytest.mark.float64)])
def test_conv_depthwise_dtypes(dtype):
    x = make((2, 6, 17, 13), torch.float32)
    w = make((12, 1, 3, 3), torch.float32, 1)
    b = make((12,), torch.float32, 2)
    if dtype in (torch.int32, torch.int64):
        x, w, b = (t.mul(4).round().to(dtype) for t in (x, w, b))
        want = F.conv2d(x.double(), w.double(), b.double(), 2, 1, 1, 6).to(dtype)
        tol = dict(rtol=0, atol=0)
    elif dtype == torch.float64:
        x, w, b = (t.to(dtype) for t in (x, w, b))
        want = F.conv2d(x, w, b, 2, 1, 1, 6)
        tol = dict(rtol=1e-12, atol=1e-12)
    else:
        # float32 sums rounded once: within one rounding of the exact result
        x, w, b = (t.to(dtype) for t in (x, w, b))
        want = F.conv2d(x.double(), w.double(), b.double(), 2, 1, 1, 6).to(dtype)
        tol = dict(rtol=torch.finfo(dtype).eps, atol=1e-3)
    with on_gpu():
        got = F.conv2d(x.to("vk"), w.to("vk"), b.to("vk"), 2, 1, 1, 6)
    check(got, want, **tol)


def test_conv_depthwise_grouped_products():
    # TORCH_VK_DWCONV=0: depthwise convolutions as grouped products (the
    # other route, for comparison), in a subprocess
    script = ("import sys; sys.path.insert(0, %r)\n" % os.path.dirname(__file__) +
              "import test_nn as t\n"
              "for case in [(1, 3, 5, 1, 2, 1, 2, 30), (2, 2, (7, 5), 1, (3, 2), 1, 1, 15), (3, 1, 3, 1, 1, 1, 1, 7)]:\n"
              "    t.compare_with_cpu(*t.depthwise_case(*case))\n"
              "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", script], env=dict(os.environ, TORCH_VK_DWCONV="0"),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "ok"


def test_conv_depthwise_layouts_and_nonfinite():
    # channels-last and strided inputs; an infinite or NaN weight does not
    # reach the outputs whose window meets it only in the padding, as in
    # oneDNN's depthwise convolution (the CPU's on x86)
    x = make((2, 4, 9, 8), torch.float32)
    w = make((4, 1, 3, 3), torch.float32, 1)
    backend = torch._C._select_conv_backend(x, w, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 4, None)
    if backend != torch._C._ConvBackend.Mkldnn:
        pytest.skip(f"the CPU's convolution here is {backend}, not oneDNN's")
    w[1, 0, 0, 0] = float("inf")
    w[2, 0, 2, 1] = float("nan")
    for xi in (x.contiguous(memory_format=torch.channels_last), x[:, :, ::2], x.transpose(2, 3)):
        want = F.conv2d(xi, w, None, 1, 1, 1, 4)
        with on_gpu():
            got = F.conv2d(xi.to("vk"), w.to("vk"), None, 1, 1, 1, 4)
        check(got, want, **TOL, equal_nan=True)


@pytest.mark.float64
def test_conv_float64():
    x, w = make((2, 3, 10, 10), torch.float64), make((4, 3, 3, 3), torch.float64, 1)
    compare_with_cpu(lambda x, w: F.conv2d(x, w, padding=1), x, w, tol=dict(rtol=1e-10, atol=1e-10))


def test_norm_layers():
    x = make((4, 6, 5, 3), torch.float32)
    w, b = make((6,), torch.float32, 1), make((6,), torch.float32, 2)
    compare_with_cpu(lambda x, w, b: F.batch_norm(x, None, None, w, b, training=True), x, w, b)
    compare_with_cpu(lambda x, w, b: F.group_norm(x, 3, w, b), x, w, b)
    lw, lb = make((5, 3), torch.float32, 3), make((5, 3), torch.float32, 4)
    compare_with_cpu(lambda x, w, b: F.layer_norm(x, (5, 3), w, b), x, lw, lb)
    rm, rv = torch.zeros(6), torch.ones(6)
    with on_gpu():
        vrm, vrv = rm.to("vk"), rv.to("vk")
        F.batch_norm(x.to("vk"), vrm, vrv, training=True, momentum=0.3)
    F.batch_norm(x, rm, rv, training=True, momentum=0.3)
    check(vrm, rm, **TOL)
    check(vrv, rv, **TOL)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("params", ["same", "float32", "none"])
def test_batch_norm_with_update_16bit(dtype, params):
    # (issue #95) as the CPU: a 16-bit input without float32 parameters is
    # normalised with its statistics rounded to its dtype (a mean far from 0
    # shows it: 17 float16 ulps before), and the running statistics are
    # updated in float32 and rounded once. The saved statistics stay float32
    # on vk, as PyTorch's meta kernel says for devices other than the CPU.
    # (The CPU's bfloat16 running statistics are computed in bfloat16
    # arithmetic, so they may differ by an ulp or two.)
    if params == "float32" and dtype == torch.float32:
        pytest.skip("not a mixed case")
    pdt = dtype if params == "same" else torch.float32
    for shape, loc, scale in [((4, 3, 5, 5), 0.0, 1.0), ((8, 4, 6), 5.0, 0.5), ((16, 5), 20.0, 3.0),
                              ((3, 2, 4, 4, 2), -7.0, 0.1)]:
        c = shape[1]
        x = (make(shape, torch.float32) * scale + loc).to(dtype)
        w, b = (make((c,), torch.float32, 1).to(pdt), make((c,), torch.float32, 2).to(pdt))
        if params == "none":
            w, b = None, None
        rm, rv = (make((c,), torch.float32, 3) * 3).to(pdt), (make((c,), torch.float32, 4).abs() + 0.5).to(pdt)
        vrm, vrv = rm.to("vk"), rv.to("vk")
        want = torch.ops.aten._batch_norm_with_update(x, w, b, rm, rv, 0.1, 1e-5)
        with on_gpu():
            got = torch.ops.aten._batch_norm_with_update(x.to("vk"), None if w is None else w.to("vk"),
                                                         None if b is None else b.to("vk"), vrm, vrv, 0.1, 1e-5)
        eps = torch.finfo(dtype).eps
        # (16-bit: an ulp or two, for statistics on either side of a rounding
        # boundary; float32: x - mean cancels to 1 / 70 of x at loc -7, scale 0.1)
        tol = dict(rtol=2 * eps, atol=2 * eps) if dtype != torch.float32 else dict(rtol=2e-5, atol=2e-5)
        check(got[0], want[0], **tol)
        for g, e in zip(got[1:3], want[1:3]):
            assert g.dtype == torch.float32
            torch.testing.assert_close(g.cpu().to(e.dtype), e, **tol)
        check(vrm, rm, **tol)
        check(vrv, rv, **tol)


def test_softmax_and_losses():
    x = make((6, 7), torch.float32)
    target = torch.randint(0, 7, (6,))
    y = make((6, 7), torch.float32, 1)
    compare_with_cpu(lambda x: F.softmax(x, 1), x)
    compare_with_cpu(lambda x: F.log_softmax(x, 0), x)
    compare_with_cpu(lambda x: F.cross_entropy(x, target.to(x.device)), x)
    compare_with_cpu(lambda x: F.nll_loss(F.log_softmax(x, 1), target.to(x.device), reduction="sum"), x)
    compare_with_cpu(lambda x, y: F.mse_loss(x, y), x, y)
    compare_with_cpu(lambda x, y: F.smooth_l1_loss(x, y, beta=0.5), x, y)
    compare_with_cpu(lambda x, y: F.huber_loss(x, y), x, y)
    p = torch.sigmoid(x)
    compare_with_cpu(lambda p, y: F.binary_cross_entropy(p, torch.sigmoid(y)), p, y)
    compare_with_cpu(lambda x, y: F.binary_cross_entropy_with_logits(x, torch.sigmoid(y)), x, y)
    compare_with_cpu(lambda x, y: F.kl_div(F.log_softmax(x, 1), F.softmax(y, 1), reduction="batchmean"), x, y)


def test_activation_backward():
    x = make((5, 9), torch.float32) * 3
    for fn in [F.relu, F.gelu, lambda t: F.gelu(t, approximate="tanh"), F.silu, torch.sigmoid, torch.tanh, F.elu,
               F.softplus, F.hardtanh, F.hardswish, F.hardsigmoid, F.leaky_relu, F.mish, F.logsigmoid, F.selu]:
        compare_with_cpu(fn, x)


def test_pooling():
    x = make((2, 3, 11, 10), torch.float32)
    compare_with_cpu(lambda x: F.max_pool2d(x, 3, 2, 1), x)
    compare_with_cpu(lambda x: F.avg_pool2d(x, 3, 2, 1), x)
    compare_with_cpu(lambda x: F.avg_pool2d(x, (2, 3), (2, 2), count_include_pad=False, ceil_mode=True), x)
    compare_with_cpu(lambda x: F.adaptive_avg_pool2d(x, (4, 3)), x)
    compare_with_cpu(lambda x: F.adaptive_max_pool2d(x, (5, 4)), x)
    x1 = make((2, 3, 13), torch.float32)
    compare_with_cpu(lambda x: F.max_pool1d(x, 4, 3), x1)
    compare_with_cpu(lambda x: F.avg_pool1d(x, 3), x1)


def test_upsample_and_grid_sample():
    x = make((2, 3, 5, 6), torch.float32)
    for mode, ac in [("nearest", None), ("bilinear", False), ("bilinear", True), ("bicubic", False)]:
        compare_with_cpu(lambda x: F.interpolate(x, size=(9, 7), mode=mode, align_corners=ac), x)
    grid = (torch.rand(2, 4, 5, 2, generator=torch.Generator().manual_seed(3)) * 2.2 - 1.1)
    for mode, pad in itertools.product(["bilinear", "nearest"], ["zeros", "border", "reflection"]):
        compare_with_cpu(lambda x: F.grid_sample(x, grid.to(x.device), mode=mode, padding_mode=pad,
                                                 align_corners=False), x)
    # bicubic: grad_input and grad_grid
    for pad, ac in itertools.product(["zeros", "border", "reflection"], [False, True]):
        compare_with_cpu(lambda x, g: F.grid_sample(x, g, mode="bicubic", padding_mode=pad, align_corners=ac),
                         x, grid.clone(), tol=dict(rtol=1e-4, atol=1e-4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grid_sample_3d_bicubic(dtype):
    # (PyTorch 2.15 on: bicubic 5-D grid_sample, 64 taps; grad_input and grad_grid)
    x = make((2, 3, 4, 5, 6), dtype)
    grid = (torch.rand(2, 3, 4, 2, 3, generator=torch.Generator().manual_seed(4)) * 2.4 - 1.2).to(dtype)
    tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
    for pad, ac in itertools.product(["zeros", "border", "reflection"], [False, True]):
        compare_with_cpu(lambda x, g: F.grid_sample(x, g, mode="bicubic", padding_mode=pad, align_corners=ac),
                         x, grid.clone(), tol=tol)


@pytest.mark.float64
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
def test_grid_sample_bicubic_non_finite(padding_mode):
    # (a tap that zeros padding drops reads nothing: an infinite voxel beside
    # it does not reach the result, at either rank, as on the CPU)
    image = make((1, 1, 5, 5), torch.float64)
    image[0, 0, 0, 0] = float("inf")
    volume = image.unsqueeze(2).expand(1, 1, 5, 5, 5).contiguous()
    for coordinate in (float("nan"), float("inf"), -float("inf"), 50.0, -1.3):
        grid_2d = torch.full((1, 1, 2, 2), coordinate, dtype=torch.float64)
        grid_2d[0, 0, 1] = 0.1
        grid_3d = torch.cat([grid_2d, torch.full_like(grid_2d[..., :1], 0.1)], dim=-1).unsqueeze(1)
        for x, g in ((image, grid_2d), (volume, grid_3d)):
            want = F.grid_sample(x, g, mode="bicubic", padding_mode=padding_mode, align_corners=False)
            with on_gpu():
                got = F.grid_sample(x.to("vk"), g.to("vk"), mode="bicubic", padding_mode=padding_mode,
                                    align_corners=False)
            torch.testing.assert_close(got.cpu(), want, equal_nan=True)


def test_grid_unnormalise_rounds_once():
    # (bicubic grid sampling unnormalises as the CPU, whose compiler fuses
    # (g + 1) * size / 2 - 0.5 into one rounding: from float32 operations,
    # the exactly rounded value; checked against float64 arithmetic)
    from torch_vulkan._kernels import _fused_multiply_add

    g = torch.rand(100000, generator=torch.Generator().manual_seed(5)) * 2 - 1
    a = g + 1
    for size in (5, 345, 456, 4097, 2 ** 23 - 3):
        with on_gpu():
            got = _fused_multiply_add(a.to("vk"), size / 2, -0.5).cpu()
        want = (a.double() * (size / 2) - 0.5).float()
        assert (got != want).sum() <= 2, size  # (a double rounding next to a tie, at most)
    # 2-D bicubic of 0-255 samples, where the 16 taps cancel: within the
    # tolerance of PyTorch's test_compare_cpu
    x = torch.randint(0, 256, (1, 2, 120, 160), generator=torch.Generator().manual_seed(6)).float()
    grid = torch.rand(1, 120, 160, 2, generator=torch.Generator().manual_seed(7)) * 2 - 1
    want = F.grid_sample(x, grid, mode="bicubic", padding_mode="zeros", align_corners=False)
    with on_gpu():
        got = F.grid_sample(x.to("vk"), grid.to("vk"), mode="bicubic", padding_mode="zeros", align_corners=False)
    torch.testing.assert_close(got.cpu(), want, atol=1e-3, rtol=1e-3)


def launches(fn):
    before = tv._native.dispatch_count()
    out = fn()
    return out, tv._native.dispatch_count() - before


@pytest.mark.parametrize("k", [2, 3])
def test_grid_sample_bicubic_one_launch(k):
    # (shaders/gridsample.slang's bicubic mode: the forward in one launch, the
    # backward's grad_grid with the taps for grad_input's scatter in one; an
    # input with strides read in place)
    x = make((2, 3) + (6, 7, 5)[:k], torch.float32)
    g = make((2,) + (4, 3, 2)[:k] + (k,), torch.float32, 1)
    layouts = (lambda t: t, lambda t: t.transpose(-1, -2).contiguous().transpose(-1, -2),
               lambda t: t[:1].expand(2, *t.shape[1:]))
    for layout in layouts:
        want = F.grid_sample(layout(x), g, mode="bicubic", padding_mode="reflection", align_corners=False)
        with on_gpu():
            vx, vg = layout(x.to("vk")), g.to("vk")
            got, n = launches(lambda: F.grid_sample(vx, vg, mode="bicubic", padding_mode="reflection",
                                                    align_corners=False))
        check(got, want, **TOL)
        assert n == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_upsample_bicubic_kernel(dtype):
    # (upsample_bicubic2d by gridsample.slang's upsample mode: the CPU's source
    # coordinates, clamped taps; forward in one launch; the backward by the
    # taps' weights and one scatter-add)
    x = make((2, 3, 7, 9), dtype)
    for kw in (dict(scale_factor=2), dict(size=(5, 4)), dict(size=(13, 20), align_corners=True),
               dict(scale_factor=(1.7, 0.6)), dict(size=(1, 1)), dict(size=(7, 9), align_corners=True)):
        kw.setdefault("align_corners", False)
        compare_with_cpu(lambda x: F.interpolate(x, mode="bicubic", **kw), x)
        with on_gpu():
            vx = x.to("vk")
            _, n = launches(lambda: F.interpolate(vx, mode="bicubic", **kw))
        assert n <= 4, n  # (the coordinates' stack and copy, and the kernel)


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
def test_grid_sample_reflection_whole_spans(mode):
    # (a coordinate a whole number of spans from the low edge: the reflection
    # counts its spans with a correctly rounded division, where the GPU's may
    # round 455 / 455 just below 1, the reference input's width minus 1)
    for width in (6, 100, 455, 456, 457, 1000):
        x = torch.arange(2 * width, dtype=torch.float32).reshape(1, 1, 2, width)
        g = torch.tensor([1.0, 1.0006271600723267, -1.0, 3.0, 0.5]).reshape(1, 1, 5, 1)
        grid = torch.cat([g, torch.full_like(g, 0.3)], -1)
        want = F.grid_sample(x, grid, mode=mode, padding_mode="reflection", align_corners=True)
        with on_gpu():
            got = F.grid_sample(x.to("vk"), grid.to("vk"), mode=mode, padding_mode="reflection", align_corners=True)
        check(got, want, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kernel,dilation,padding,stride,size", [
    (3, 1, 1, 1, (9, 8)), ((2, 3), (2, 1), (0, 2), (2, 3), (11, 10)), (4, 1, 0, 4, (8, 12)), (1, 1, 0, 1, (3, 5))])
def test_fold_unfold(dtype, kernel, dilation, padding, stride, size):
    # fold (col2im) by shaders/col2im.slang: each output gathers its blocks'
    # values in the CPU's order, so the sums are the CPU's bit for bit;
    # unfold's backward is fold
    x = make((2, 3) + size, dtype)
    blocks = dict(kernel_size=kernel, dilation=dilation, padding=padding, stride=stride)
    cols = F.unfold(x, **blocks)
    c = make(tuple(cols.shape), dtype, 1)
    want = F.fold(c, size, **blocks)
    with on_gpu():
        got, n = launches(lambda: F.fold(c.to("vk"), size, **blocks))
        unbatched = F.fold(c[0].to("vk"), size, **blocks)
    assert torch.equal(got.cpu(), want) and n == 1
    assert torch.equal(unbatched.cpu(), want[0])
    compare_with_cpu(lambda x: F.unfold(x, **blocks), x)
    compare_with_cpu(lambda c: F.fold(c, size, **blocks), c)


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_nearest_half_way(align_corners):
    # (2-D and 3-D unnormalise the grid with the CPU's arithmetic, so samples at
    # half-way coordinates round the same way at both ranks)
    n = 10
    i = torch.arange(0, n - 1 + 0.1, 0.1, dtype=torch.float32)
    q = 2 * i / (n - 1) - 1 if align_corners else (i * 2 + 1) / n - 1
    fixed = torch.full_like(q, 0.5)
    data = make((n,), torch.float32)
    x2, g2 = data.reshape(1, n).repeat(1, 1, 9, 1), torch.stack([q, fixed], 1).reshape(1, 1, -1, 2)
    x3, g3 = data.reshape(1, n).repeat(1, 1, 9, 9, 1), torch.stack([q, fixed, fixed], 1).reshape(1, 1, 1, -1, 3)
    for pad in ("zeros", "border", "reflection"):
        for x, g in ((x2, g2), (x3, g3)):
            want = F.grid_sample(x, g, mode="nearest", padding_mode=pad, align_corners=align_corners)
            with on_gpu():
                got = F.grid_sample(x.to("vk"), g.to("vk"), mode="nearest", padding_mode=pad,
                                    align_corners=align_corners)
            assert torch.equal(got.cpu(), want), (pad, x.dim())


@pytest.mark.parametrize("size", [2 ** 24 + 1, 2 ** 31 + 1, 2 ** 53 + 1])
def test_grid_sample_3d_bicubic_wide_expanded_axis(size):
    # (an expanded axis wider than the indexing kernels address: every tap
    # reads its one element)
    if size > MAX_NUMEL:
        pytest.skip("more elements than this build of PyTorch allows (32-bit)")
    volume = torch.ones(1, 1, 1, 1, 1, device="vk").expand(1, 1, 1, 1, size)
    grid = torch.tensor([[[[[1.0, 0.0, 0.0]]]]], device="vk")
    for pad in ("zeros", "border", "reflection"):
        with on_gpu():
            out = F.grid_sample(volume, grid, mode="bicubic", padding_mode=pad, align_corners=True)
        assert out.item() == 1.0


def test_index_wide_expanded_dim():
    # (indices into an expanded dim beyond int32's range, below it)
    if 2 ** 33 > MAX_NUMEL:
        pytest.skip("more elements than this build of PyTorch allows (32-bit)")
    x = torch.arange(3.0, device="vk")[1:2].expand(2 ** 33)
    with on_gpu():
        assert x[torch.tensor([5, 2 ** 31 - 2, -1], device="vk")].tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_autocast(dtype):
    # (CUDA's policy: products in the autocast dtype, exp and sums in float32,
    # mixed inputs promoted; under checkpointing too, as PyTorch's test)
    from torch.utils.checkpoint import checkpoint

    a, b = make((3, 3), torch.float32).to("vk"), make((3, 3), torch.float32, seed=1).to("vk")
    with torch.autocast("vk", dtype=dtype):
        assert torch.mm(a, b).dtype == dtype
        assert F.linear(a, b).dtype == dtype
        assert torch.exp(a.to(dtype)).dtype == torch.float32
        assert torch.addcmul(a, a.to(dtype), b).dtype == torch.float32
        assert torch.add(a, b).dtype == torch.float32
    assert torch.mm(a, b).dtype == torch.float32
    with torch.autocast("vk", dtype=dtype):
        torch.testing.assert_close(torch.mm(a, b).cpu().float(), torch.mm(a.cpu().to(dtype), b.cpu().to(dtype)).float(),
                                   rtol=2e-2, atol=2e-2)

    for enabled in (True, False):
        x, y, z = (make((3, 3), torch.float32, seed=s).to("vk").requires_grad_() for s in range(3))

        def foo(x, y, z):
            z = torch.mm(torch.mm(torch.mm(x, y), z), z)
            assert z.dtype == (dtype if enabled else torch.float32)
            return z

        with torch.autocast("vk", enabled=enabled, dtype=dtype):
            loss = checkpoint(foo, x, y, z, use_reentrant=False).sum()
        loss.backward()
        assert x.grad.dtype == torch.float32


def test_embedding_and_small_model_training():
    import copy

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                                torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(8 * 4 * 4, 10))
    x = make((4, 3, 8, 8), torch.float32)
    y = torch.randint(0, 10, (4,))

    def train(m, xs, ys):
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(m(xs), ys).backward()
            opt.step()
        return [p.detach() for p in m.parameters()]

    vk_model = copy.deepcopy(model).to("vk")
    got = train(vk_model, x.to("vk"), y.to("vk"))
    want = train(copy.deepcopy(model), x, y)
    for a, b in zip(got, want):
        check(a, b, rtol=1e-3, atol=1e-3)
    emb = make((10, 4), torch.float32)
    idx = torch.tensor([[1, 3, 3], [9, 0, 1]])
    compare_with_cpu(lambda w: F.embedding(idx.to(w.device), w), emb)


def test_distances():
    x1, x2 = make((2, 5, 4), torch.float32), make((2, 6, 4), torch.float32, 1)
    for p in [1.0, 2.0, 3.0, float("inf")]:
        compare_with_cpu(lambda a, b: torch.cdist(a, b, p, compute_mode="donot_use_mm_for_euclid_dist"), x1, x2)
    compare_with_cpu(lambda a: F.pdist(a, 2.0), x1[0])
    compare_with_cpu(lambda a: F.pdist(a, 1.5), x1[0])


def test_fractional_max_pool_rrelu_embedding_bag():
    x = make((2, 3, 9, 10), torch.float32)
    samples = torch.rand(2, 3, 2, generator=torch.Generator().manual_seed(5))
    compare_with_cpu(lambda t: F.fractional_max_pool2d(t, 3, output_size=(4, 5), _random_samples=samples.to(t.device)), x)
    x3 = make((1, 2, 6, 7, 8), torch.float32)
    s3 = torch.rand(1, 2, 3, generator=torch.Generator().manual_seed(6))
    compare_with_cpu(lambda t: F.fractional_max_pool3d(t, 2, output_size=(3, 4, 5), _random_samples=s3.to(t.device)), x3)
    compare_with_cpu(lambda t: F.rrelu(t, training=False), x)
    with on_gpu():
        y = F.rrelu(x.to("vk"), 0.1, 0.3, training=True).cpu()
    ratio = (y / x)[x < 0]
    assert ratio.min() >= 0.1 - 1e-6 and ratio.max() <= 0.3 + 1e-6 and torch.equal(y[x >= 0], x[x >= 0])
    w = make((10, 4), torch.float32)
    idx = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
    off = torch.tensor([0, 4, 4, 6])
    for mode in ["sum", "mean", "max"]:
        compare_with_cpu(lambda w: F.embedding_bag(idx.to(w.device), w, off.to(w.device), mode=mode), w)
    psw = torch.rand(8)
    compare_with_cpu(lambda w: F.embedding_bag(idx.to(w.device), w, off.to(w.device), mode="sum",
                                               per_sample_weights=psw.to(w.device)), w)


def test_antialiased_upsample():
    x = make((2, 3, 12, 9), torch.float32)
    for mode in ["bilinear", "bicubic"]:
        compare_with_cpu(lambda t: F.interpolate(t, size=(5, 4), mode=mode, antialias=True), x)


def test_uint8_resize_matches_cpu_exactly():
    # the CPU's fixed-point separable kernel: width pass, height pass, int16 weights
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 3, 13, 17), dtype=torch.uint8)
    for mode, aa, ac, size in itertools.product(("bilinear", "bicubic"), (False, True), (False, True),
                                                [(7, 9), (26, 40), (13, 5), (3, 34)]):
        want = F.interpolate(x, size=size, mode=mode, antialias=aa, align_corners=ac)
        with on_gpu():
            got = F.interpolate(x.to("vk"), size=size, mode=mode, antialias=aa, align_corners=ac)
        assert torch.equal(got.cpu(), want), (mode, aa, ac, size)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rms_norm_and_robust_loss_backward_kernels(dtype):
    # rms_norm forward/backward and smooth_l1/huber backward as single kernels
    torch.manual_seed(0)
    tol = dict(rtol=2e-2, atol=2e-2) if dtype == torch.float16 else dict(rtol=1e-4, atol=1e-5)
    x, w, g = torch.randn(6, 33, dtype=dtype), torch.randn(33, dtype=dtype), torch.randn(6, 33, dtype=dtype)
    outs = []
    for dev in ("cpu", "vk"):
        xx, ww = x.clone().to(dev).requires_grad_(True), w.clone().to(dev).requires_grad_(True)
        ctx = on_gpu() if dev == "vk" else contextlib.nullcontext()
        with ctx:
            y = F.rms_norm(xx, (33,), ww, 1e-5)
            y.backward(g.to(dev))
        outs.append([t.detach().cpu().float() for t in (y, xx.grad, ww.grad)])
    for a, b in zip(*outs):
        torch.testing.assert_close(b, a, **tol)
    t = torch.randn(6, 33)
    for red, param in (("mean", 1.0), ("sum", 0.5), ("none", 0.0)):
        for fn in (lambda a: F.smooth_l1_loss(a, t.to(a.device), reduction=red, beta=param),
                   lambda a: F.huber_loss(a, t.to(a.device), reduction=red, delta=param or 1.0)):
            grads = []
            for dev in ("cpu", "vk"):
                xx = x.float().clone().to(dev).requires_grad_(True)
                ctx = on_gpu() if dev == "vk" else contextlib.nullcontext()
                with ctx:
                    out = fn(xx)
                    out.backward(torch.ones_like(out))
                grads.append(xx.grad.cpu())
            torch.testing.assert_close(grads[1], grads[0])


DIRECT_CONV_SCRIPT = r"""
import os, torch, torch.nn.functional as F, torch_vulkan as tv
if os.environ["TORCH_VK_CONV"] == "direct" and tv._native.kernel_subgroup_size("conv4s_f32_d1_r8n1") != 16:
    print("skip")
    raise SystemExit
torch.manual_seed(0)
worst = 0.0
convs = [F.conv1d, F.conv2d, F.conv3d]
def check(got, want, mag, dtype):
    # float32 sums of the same products in another order (and fused), plus
    # one rounding to dtype
    global worst
    eps = 0.0 if dtype == torch.float32 else torch.finfo(dtype).eps
    err = (got.cpu().double() - want).abs()
    tol = eps * want.abs() + 2**-20 * mag + 2**-30
    worst = max(worst, float((err / tol).max()))
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    cases = [
        # n, cin, cout, kernel, padding, dilation, groups, spatial
        (1, 16, 24, 11, 5, 1, 1, (301,)),
        (1, 8, 20, 7, 9, 3, 1, (150,)),
        (2, 6, 9, 3, 1, 5, 3, (67,)),
        (1, 5, 3, 16, 0, 1, 1, (200,)),
        (1, 4, 17, 1, 0, 1, 1, (33,)),
        (2, 4, 6, 2, (0,), 2, 2, (129,)),
        (1, 3, 8, 3, 1, 1, 1, (20, 23)),
        (2, 4, 6, (3, 5), (2, 1), (1, 2), 2, (11, 19)),
        (1, 2, 4, 3, 1, 1, 1, (5, 6, 13)),
    ]
    for n, cin, cout, k, pad, dil, groups, spatial in cases:
        dims = len(spatial)
        ks = k if isinstance(k, tuple) else (k,) * dims
        x = torch.randn((n, cin) + spatial).to(dtype)
        w = torch.randn((cout, cin // groups) + ks).to(dtype)
        b = torch.randn(cout).to(dtype)
        conv = convs[dims - 1]
        for bias in (b, None):
            want = conv(x.double(), w.double(), None if bias is None else bias.double(), 1, pad, dil, groups)
            mag = conv(x.double().abs(), w.double().abs(), None if bias is None else bias.double().abs(), 1, pad, dil, groups)
            got = conv(x.to("vk"), w.to("vk"), None if bias is None else bias.to("vk"), 1, pad, dil, groups)
            check(got, want, mag, dtype)
    # transposed convolutions by phases: stride-1 convolutions with cropped inputs
    x, w = torch.randn(1, 8, 40).to(dtype), torch.randn(8, 4, 20).to(dtype)
    want = F.conv_transpose1d(x.double(), w.double(), None, 10, 5)
    mag = F.conv_transpose1d(x.double().abs(), w.double().abs(), None, 10, 5)
    check(F.conv_transpose1d(x.to("vk"), w.to("vk"), None, 10, 5), want, mag, dtype)
assert not tv.fallback_counts(), tv.fallback_counts()
print(worst)
"""


@pytest.mark.parametrize("route", ["direct", "im2col"])
def test_direct_conv(route):
    # shaders/conv4s.slang (TORCH_VK_CONV=direct: wherever the GPU runs
    # 16-lane subgroups, by default on Intel and for some 16-bit cases on
    # Mali): stride-1 convolutions reading the padded input in place, float32
    # sums rounded once; and the same through im2col columns, which the
    # direct kernel otherwise takes on those GPUs
    out = subprocess.run([sys.executable, "-c", DIRECT_CONV_SCRIPT], env=dict(os.environ, TORCH_VK_CONV=route),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    last = out.stdout.strip().splitlines()[-1]
    if last == "skip":
        pytest.skip("no 16-lane subgroups with shuffles")
    assert float(last) <= 1.0
