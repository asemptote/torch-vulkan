"""Neural-network ops on "vk" vs the CPU (M12), forward and backward."""

import contextlib
import itertools
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from _util import check, make, on_gpu

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
    volume = torch.ones(1, 1, 1, 1, 1, device="vk").expand(1, 1, 1, 1, size)
    grid = torch.tensor([[[[[1.0, 0.0, 0.0]]]]], device="vk")
    for pad in ("zeros", "border", "reflection"):
        with on_gpu():
            out = F.grid_sample(volume, grid, mode="bicubic", padding_mode=pad, align_corners=True)
        assert out.item() == 1.0


def test_index_wide_expanded_dim():
    # (indices into an expanded dim beyond int32's range, below it)
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
import torch, torch.nn.functional as F, torch_vulkan as tv
if tv._native.kernel_subgroup_size("conv4s_f32_r8n1") != 16:
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


def test_direct_conv():
    # shaders/conv4s.slang (the default on Mali; elsewhere through
    # TORCH_VK_CONV=direct where the GPU runs 16-lane subgroups): stride-1
    # convolutions reading the padded input in place, float32 sums rounded once
    out = subprocess.run([sys.executable, "-c", DIRECT_CONV_SCRIPT], env=dict(os.environ, TORCH_VK_CONV="direct"),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    last = out.stdout.strip().splitlines()[-1]
    if last == "skip":
        pytest.skip("no 16-lane subgroups with shuffles")
    assert float(last) <= 1.0
