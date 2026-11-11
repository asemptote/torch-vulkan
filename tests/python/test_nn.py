"""Neural-network ops on "vk" vs the CPU (M12), forward and backward."""

import contextlib
import itertools

import pytest
import torch
import torch.nn.functional as F

from _util import ARM32, check, make, on_gpu

# (On the Mali phone, Termux's armv7 torch computed CPU convolutions wrong in
# about 1 of 50 float32 and 5 of 50 float64 calls, against a pure-Python
# convolution, while vk's matched that reference throughout: the CPU is no
# reference for these tests there.)
UNRELIABLE_CPU_CONV = pytest.mark.skipif(ARM32, reason="a 32-bit Arm CPU's convolutions are intermittently wrong")

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
]


@pytest.mark.parametrize("dims,cin,cout,k,stride,pad,dil,groups", CONV_CASES)
@UNRELIABLE_CPU_CONV
def test_conv(dims, cin, cout, k, stride, pad, dil, groups):
    spatial = (11, 9, 7)[:dims]
    x = make((2, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cout, cin // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, dil, groups), x, w, b)


@pytest.mark.parametrize("dims,cin,cout,k,stride,pad,dil,groups,outpad", [
    (1, 3, 5, 3, 2, 1, 1, 1, 1),
    (2, 4, 6, 3, 2, 1, 1, 2, (1, 0)),
    (2, 3, 4, (2, 4), (3, 1), (0, 2), (2, 1), 1, (2, 0)),
    (3, 2, 2, 3, 1, 0, 1, 2, 0),
])
@UNRELIABLE_CPU_CONV
def test_conv_transpose(dims, cin, cout, k, stride, pad, dil, groups, outpad):
    spatial = (6, 5, 4)[:dims]
    x = make((2, cin) + spatial, torch.float32)
    ks = k if isinstance(k, tuple) else (k,) * dims
    w = make((cin, cout // groups) + ks, torch.float32, 1)
    b = make((cout,), torch.float32, 2)
    conv = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][dims - 1]
    compare_with_cpu(lambda x, w, b: conv(x, w, b, stride, pad, outpad, groups, dil), x, w, b)


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


@UNRELIABLE_CPU_CONV
def test_embedding_and_small_model_training():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                                torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Linear(8 * 4 * 4, 10))
    x = make((4, 3, 8, 8), torch.float32)
    y = torch.randint(0, 10, (4,))
    vk_model = torch.nn.Sequential(*[m for m in model])
    import copy
    vk_model = copy.deepcopy(model).to("vk")
    for m, xs, ys in [(model, x, y), (vk_model, x.to("vk"), y.to("vk"))]:
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
        for _ in range(3):
            opt.zero_grad()
            F.cross_entropy(m(xs), ys).backward()
            opt.step()
    for a, b in zip(vk_model.parameters(), model.parameters()):
        check(a.detach(), b.detach(), rtol=1e-3, atol=1e-3)
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
