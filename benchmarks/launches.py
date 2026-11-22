"""Kernel launches per call of composite ops on "vk", and the aten ops they come from.

    python benchmarks/launches.py                 # every case: launches and ms per call
    python benchmarks/launches.py --filter grid --ops 8

For each case (forward, backward, double backward where listed): the kernels
one call launches (the runtime's dispatch counter, after a warm-up call) and
the median time per call, and with --ops the aten ops that launched the
most (each op's launches include those of the ops a Python or C++ kernel
runs inside it). Ops built from many small launches are the candidates for
a dedicated kernel, or for graph capture (#78) where the launches are
inherent (iterative algorithms). TORCH_VK_TRACE=1 prints each launch's
kernel name instead (slow: it waits for each one).
"""
import argparse
import collections
import statistics
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import torch_vulkan  # noqa: F401
from torch_vulkan import _native


class PerOp(TorchDispatchMode):
    """Launches per aten op (outermost ops only: the mode is off inside an op)."""

    def __init__(self):
        super().__init__()
        self.launches = collections.Counter()
        self.calls = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        before = _native.dispatch_count()
        out = func(*args, **(kwargs or {}))
        name = str(func.overloadpacket.__name__ if hasattr(func, "overloadpacket") else func)
        self.launches[name] += _native.dispatch_count() - before
        self.calls[name] += 1
        return out


def _grad(fn, *inputs, create_graph=False):
    def run():
        ins = [t.detach().requires_grad_(t.is_floating_point()) for t in inputs]
        out = fn(*ins)
        wrt = [t for t in ins if t.requires_grad]
        return torch.autograd.grad(out, wrt, torch.ones_like(out), create_graph=create_graph), wrt
    return run


def _gradgrad(fn, *inputs):
    def run():
        grads, wrt = _grad(fn, *inputs, create_graph=True)()
        return torch.autograd.grad(sum(g.sum() for g in grads), wrt, allow_unused=True)
    return run


def cases():
    r = torch.randn
    img = lambda: r(8, 64, 56, 56)  # noqa: E731
    grid2 = lambda: r(8, 28, 28, 2) * 0.9  # noqa: E731
    vol = lambda: r(2, 8, 12, 12, 12)  # noqa: E731
    grid3 = lambda: r(2, 6, 6, 6, 3) * 0.9  # noqa: E731
    yield "flip 256x128x4x4 (2, 3)", (lambda: r(256, 128, 4, 4),), lambda w: w.flip(2, 3), False
    yield "flip [1, 256, 128, 2, 2, 2, 2] (3, 5)", (lambda: r(1, 256, 128, 2, 2, 2, 2),), lambda w: w.flip(3, 5), False
    yield "conv2d depthwise k3", (img, lambda: r(64, 1, 3, 3)), lambda x, w: F.conv2d(x, w, padding=1, groups=64), True
    yield "conv2d depthwise k3 s2", (img, lambda: r(64, 1, 3, 3)), \
        lambda x, w: F.conv2d(x, w, stride=2, padding=1, groups=64), True
    yield "conv2d depthwise k7", (img, lambda: r(64, 1, 7, 7)), lambda x, w: F.conv2d(x, w, padding=3, groups=64), True
    yield "conv_transpose1d depthwise k3 s2", (lambda: r(1, 640, 121), lambda: r(640, 1, 3)), \
        lambda x, w: F.conv_transpose1d(x, w, stride=2, padding=1, groups=640), True
    yield "conv_transpose2d k4 s2", (lambda: r(1, 64, 32, 32), lambda: r(64, 32, 4, 4)), \
        lambda x, w: F.conv_transpose2d(x, w, stride=2, padding=1), True
    yield "conv2d k3 s2", (img, lambda: r(64, 64, 3, 3)), lambda x, w: F.conv2d(x, w, stride=2, padding=1), True
    yield "conv1d k3 (input grad: flip)", (lambda: r(4, 64, 1000), lambda: r(64, 64, 3)), \
        lambda x, w: F.conv1d(x, w, padding=1), True
    for mode in ("bilinear", "nearest", "bicubic"):
        yield f"grid_sample 2d {mode}", (img, grid2), \
            lambda x, g, m=mode: F.grid_sample(x, g, mode=m, align_corners=False), True
    for mode in ("bilinear", "nearest"):
        yield f"grid_sample 3d {mode}", (vol, grid3), \
            lambda x, g, m=mode: F.grid_sample(x, g, mode=m, align_corners=False), True
    for xs, gs in (((1, 1, 2, 2, 2), (1, 2, 2, 2, 3)), ((10, 11, 12, 13, 14), (10, 23, 22, 21, 3))):
        yield f"grid_sampler_3d gradgrad float64 {list(xs)}", \
            (lambda xs=xs: r(*xs, dtype=torch.float64), lambda gs=gs: r(*gs, dtype=torch.float64) * 2), \
            lambda x, g: torch.grid_sampler_3d(x, g, 0, 0, False), "gradgrad"
    yield "interpolate bicubic x2", (img,), lambda x: F.interpolate(x, scale_factor=2, mode="bicubic"), True
    yield "interpolate bilinear aa /2", (img,), \
        lambda x: F.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True), True
    yield "pad reflect 2d", (img,), lambda x: F.pad(x, (2, 2, 2, 2), mode="reflect"), True
    yield "pad replicate 3d", (vol,), lambda x: F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), True
    yield "adaptive_max_pool2d 7", (img,), lambda x: F.adaptive_max_pool2d(x, 7), True
    yield "adaptive_avg_pool2d 7", (img,), lambda x: F.adaptive_avg_pool2d(x, 7), True
    yield "fractional_max_pool2d", (img,), lambda x: F.fractional_max_pool2d(x, 2, output_size=20, _random_samples=torch.rand(8, 64, 2, device=x.device)), True
    yield "roll (1, 2) dims (2, 3)", (img,), lambda x: x.roll((1, 2), (2, 3)), False
    yield "unfold k3", (img,), lambda x: F.unfold(x, 3, padding=1), True
    yield "fold k3", (lambda: r(8, 64 * 9, 56 * 56),), lambda c: F.fold(c, (56, 56), 3, padding=1), True
    for t in (50, 200):
        yield f"ctc_loss T{t} S10", (lambda t=t: r(t, 4, 20).log_softmax(2),), \
            lambda lp, t=t: F.ctc_loss(lp, torch.randint(1, 20, (4, 10), device=lp.device), [t] * 4, [10] * 4), True
    yield "embedding_bag sum", (lambda: r(1000, 64),), \
        lambda w: F.embedding_bag(torch.randint(0, 1000, (256,), device=w.device), w,
                                  torch.arange(0, 256, 8, device=w.device), mode="sum"), True
    yield "cdist p=2", (lambda: r(4, 128, 32), lambda: r(4, 96, 32)), lambda a, b: torch.cdist(a, b), True
    yield "pdist", (lambda: r(128, 32),), lambda a: F.pdist(a), True
    yield "lstm cell x16 steps", (lambda: r(16, 4, 64),), None, True
    yield "adam step, 50 tensors (foreach)", (lambda: r(1),), adam_step, False
    yield "conv2d grouped dw-multiplier 2", (img, lambda: r(128, 1, 3, 3)), \
        lambda x, w: F.conv2d(x, w, padding=1, groups=64), True


_ADAM = {}


def adam_step(x):
    """One Adam step over 50 parameters (foreach, the default on vk)."""
    if "opt" not in _ADAM:
        params = [torch.randn(64, 64, device=x.device, requires_grad=True) for _ in range(50)]
        for q in params:
            q.grad = torch.randn_like(q)
        _ADAM["opt"] = torch.optim.Adam(params, lr=1e-3)
    _ADAM["opt"].step()
    return x


def lstm_steps(x):
    cell = _LSTM[x.device.type]
    h = torch.zeros(4, 64, device=x.device)
    c = torch.zeros(4, 64, device=x.device)
    for t in range(x.shape[0]):
        h, c = cell(x[t], (h, c))
    return h


_LSTM = {}


def count(run):
    _native.synchronize()
    before = _native.dispatch_count()
    run()
    _native.synchronize()
    return _native.dispatch_count() - before


def median_ms(run, repeat):
    times = []
    for _ in range(repeat):
        _native.synchronize()
        t = time.perf_counter()
        run()
        _native.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--ops", type=int, default=0, help="show the N aten ops with the most launches")
    args = ap.parse_args()
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(64, 64)
    _LSTM["vk"] = torch.nn.LSTMCell(64, 64).to("vk")
    _LSTM["vk"].load_state_dict(cell.state_dict())
    print(f"device: {_native.device_name()}")
    print(f"{'case':52s} {'launches':>9s} {'ms':>9s}")
    for name, makers, fn, grad in cases():
        if args.filter not in name:
            continue
        if fn is None:
            fn = lstm_steps
        inputs = [m().to("vk") for m in makers]
        runs = [("", lambda: fn(*inputs))]
        if grad:
            runs.append((" bwd", _grad(fn, *inputs)))
        if grad == "gradgrad":
            runs = [(" gradgrad", _gradgrad(fn, *inputs))]
        for suffix, run in runs:
            try:
                run()  # warm-up: pipelines compiled, pool filled
                n = count(run)
                ms = median_ms(run, args.repeat)
            except Exception as e:  # noqa: BLE001
                print(f"{name + suffix:52s} error: {str(e)[:70]}")
                continue
            print(f"{name + suffix:52s} {n:9d} {ms:9.2f}")
            if args.ops:
                mode = PerOp()
                with mode:
                    run()
                top = mode.launches.most_common(args.ops)
                print("    " + ", ".join(f"{op} {n}/{mode.calls[op]}" for op, n in top if n))


if __name__ == "__main__":
    main()
