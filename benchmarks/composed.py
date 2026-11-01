"""Composed neural-network ops on "vk": time, kernels launched, CPU fallbacks.

    python benchmarks/composed.py [--size small|large] [--repeat 20] [--filter substr]

For each op (forward, and backward where it has one): the median time per call
on "vk" and on the CPU, the Vulkan kernels one call launches, and the aten ops
that ran on the CPU instead. Ops implemented as chains of generic kernels
(softmax, normalization, losses, ...) show up as many launches per call: the
candidates for dedicated kernels (M16), ranked by vk time.
"""
import argparse
import statistics
import time

import torch
import torch.nn.functional as F
import torch_vulkan
from torch_vulkan import _native


def cases(large):
    n, c, hw, d, seq = (64, 256, 56, 1024, 512) if large else (8, 64, 28, 256, 128)
    x4 = lambda: torch.randn(n, c, hw, hw)  # noqa: E731
    x2 = lambda: torch.randn(n * seq // 8, d)  # noqa: E731
    w = torch.randn(c, c, 3, 3) * 0.05
    yield "softmax", x2, lambda x: F.softmax(x, -1)
    yield "log_softmax", x2, lambda x: F.log_softmax(x, -1)
    yield "layer_norm", x2, lambda x: F.layer_norm(x, (x.shape[-1],))
    yield "rms_norm", x2, lambda x: F.rms_norm(x, (x.shape[-1],))
    yield "group_norm", x4, lambda x: F.group_norm(x, 8)
    yield "batch_norm (train)", x4, lambda x: F.batch_norm(x, None, None, training=True)
    yield "gelu", x2, F.gelu
    yield "silu", x2, F.silu
    yield "cross_entropy", lambda: torch.randn(n * 16, 1000), lambda x: F.cross_entropy(x, torch.arange(x.shape[0], device=x.device) % 1000)
    yield "mse_loss", x2, lambda x: F.mse_loss(x, torch.zeros_like(x))
    yield "max_pool2d", x4, lambda x: F.max_pool2d(x, 3, 2, 1)
    yield "avg_pool2d", x4, lambda x: F.avg_pool2d(x, 2)
    yield "adaptive_avg_pool2d", x4, lambda x: F.adaptive_avg_pool2d(x, 1)
    yield "interpolate bilinear", x4, lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")
    yield "conv2d 3x3", x4, lambda x: F.conv2d(x, w.to(x.device), padding=1)
    yield "linear", x2, lambda x: F.linear(x, torch.eye(x.shape[-1], device=x.device))
    yield "embedding", lambda: torch.randn(30000, d), lambda wt: F.embedding(torch.arange(n * seq, device=wt.device) % 30000, wt)
    yield "attention (sdpa)", lambda: torch.randn(n, 8, seq, 64), lambda q: F.scaled_dot_product_attention(q, q, q)


def median_time(fn, device, repeat):
    times = []
    for _ in range(repeat):
        if device == "vk":
            _native.synchronize()
        t = time.perf_counter()
        fn()
        if device == "vk":
            _native.synchronize()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def launches(fn):
    if not hasattr(_native, "dispatch_count"):
        return None
    _native.synchronize()
    before = _native.dispatch_count()
    fn()
    _native.synchronize()
    return _native.dispatch_count() - before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("small", "large"), default="small")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    print(f"device: {_native.device_name()}, size {args.size}")
    rows = []
    for name, make, op in cases(args.size == "large"):
        if args.filter not in name:
            continue
        for backward in (False, True):
            x_cpu = make()
            x_vk = x_cpu.to("vk")
            if backward:
                x_cpu.requires_grad_(True)
                x_vk.requires_grad_(True)
                run = lambda x: op(x).sum().backward()  # noqa: E731
            else:
                run = lambda x: op(x)  # noqa: E731
            try:
                run(x_vk)  # warm-up: pipelines compiled, pool filled
                before = dict(torch_vulkan.fallback_counts())
                n_launch = launches(lambda: run(x_vk))
                after = torch_vulkan.fallback_counts()
                fallbacks = sorted(k for k in after if after[k] != before.get(k, 0))
                t_vk = median_time(lambda: run(x_vk), "vk", args.repeat)
                t_cpu = median_time(lambda: run(x_cpu), "cpu", max(3, args.repeat // 4))
            except Exception as e:  # noqa: BLE001
                rows.append((name + (" bwd" if backward else ""), None, None, None, [f"error: {e}"[:80]]))
                continue
            rows.append((name + (" bwd" if backward else ""), t_vk, t_cpu, n_launch, fallbacks))
    rows.sort(key=lambda r: -(r[1] or 0))
    print(f"{'op':28s} {'vk ms':>9s} {'cpu ms':>9s} {'launches':>9s}  fallbacks")
    for name, t_vk, t_cpu, n, fb in rows:
        vk = f"{t_vk * 1e3:9.2f}" if t_vk is not None else f"{'-':>9s}"
        cpu = f"{t_cpu * 1e3:9.2f}" if t_cpu is not None else f"{'-':>9s}"
        nl = f"{n:9d}" if n is not None else f"{'-':>9s}"
        print(f"{name:28s} {vk} {cpu} {nl}  {', '.join(fb)}")


if __name__ == "__main__":
    main()
