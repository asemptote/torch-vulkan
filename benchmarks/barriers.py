"""Cost of the barriers between commands on "vk" (issue #45, docs/log/M37).

    python benchmarks/barriers.py [--reps 200] [--filter substr]
    TORCH_VK_BARRIERS=full python benchmarks/barriers.py   # every command barriered, for comparison

Each case is timed as a whole, including its synchronize, and reported as the
median over --reps runs (microseconds), with the barriers the stream put in
per run where the build counts them:

- 10 / 1000 independent: `torch.add(a[i], 1, out=o[i])` on distinct tensors
  of 256 float32 values, then a synchronize. Nothing orders these kernels,
  so the stream needs no barrier between them.
- 10 / 1000 dependent: `x.add_(1)` on one tensor, then a synchronize: each
  kernel must wait for the one before.
- adam: one `torch.optim.Adam(foreach=True)` step over the parameters of a
  small MLP (8 layers of 256 x 256 plus biases: 16 tensors).
- mlp fwd / fwd+bwd: that MLP on a batch of 32 (8 linear layers with GELU
  and layer norm), forward only and forward plus backward.
"""
import argparse
import statistics
import time

import torch
import torch_vulkan
from torch_vulkan import _native

DEV = "vk"


def barrier_count():
    stats = getattr(_native, "barrier_stats", None)
    return dict(stats())["barriers"] if stats else None


def bench(fn, reps):
    for _ in range(5):
        fn()
    torch_vulkan.synchronize()
    times = []
    b0 = barrier_count()
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch_vulkan.synchronize()
        times.append(time.perf_counter() - t)
    b1 = barrier_count()
    per_run = None if b0 is None else (b1 - b0) / reps
    return statistics.median(times) * 1e6, per_run


def cases():
    for n in (10, 1000):
        a = [torch.randn(256, device=DEV) for _ in range(n)]
        o = [torch.empty(256, device=DEV) for _ in range(n)]

        def independent(a=a, o=o):
            for x, y in zip(a, o):
                torch.add(x, 1.0, out=y)

        x = torch.zeros(256, device=DEV)

        def dependent(x=x, n=n):
            for _ in range(n):
                x.add_(1.0)

        yield f"{n} independent adds", independent
        yield f"{n} dependent adds", dependent

    torch.manual_seed(0)
    layers = []
    for _ in range(8):
        layers += [torch.nn.Linear(256, 256), torch.nn.LayerNorm(256), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-3
    yield f"adam step ({len(list(model.parameters()))} tensors)", opt.step

    xb = torch.randn(32, 256, device=DEV)

    def fwd():
        with torch.no_grad():
            model(xb)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        model(xb).square().mean().backward()

    yield "mlp forward", fwd
    yield "mlp forward+backward", fwd_bwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    print(f"# {torch_vulkan.device_name()}")
    print(f"{'case':34} {'us':>10} {'barriers':>9}")
    for name, fn in cases():
        if args.filter not in name:
            continue
        us, barriers = bench(fn, args.reps if "1000" not in name else max(args.reps // 10, 5))
        b = "-" if barriers is None else f"{barriers:.1f}"
        print(f"{name:34} {us:10.1f} {b:>9}", flush=True)


if __name__ == "__main__":
    main()
