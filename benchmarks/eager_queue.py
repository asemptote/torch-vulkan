"""How far the host runs ahead of the GPU in a long eager loop (issue #100).

    python benchmarks/eager_queue.py [--seconds 10] [--case small|matmul|train]

Each case runs in a fresh process (so that its peak memory is its own) and
queues work without synchronising, then synchronises once at the end:

- small: element-wise ops on 2^16 float32 values (`y.mul_(a).add_(b)`), many
  thousands of tiny kernels;
- matmul: float32 matrix products into a preallocated output, sized so that
  one takes about 2 ms of GPU time;
- train: an MLP (4 x 1024 linear + GELU, batch 256) trained with Adam,
  reading the loss back only once, at the end.

Each runs for about --seconds of GPU time. Reported per case:

- per op (or step): wall time over the whole run, the final synchronise
  included (throughput);
- enqueue: the host's time in the loop, before the final synchronise;
- sync: the final synchronise alone: how much work was still queued;
- mid-sync: one synchronise in the middle of the loop (latency of a host
  read while the loop runs);
- peak RSS: the process's peak resident memory less its peak before the
  loop (host and driver memory held by queued work; the device memory pool
  is stream-ordered and does not grow with the queue);
- the stream's queue statistics where the build has them
  (`_native.queue_stats()`).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

import torch
import torch_vulkan
from torch_vulkan import _native

DEV = "vk"


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20


def queue_stats():
    f = getattr(_native, "queue_stats", None)
    return dict(f()) if f else None


def timed_loop(step, n):
    """Runs step(i) for i < n with a synchronise at n/2; returns (enqueue,
    final sync, mid sync) in seconds."""
    mid = 0.0
    t0 = time.perf_counter()
    for i in range(n):
        step(i)
        if i == n // 2:
            t = time.perf_counter()
            torch_vulkan.synchronize()
            mid = time.perf_counter() - t
    t1 = time.perf_counter()
    torch_vulkan.synchronize()
    t2 = time.perf_counter()
    return t1 - t0 - mid, t2 - t1, mid


def gpu_time(step, reps):
    """Seconds per call of step, synchronising after each batch of reps."""
    step(0)
    torch_vulkan.synchronize()
    t = time.perf_counter()
    for i in range(reps):
        step(i)
    torch_vulkan.synchronize()
    return (time.perf_counter() - t) / reps


def case_small(seconds):
    y = torch.ones(1 << 16, device=DEV)

    def step(i):
        y.mul_(1.0001).add_(1e-4)

    return step, 2, max(1, int(seconds / gpu_time(step, 2000))), "op"


def case_matmul(seconds):
    # (sized so that one product is about 2 ms)
    n = 256
    while True:
        a = torch.randn(n, n, device=DEV)
        b = torch.randn(n, n, device=DEV)
        out = torch.empty(n, n, device=DEV)

        def step(i, a=a, b=b, out=out):
            torch.mm(a, b, out=out)

        t = gpu_time(step, 10)
        if t > 2e-3 or n >= 8192:
            break
        n *= 2
    return step, 1, max(1, int(seconds / t)), f"op ({n}^2 matmul)"


def case_train(seconds):
    torch.manual_seed(0)
    layers = []
    for _ in range(4):
        layers += [torch.nn.Linear(1024, 1024), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers, torch.nn.Linear(1024, 10)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.randn(256, 1024, device=DEV)
    target = torch.randint(0, 10, (256,), device=DEV)
    last = {}

    def step(i):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), target)
        loss.backward()
        opt.step()
        last["loss"] = loss

    return step, 1, max(1, int(seconds / gpu_time(step, 5))), "step"


CASES = {"small": case_small, "matmul": case_matmul, "train": case_train}


def run_case(name, seconds):
    step, ops_per_step, n, unit = CASES[name](seconds)
    torch_vulkan.synchronize()
    rss0 = peak_rss_mb()
    q0 = queue_stats()
    enqueue, sync, mid = timed_loop(step, n)
    total = enqueue + sync + mid
    q1 = queue_stats()
    out = {
        "case": name,
        "device": _native.device_name(),
        "unit": unit,
        "n": n * ops_per_step,
        "per_op_us": total / (n * ops_per_step) * 1e6,
        "enqueue_s": enqueue,
        "sync_s": sync,
        "mid_sync_s": mid,
        "total_s": total,
        "peak_rss_delta_mb": peak_rss_mb() - rss0,
    }
    if q1 is not None:
        counters = ("submissions", "waits", "wait_s", "timed")
        keep = counters + ("max_queued_s", "max_in_flight")
        out["queue"] = {k: q1[k] - q0[k] if k in counters else q1[k] for k in keep}
    print(json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=10.0, help="GPU time per case (approximate)")
    p.add_argument("--case", choices=list(CASES), action="append")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    cases = args.case or list(CASES)
    if args.child:
        run_case(cases[0], args.seconds)
        return
    env_keys = [k for k in os.environ if k.startswith("TORCH_VK_")]
    print(f"# {_native.device_name()}; " + " ".join(f"{k}={os.environ[k]}" for k in sorted(env_keys)))
    print(f"{'case':<8} {'per op':>10} {'enqueue':>8} {'sync':>7} {'mid-sync':>8} {'peak RSS':>9}  queue")
    for name in cases:
        proc = subprocess.run(
            [sys.executable, __file__, "--child", "--case", name, "--seconds", str(args.seconds)],
            capture_output=True, text=True,
        )
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode or not lines:
            print(f"{name}: failed\n{proc.stdout}{proc.stderr}")
            continue
        r = json.loads(lines[-1])
        q = r.get("queue")
        qs = "" if q is None else " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in q.items())
        print(
            f"{name:<8} {r['per_op_us']:>8.1f}us {r['enqueue_s']:>7.2f}s {r['sync_s']:>6.2f}s {r['mid_sync_s']:>7.2f}s"
            f" {r['peak_rss_delta_mb']:>7.1f}MB  {r['unit']}, n={r['n']} {qs}"
        )


if __name__ == "__main__":
    main()
