"""The same workloads on "vk" and on CUDA (same PyTorch release), as a table.

    python benchmarks/vs_cuda.py --vk-python VENV_VK/bin/python \\
        --cuda-python VENV_CUDA/bin/python --out benchmarks/results/NAME.md

Each (device, workload) runs in its own process (a fresh driver state, and
a hang or a crash costs one cell): one untimed call, calls until 0.3 s have
passed (clocks ramp under load), then up to 7 timed calls, each followed by
a synchronize, within a few seconds, and the median. Workloads that take
longer than --timeout seconds are reported as such. Either Python may be
left out (that column is then empty).

Workloads: matrix products (float16, bfloat16, float32 with CUDA's TF32 off,
and once with it on for reference, float64) at 2048^3 and 8192^3; FFTs and
real FFTs of 2^20 and 2^24 points in complex64 / complex128; float64
linalg.solve, eigh and svd at n = 1024 and 4096; an element-wise chain over
2^26 float32 values (GB/s); the per-op overhead of small additions.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

WORKLOADS = []  # (name, unit, setup source): setup defines fn() and `amount` (per call, in `unit`'s terms)


def add(name, unit, setup):
    WORKLOADS.append((name, unit, setup))


for dt in ("float16", "bfloat16", "float32", "float64"):
    for n in (2048, 8192):
        add(f"matmul {dt} {n}^3", "TFLOP/s",
            f"a = torch.randn({n}, {n}, device=dev, dtype=torch.{dt}); b = torch.randn({n}, {n}, device=dev, "
            f"dtype=torch.{dt})\nfn = lambda: a @ b\namount = 2 * {n}**3 / 1e12")
for n in (2048, 8192):
    add(f"matmul float32 {n}^3, TF32 on (CUDA)", "TFLOP/s",
        f"torch.backends.cuda.matmul.allow_tf32 = True\na = torch.randn({n}, {n}, device=dev); "
        f"b = torch.randn({n}, {n}, device=dev)\nfn = lambda: a @ b\namount = 2 * {n}**3 / 1e12")
for bits in (20, 24):
    for dt in ("complex64", "complex128"):
        add(f"fft 2^{bits} {dt}", "ms",
            f"x = torch.randn(1 << {bits}, device=dev, dtype=torch.{dt})\nfn = lambda: torch.fft.fft(x)\namount = None")
        real = "float32" if dt == "complex64" else "float64"
        add(f"rfft 2^{bits} {real}", "ms",
            f"x = torch.randn(1 << {bits}, device=dev, dtype=torch.{real})\nfn = lambda: torch.fft.rfft(x)\namount = None")
for n in (1024, 4096):
    add(f"linalg.solve float64 n={n}", "ms",
        f"a = torch.randn({n}, {n}, device=dev, dtype=torch.float64) + {n} * torch.eye({n}, device=dev, "
        f"dtype=torch.float64); b = torch.randn({n}, 64, device=dev, dtype=torch.float64)\n"
        f"fn = lambda: torch.linalg.solve(a, b)\namount = None")
    add(f"eigh float64 n={n}", "ms",
        f"a = torch.randn({n}, {n}, device=dev, dtype=torch.float64); a = a + a.mT\n"
        f"fn = lambda: torch.linalg.eigh(a)\namount = None")
    add(f"svd float64 n={n}", "ms",
        f"a = torch.randn({n}, {n}, device=dev, dtype=torch.float64)\n"
        f"fn = lambda: torch.linalg.svd(a)\namount = None")
add("elementwise chain 2^26 float32 (x * y + z).relu()", "GB/s",
    "x, y, z = (torch.randn(1 << 26, device=dev) for _ in range(3))\n"
    "fn = lambda: (x * y + z).relu()\n"
    "amount = (3 + 3 + 2) * 4 * (1 << 26) / 1e9  # bytes read and written by mul, add, relu")
add("per-op overhead: 1000 additions of 1024 floats", "us/op",
    "x = torch.randn(1024, device=dev)\n"
    "def fn():\n    y = x\n    for _ in range(1000):\n        y = y + 1.0\n    return y\n"
    "amount = 'per-op'")

WORKER = r'''
import json, statistics, sys, time
import torch
dev = sys.argv[1]
if dev == "vk":
    import torch_vulkan as tv
    sync = tv.synchronize
    name = tv.device_name()
else:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sync = torch.cuda.synchronize
    name = torch.cuda.get_device_name()
exec(sys.stdin.read())
fn(); sync()
start = time.perf_counter()
while time.perf_counter() - start < 0.3:
    fn(); sync()
times = []
budget = time.perf_counter()
while len(times) < 7 and (len(times) < 3 or time.perf_counter() - budget < 5.0):
    t = time.perf_counter(); fn(); sync(); times.append(time.perf_counter() - t)
    if times[-1] > 20 and len(times) >= 1:
        break
print(json.dumps({"device": name, "seconds": statistics.median(times), "amount": amount, "torch": torch.__version__}))
'''


def run(python, dev, setup, timeout):
    if not python:
        return None
    try:
        r = subprocess.run([python, "-c", WORKER, dev], input=setup, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"over {timeout} s"}
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not lines:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return {"error": tail[:120]}
    return json.loads(lines[-1])


def value(res, unit):
    """The cell's number in `unit` (None if the run failed)."""
    if not res or "error" in res:
        return None
    s, amount = res["seconds"], res["amount"]
    if unit == "ms":
        return s * 1e3
    if unit == "us/op":
        return s * 1e6 / 1000
    return amount / s  # TFLOP/s, GB/s


def cell(res, unit):
    v = value(res, unit)
    if res is None:
        return ""
    if v is None:
        return res["error"]
    return f"{v:.3g}" if v < 100 else f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vk-python", default="")
    ap.add_argument("--cuda-python", default="")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--filter", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rows, devices, versions = [], {}, set()
    for name, unit, setup in WORKLOADS:
        if args.filter and args.filter not in name:
            continue
        vk = run(args.vk_python, "vk", setup, args.timeout) if "(CUDA)" not in name else None
        cu = run(args.cuda_python, "cuda", setup, args.timeout)
        for d, r in (("vk", vk), ("CUDA", cu)):
            if r and "device" in r:
                devices[d] = r["device"]
                versions.add(r["torch"])
        a, b = value(vk, unit), value(cu, unit)
        # (ratio: how many times slower vk is; for rates the inverse of the rate ratio)
        ratio = ""
        if a and b:
            ratio = f"{(a / b if unit in ('ms', 'us/op') else b / a):.2g}"
        rows.append(f"| {name} | {unit} | {cell(vk, unit)} | {cell(cu, unit)} | {ratio} |")
        print(rows[-1], flush=True)
    head = [f"vk: {devices.get('vk', '-')}; CUDA: {devices.get('CUDA', '-')}; torch {', '.join(sorted(versions))}",
            "", "Medians of synchronised, warm calls; the last column is vk's time over CUDA's (how many times "
            "slower vk is; under 1: faster).", "",
            "| workload | unit | vk | CUDA | vk time / CUDA time |", "|---|---|---|---|---|"]
    text = "\n".join(head + rows) + "\n"
    print()
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
