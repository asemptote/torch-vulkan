"""A chaotic rate RNN (Sompolinsky, Crisanti and Sommers 1988) as an
end-to-end scientific workload, in plain eager PyTorch.

    dx/dt = -x + J tanh(x),   J_ij ~ N(0, g^2 / N),   g = 1.5

integrated by forward Euler (dt = 0.1). Three workloads, timed as whole runs
by benchmarks/vs_cuda.py (which embeds this file):
- one trajectory, N = 1000 or 4000, 1000 steps: a matrix-vector product per
  step (latency- and bandwidth-bound);
- a batch of 256 trajectories, N = 4000, 200 steps: a matrix product per step;
- the leading Lyapunov exponents by Benettin's method: k = 64 tangent
  vectors propagated with the Euler step's Jacobian
  (1 - dt) I + dt J diag(1 - tanh(x)^2), re-orthonormalised by
  torch.linalg.qr every 10 steps, log |diag R| accumulated; N = 1000,
  500 steps.

J and the initial states come from a seeded CPU generator in float64 and are
then cast, so every device starts from the same numbers.

    python benchmarks/rnn.py --device vk            # the exponents on one device
    python benchmarks/rnn.py --compare --vk-python VENV_VK/bin/python --cuda-python VENV_CUDA/bin/python

--compare runs both and prints the leading exponents side by side. The
system is chaotic, but at t = 50 rounding differences have grown only by
about e^(lambda_max t), so float64 agrees to many digits and float32 to two
or three: a correctness check of the whole pipeline (matrix products,
tanh, QR).
"""

import torch

G, DT, SEED = 1.5, 0.1, 0


def network(n, dtype, dev, batch=None, k=None):
    """J [n, n], the initial state(s) and, with k, orthonormal tangent vectors."""
    gen = torch.Generator().manual_seed(SEED)
    J = torch.randn(n, n, generator=gen, dtype=torch.float64) * (G / n ** 0.5)
    x = torch.randn(n, *([batch] if batch else []), generator=gen, dtype=torch.float64)
    out = [J.to(dtype).to(dev), x.to(dtype).to(dev)]
    if k:
        q, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float64))
        out.append(q.to(dtype).to(dev))
    return out


def trajectory(J, x, steps):
    """`steps` Euler steps of one trajectory x [n] (or a batch, x [n, b])."""
    for _ in range(steps):
        x = x + DT * (J @ torch.tanh(x) - x)
    return x


def lyapunov(J, x, q, steps, every=10):
    """The leading exponents (per unit time) from tangent vectors q [n, k]."""
    acc = torch.zeros(q.shape[1], dtype=q.dtype, device=q.device)
    for t in range(1, steps + 1):
        d = 1 - torch.tanh(x) ** 2
        q = q + DT * (J @ (d[:, None] * q) - q)  # the Euler step's Jacobian times q
        x = x + DT * (J @ torch.tanh(x) - x)
        if t % every == 0 or t == steps:
            q, r = torch.linalg.qr(q)
            acc = acc + torch.log(torch.abs(torch.diagonal(r)))
    return acc / (steps * DT)


def exponents(dev, dtype, n=1000, k=64, steps=500):
    J, x, q = network(n, dtype, dev, k=k)
    return lyapunov(J, x, q, steps).cpu().tolist()


def main():
    import argparse
    import json
    import subprocess
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--vk-python", default="")
    ap.add_argument("--cuda-python", default="")
    args = ap.parse_args()
    if args.device:
        if args.device == "vk":
            import torch_vulkan  # noqa: F401
        print(json.dumps({str(dt)[6:]: exponents(args.device, dt) for dt in (torch.float32, torch.float64)}))
        return
    if args.compare:
        res = {}
        for dev, py in (("vk", args.vk_python), ("cuda", args.cuda_python)):
            r = subprocess.run([py, __file__, "--device", dev], capture_output=True, text=True)
            lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
            if not lines:
                sys.exit(f"{dev}: {r.stderr.strip()[-500:]}")
            res[dev] = json.loads(lines[-1])
        for dt in ("float32", "float64"):
            a, b = res["vk"][dt], res["cuda"][dt]
            diff = max(abs(u - v) for u, v in zip(a, b))
            print(f"{dt}: leading exponents vk {', '.join(f'{v:.6f}' for v in a[:4])}")
            print(f"{' ' * len(dt)}  leading exponents CUDA {', '.join(f'{v:.6f}' for v in b[:4])}")
            print(f"{' ' * len(dt)}  max |difference| over all {len(a)}: {diff:.2e}")
        return
    ap.print_help()


if __name__ == "__main__" and not globals().get("_RNN_EMBEDDED"):
    main()
