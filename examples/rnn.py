"""A chaotic random recurrent network (Sompolinsky, Crisanti and Sommers, 1988)
on device "vk", checked against the same Euler steps on the CPU.

    python examples/rnn.py --g 1.5
    python examples/rnn.py --n 8000 --steps 50 --float32 --save trajectory.npz

    dx/dt = -x + J tanh(x),   J_ij ~ N(0, g^2 / N),  J_ii = 0

For g < 1 the activity decays to zero; for g > 1 it stays O(1) and is
chaotic, so the GPU and CPU trajectories agree to rounding error at first and
then separate at the network's largest Lyapunov exponent. The network and the
initial state come from NumPy's generator (seeded), so a run is reproducible
on any device.
"""

import argparse
import time

import numpy as np
import torch

import torch_vulkan as tv


def simulate(J, x0, dt, steps, n_rec, sync=lambda: None):
    """Euler steps x <- x + dt (-x + J tanh(x)) on J's device; returns the first
    n_rec neurons' trajectory [steps, n_rec] and the time taken (s)."""
    x = x0.clone()
    traj = torch.empty(steps, n_rec, dtype=x.dtype, device=x.device)
    sync()
    start = time.perf_counter()
    for t in range(steps):
        x = torch.addmv(x, J, torch.tanh(x), beta=1 - dt, alpha=dt)  # (1 - dt) x + dt J tanh(x)
        traj[t] = x[:n_rec]
    sync()
    return traj, x, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="number of neurons")
    p.add_argument("--g", type=float, default=1.5, help="coupling gain (chaotic for g > 1)")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--record", type=int, default=10, help="neurons whose trajectories are kept")
    p.add_argument("--float32", action="store_true", help="float32 instead of float64")
    p.add_argument("--check", type=int, default=200, help="steps to compare against the CPU (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="save the vk trajectory to this .npz file")
    a = p.parse_args()

    dtype = torch.float32 if a.float32 else torch.float64
    n_rec = min(a.record, a.n)
    rng = np.random.default_rng(a.seed)
    J = a.g / np.sqrt(a.n) * rng.standard_normal((a.n, a.n))
    np.fill_diagonal(J, 0)
    x0 = rng.standard_normal(a.n)
    J_cpu, x0_cpu = torch.from_numpy(J).to(dtype), torch.from_numpy(x0).to(dtype)

    traj, x_end, seconds = simulate(J_cpu.to("vk"), x0_cpu.to("vk"), a.dt, a.steps, n_rec, tv.synchronize)
    traj = traj.cpu()
    print(f"{tv.device_name()} ({str(dtype)[6:]}), N={a.n}, g={a.g}, dt={a.dt}, "
          f"steps={a.steps} (t = {a.steps * a.dt:g})")
    print(f"vk:  {seconds:.3f} s  ({seconds / a.steps * 1e3:.3f} ms/step)")
    print(f"activity std at the end: {x_end.std().item():.4f}   (-> 0 for g < 1, stays O(1) for g > 1)")

    if a.check:
        k = min(a.check, a.steps)
        ref, _, cpu_seconds = simulate(J_cpu, x0_cpu, a.dt, k, n_rec)
        print(f"CPU: {cpu_seconds / k * a.steps:.3f} s for the same {a.steps} steps (extrapolated from {k})")
        err = (traj[:k] - ref).abs()
        scale = ref.abs().max().item()
        for s in sorted({1, 10, k // 2, k} - {0}):
            print(f"  max |vk - CPU| / max |x| after {s:4d} steps: {err[:s].max().item() / scale:.1e}")

    if a.save:
        np.savez(a.save, traj=traj.numpy(), dt=a.dt, g=a.g, N=a.n)
        print(f"saved {a.save} (traj[t, i] for the first {n_rec} neurons)")
    if tv.fallback_counts():
        print("CPU fallbacks:", tv.fallback_counts())


if __name__ == "__main__":
    main()
