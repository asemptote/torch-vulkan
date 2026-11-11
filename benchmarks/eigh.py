"""Symmetric eigendecomposition on device "vk".

    python benchmarks/eigh.py
    TORCH_VK_LINALG=cpu python benchmarks/eigh.py   # LAPACK on the CPU, for comparison

eigh runs as LAPACK's dsyevd does (csrc/dense.cpp tridiag_eigh): Householder
reduction to a tridiagonal, divide and conquer on it (shaders/stedc.slang;
QL below n = 64), then the back-transform with blocked reflectors.

- An already diagonal matrix: the reduction's dispatches with nothing to
  rotate, and every merge deflated.
- Random symmetric matrices at growing n, eigenvalues only (eigvalsh) and
  with eigenvectors (eigh), in float32, float64 and complex128.
- A batch of 4 shares the dispatches of one matrix.
"""

import os
import time

import torch

import torch_vulkan as tv


def timed(fn, sync, reps=1):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


def symmetric(n, dtype, batch=()):
    a = torch.randn(*batch, n, n, dtype=dtype)
    return (a + a.mH).to("vk")


def main():
    torch.manual_seed(0)
    route = os.environ.get("TORCH_VK_LINALG", "gpu")
    print(f"{tv.device_name()}, torch {torch.__version__}, TORCH_VK_LINALG={route}")
    for n in (128, 256, 512):
        d = torch.diag(torch.arange(1.0, n + 1, dtype=torch.float64)).to("vk")
        ms = timed(lambda: torch.linalg.eigvalsh(d), tv.synchronize) * 1000
        print(f"  diagonal float64 n={n:4d}           {ms:10.1f} ms  (dispatches only)")
    for dtype in (torch.float32, torch.float64):
        for n in (128, 256, 512, 1024):
            a = symmetric(n, dtype)
            ms = timed(lambda: torch.linalg.eigvalsh(a), tv.synchronize) * 1000
            print(f"  random   {str(dtype)[6:]:7s} n={n:4d}           {ms:10.1f} ms")
    for dtype in (torch.float32, torch.float64, torch.complex128):
        for n in (512, 1024, 2048):
            if dtype == torch.complex128 and n > 1024:
                continue
            a = symmetric(n, dtype)
            ms = timed(lambda: torch.linalg.eigh(a), tv.synchronize) * 1000
            print(f"  eigh     {str(dtype)[6:]:10s} n={n:4d}        {ms:10.1f} ms  (with eigenvectors)")
    a4 = symmetric(256, torch.float64, (4,))
    ms = timed(lambda: torch.linalg.eigvalsh(a4), tv.synchronize) * 1000
    print(f"  random   float64 n= 256, batch of 4 {ms:10.1f} ms")


if __name__ == "__main__":
    main()
