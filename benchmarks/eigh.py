"""Where Jacobi eigh's time goes on device "vk".

    python benchmarks/eigh.py
    TORCH_VK_LINALG=cpu python benchmarks/eigh.py   # LAPACK on the CPU, for comparison

- An already diagonal matrix skips every rotation (s == 0 returns early in
  factor.slang), so it measures the dispatches and convergence checks alone.
- Random symmetric matrices at growing n, in float32 and float64: cost per
  sweep grows as n^3, plus whatever the memory system adds.
- A batch of 4 has the same number of dispatches as one matrix and 4 times
  the arithmetic and memory traffic.
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
    return (a + a.mT).to("vk")


def main():
    torch.manual_seed(0)
    route = os.environ.get("TORCH_VK_LINALG", "gpu")
    print(f"{tv.device_name()}, torch {torch.__version__}, TORCH_VK_LINALG={route}")
    for n in (128, 256, 512):
        d = torch.diag(torch.arange(1.0, n + 1, dtype=torch.float64)).to("vk")
        ms = timed(lambda: torch.linalg.eigvalsh(d), tv.synchronize) * 1000
        print(f"  diagonal float64 n={n:4d}           {ms:10.1f} ms  (dispatches only)")
    for dtype in (torch.float32, torch.float64):
        for n in (128, 256, 512):
            a = symmetric(n, dtype)
            ms = timed(lambda: torch.linalg.eigvalsh(a), tv.synchronize) * 1000
            print(f"  random   {str(dtype)[6:]:7s} n={n:4d}           {ms:10.1f} ms")
    a4 = symmetric(256, torch.float64, (4,))
    ms = timed(lambda: torch.linalg.eigvalsh(a4), tv.synchronize) * 1000
    print(f"  random   float64 n= 256, batch of 4 {ms:10.1f} ms")


if __name__ == "__main__":
    main()
