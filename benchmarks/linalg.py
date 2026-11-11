"""Dense linear algebra on device "vk" vs PyTorch (LAPACK/MKL) on the CPU.

    python benchmarks/linalg.py
"""

import os
import time

import torch

import torch_vulkan as tv

SMALL = os.environ.get("TORCH_VK_BENCH_SMALL") == "1"  # (benchmarks/integrated.py --small: a quick check)


def timed(fn, sync, reps=3):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


def main():
    print(f"{tv.device_name()} vs CPU ({torch.get_num_threads()} threads), torch {torch.__version__}")
    for dtype in (torch.float32, torch.float64):
        for n in (64,) if SMALL else (512, 2048):
            a = torch.randn(n, n, dtype=dtype) + n**0.5 * torch.eye(n, dtype=dtype)
            spd = a @ a.T
            b = torch.randn(n, 8, dtype=dtype)
            va, vspd, vb = a.to("vk"), spd.to("vk"), b.to("vk")
            for label, f, args_vk, args_cpu in [
                ("solve", torch.linalg.solve, (va, vb), (a, b)),
                ("cholesky", torch.linalg.cholesky, (vspd,), (spd,)),
                ("inv", torch.linalg.inv, (va,), (a,)),
            ]:
                t_vk = timed(lambda: f(*args_vk), tv.synchronize)
                t_cpu = timed(lambda: f(*args_cpu), lambda: None)
                print(f"  {label:9s} {n:5d} {str(dtype)[6:]:8s} vk {t_vk * 1e3:8.1f} ms   cpu {t_cpu * 1e3:8.1f} ms")
    print(f"  CPU fallbacks: {tv.fallback_counts() or 'none'}")


if __name__ == "__main__":
    main()
