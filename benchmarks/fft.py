"""FFT on device "vk" vs PyTorch on the CPU.

    python benchmarks/fft.py
"""

import os
import statistics
import time

import torch

import torch_vulkan as tv

SMALL = os.environ.get("TORCH_VK_BENCH_SMALL") == "1"  # (benchmarks/integrated.py --small: a quick check)


def timed(fn, sync, reps=5, blocks=7):
    """The median over `blocks` of the mean time of `reps` calls: some
    drivers (Intel's on Windows, 2^20 complex64) switch between a fast and a
    slow mode for a whole block, which a single mean would land on by chance."""
    fn()
    sync()
    times = []
    for _ in range(blocks):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        times.append((time.perf_counter() - start) / reps)
    return statistics.median(times)


def main():
    print(f"{tv.device_name()} vs CPU ({torch.get_num_threads()} threads), torch {torch.__version__}")
    cases = [
        ("fft 2^20", (1 << 20,), lambda t: torch.fft.fft(t), True),
        ("fft 64 x 2^14", (64, 1 << 14), lambda t: torch.fft.fft(t), True),
        ("fft 1000003 (Bluestein)", (1_000_003,), lambda t: torch.fft.fft(t), True),
        ("fft2 1024x1024", (1024, 1024), lambda t: torch.fft.fft2(t), True),
        ("rfft 2^22", (1 << 22,), lambda t: torch.fft.rfft(t), False),
        ("irfft 2^20", ((1 << 19) + 1,), lambda t: torch.fft.irfft(t), True),
    ]
    if SMALL:
        cases = [(label, tuple(max(2, d // 256) for d in shape), f, c) for label, shape, f, c in cases]
    for dtype in (torch.complex64, torch.complex128):
        real = torch.float32 if dtype == torch.complex64 else torch.float64
        for label, shape, f, complex_input in cases:
            x = torch.randn(shape, dtype=dtype if complex_input else real)
            xv = x.to("vk")
            t_vk = timed(lambda: f(xv), tv.synchronize)
            t_cpu = timed(lambda: f(x), lambda: None)
            print(f"  {label:24s} {str(dtype)[6:]:10s} vk {t_vk * 1e3:8.1f} ms   cpu {t_cpu * 1e3:8.1f} ms")
    print(f"  CPU fallbacks: {tv.fallback_counts() or 'none'}")


if __name__ == "__main__":
    main()
