"""FFT on device "vk" vs PyTorch on the CPU.

    python benchmarks/fft.py
"""

import time

import torch

import torch_vulkan as tv


def timed(fn, sync, reps=5):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


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
