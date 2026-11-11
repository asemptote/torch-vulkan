"""Element-wise performance on device "vk" vs PyTorch on the CPU.

    python benchmarks/elementwise.py
"""

import time

import torch

import torch_vulkan as tv


def timed(fn, reps, sync):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


def main():
    print(f"{tv.device_name()} vs CPU ({torch.get_num_threads()} threads), torch {torch.__version__}")
    cpu_sync = lambda: None  # noqa: E731

    # Dispatch overhead: tiny tensors, so the time is all Python + submission.
    small, small_cpu = torch.ones(1024, device="vk"), torch.ones(1024)
    t = timed(lambda: small + 1, 2000, tv.synchronize)
    print(f"  per-op overhead, 1024 floats:     vk {t * 1e6:7.1f} us   cpu {timed(lambda: small_cpu + 1, 2000, cpu_sync) * 1e6:7.1f} us")

    # Throughput: y = relu(x * 2 + 1) on 16M elements (3 kernels on vk).
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(1 << 24, dtype=dtype)
        xv = x.to("vk")
        reps = 5
        tv_t = timed(lambda: (xv * 2 + 1).relu(), reps, tv.synchronize)
        cpu_t = timed(lambda: (x * 2 + 1).relu(), reps, cpu_sync)
        gbs = 3 * 2 * x.numel() * x.element_size() / tv_t / 1e9
        print(f"  relu(x*2+1), 16M {str(dtype)[6:]:8s}       vk {tv_t * 1e3:7.1f} ms   cpu {cpu_t * 1e3:7.1f} ms   ({gbs:.1f} GB/s on vk)")
    # Division (correctly rounded: _fmath.slang's m_div) on 16M elements.
    for dtype in (torch.float32, torch.float64):
        x, y = torch.randn(1 << 24, dtype=dtype), torch.randn(1 << 24, dtype=dtype)
        xv, yv = x.to("vk"), y.to("vk")
        for label, f in [("x / y", lambda a, b: a / b), ("1 / x", lambda a, b: 1 / a)]:
            tv_t = timed(lambda: f(xv, yv), 5, tv.synchronize)
            cpu_t = timed(lambda: f(x, y), 5, cpu_sync)
            print(f"  {label:22s} 16M {str(dtype)[6:]:8s} vk {tv_t * 1e3:7.1f} ms   cpu {cpu_t * 1e3:7.1f} ms")
    # Reductions and scans on 16M elements.
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(1 << 24, dtype=dtype)
        xv = x.to("vk")
        m = x.view(4096, 4096)
        mv = xv.view(4096, 4096)
        for label, f in [
            ("sum()", lambda t: t.sum()),
            ("var(dim=1) 4096x4096", None),
            ("amax(dim=0) 4096x4096", None),
            ("cumsum", lambda t: t.cumsum(0)),
        ]:
            if f is None:
                f = (lambda t: t.var(1)) if label.startswith("var") else (lambda t: t.amax(0))
                a, b = mv, m
            else:
                a, b = xv, x
            tv_t = timed(lambda: f(a), 5, tv.synchronize)
            cpu_t = timed(lambda: f(b), 5, cpu_sync)
            print(f"  {label:22s} 16M {str(dtype)[6:]:8s} vk {tv_t * 1e3:7.1f} ms   cpu {cpu_t * 1e3:7.1f} ms")
    print(f"  CPU fallbacks during benchmark: {tv.fallback_counts() or 'none'}")


if __name__ == "__main__":
    main()
