"""Matrix multiply on device "vk" vs PyTorch on the CPU (GFLOP/s).

    python benchmarks/matmul.py
    TORCH_VK_GEMM=tiled python benchmarks/matmul.py   # gemm.slang's tiles instead of gemmsg.slang

a @ b with both operands row-major, and a @ b.T (nn.Linear's x @ W.t()) with
a column-major second operand.
"""

import os
import time

import torch

import torch_vulkan as tv


def gflops(fn, flops, sync, reps=5):
    fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return flops * reps / (time.perf_counter() - start) / 1e9


def main():
    kernel = os.environ.get("TORCH_VK_GEMM", "default")
    print(f"{tv.device_name()} (TORCH_VK_GEMM={kernel}) vs CPU ({torch.get_num_threads()} threads), "
          f"torch {torch.__version__}")
    for dtype in (torch.float32, torch.float64):
        for n in (256, 1024, 2048):
            a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
            va, vb = a.to("vk"), b.to("vk")
            flops = 2 * n**3
            g_vk = gflops(lambda: va @ vb, flops, tv.synchronize)
            g_vkt = gflops(lambda: va @ vb.t(), flops, tv.synchronize)
            g_cpu = gflops(lambda: a @ b, flops, lambda: None)
            print(f"  {str(dtype)[6:]:8s} {n:5d}^3   vk {g_vk:7.1f}   vk a@b.T {g_vkt:7.1f}   cpu {g_cpu:7.1f} GFLOP/s")
    print(f"  CPU fallbacks: {tv.fallback_counts() or 'none'}")


if __name__ == "__main__":
    main()
