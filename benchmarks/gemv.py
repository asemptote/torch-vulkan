"""Matrix-vector products: GB/s of weights streamed, on "vk" and on the CPU.

    python benchmarks/gemv.py                 # 4096 x 4096 weights, 1 to 8 vectors
    python benchmarks/gemv.py 2048 1 4        # n x n weights, these vector counts
    TORCH_VK_GEMV=scalar python benchmarks/gemv.py   # the previous paths, for comparison

x @ W.t() (nn.Linear's decoding step: W [n, n] row-major, x [cols, n]) and
W @ x.t() for float32 and float16 weights, int8 weights with a scale per row
(_weight_int8pack_mm) and 4-bit weights with a scale and zero per group of
32 (_weight_int4pack_mm; on the CPU its own packing, _for_cpu). A product
reads every weight once, so it is bound by memory bandwidth: GB/s counts
the weights' bytes (and the int4 scales and zeros) per second. The CPU runs
with torch's thread count (set it with OMP_NUM_THREADS or --threads N).
"""

import os
import sys
import time

import torch

import torch_vulkan as tv

SMALL = os.environ.get("TORCH_VK_BENCH_SMALL") == "1"


def seconds(fn, sync, budget=0.5):
    fn()
    sync()
    start = time.perf_counter()
    fn()
    sync()
    once = time.perf_counter() - start
    # (queued without waiting, so that the GPU never idles; as many as fit
    # in the budget, not more: they all run before the final sync)
    reps = max(3, int(budget / max(once, 1e-6)))
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


def quantise4(w, group=32):
    n, k = w.shape
    g = w.reshape(-1, group)
    lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-6) / 15
    q = g.sub(lo).div(scale).round().clamp(0, 15).to(torch.int32).reshape(n, k)
    sz = torch.cat([scale.reshape(n, -1, 1), (lo + 8 * scale).reshape(n, -1, 1)], 2).transpose(0, 1).contiguous()
    return q, sz


def main():
    args = sys.argv[1:]
    if "--threads" in args:
        i = args.index("--threads")
        torch.set_num_threads(int(args[i + 1]))
        del args[i : i + 2]
    nums = [int(a) for a in args]
    n = nums[0] if nums else (512 if SMALL else 4096)
    cols = nums[1:] or [1, 2, 4, 8]
    print(f"{tv.device_name()} (TORCH_VK_GEMV={os.environ.get('TORCH_VK_GEMV', 'default')}) vs CPU "
          f"({torch.get_num_threads()} threads), torch {torch.__version__}; {n} x {n} weights")
    w = torch.randn(n, n)
    q4, sz = quantise4(w)
    packed4 = (q4[:, ::2] << 4 | q4[:, 1::2]).to(torch.uint8)
    w8 = torch.randint(-128, 128, (n, n), dtype=torch.int8)
    s8 = torch.rand(n) / 50
    vk_w4 = torch._convert_weight_to_int4pack(packed4.to("vk"), 2)
    try:
        cpu_w4 = torch._convert_weight_to_int4pack_for_cpu(q4, 2)
    except RuntimeError:
        cpu_w4 = None
    cases = [
        ("float32", 4 * n * n, lambda x, d: torch.nn.functional.linear(x, w32[d]), torch.float32),
        ("float32 W@x", 4 * n * n, lambda x, d: w32[d] @ x.t(), torch.float32),
        ("float16", 2 * n * n, lambda x, d: torch.nn.functional.linear(x, w16[d]), torch.float16),
        ("float16 W@x", 2 * n * n, lambda x, d: w16[d] @ x.t(), torch.float16),
        ("int8", n * n, lambda x, d: torch._weight_int8pack_mm(x, w8d[d], s8d[d]), torch.float32),
        ("int4", n * n // 2 + sz.numel() * 4, None, torch.float32),
    ]
    w32 = {"cpu": w, "vk": w.to("vk")}
    w16 = {"cpu": w.half(), "vk": w.half().to("vk")}
    w8d = {"cpu": w8, "vk": w8.to("vk")}
    s8d = {"cpu": s8, "vk": s8.to("vk")}
    szd = {"cpu": sz, "vk": sz.to("vk")}

    def int4(x, d):
        if d == "vk":
            return torch._weight_int4pack_mm(x, vk_w4, 32, szd[d])
        return torch._weight_int4pack_mm_for_cpu(x, cpu_w4, 32, szd[d])

    for name, nbytes, fn, dtype in cases:
        fn = fn or int4
        for c in cols:
            x = torch.randn(c, n).to(dtype)
            xv = x.to("vk")
            t_vk = seconds(lambda: fn(xv, "vk"), tv.synchronize)
            line = f"  {name:11s} {c} col   vk {t_vk * 1e3:8.3f} ms {nbytes / t_vk / 1e9:6.2f} GB/s"
            if name != "int4" or cpu_w4 is not None:
                try:
                    t_cpu = seconds(lambda: fn(x, "cpu"), lambda: None)
                    line += f"   cpu {t_cpu * 1e3:8.3f} ms {nbytes / t_cpu / 1e9:6.2f} GB/s"
                except RuntimeError as e:
                    line += f"   cpu: {str(e).splitlines()[0][:60]}"
            print(line, flush=True)
    print(f"  CPU fallbacks: {tv.fallback_counts() or 'none'}")


if __name__ == "__main__":
    main()
