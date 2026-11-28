"""How much device memory a process can hold on "vk" (#133, docs/log/M55).

    python benchmarks/memtest.py [MIB [LIMIT]]

Allocates 1 GiB (or MIB MiB) float32 tensors, each filled on the GPU, until an
allocation fails (or LIMIT tensors), then reads and writes the first (mapped)
and the last (unmapped, past the driver's mapped limit) and multiplies the two;
times host reads and writes."""
import sys
import time

import torch
import torch_vulkan as tv

mib = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
print(tv.device_name(), f"{mib} MiB tensors")
xs = []
t0 = time.perf_counter()
try:
    for i in range(limit):
        x = torch.empty(mib << 18, device="vk")  # (float32: mib MiB)
        x.fill_(float(i))
        xs.append(x)
        tv.synchronize()
except Exception as e:
    print(f"failed after {len(xs)} tensors ({len(xs) * mib / 1024:.2f} GiB):", type(e).__name__, str(e)[:1500])
else:
    print(f"held {len(xs)} tensors ({len(xs) * mib / 1024:.2f} GiB)")
print(f"{time.perf_counter() - t0:.1f} s;", tv.memory_stats())
if xs:
    for name, x in (("first", xs[0]), ("last", xs[-1])):
        tv.synchronize()
        t = time.perf_counter()
        h = x.cpu()
        r = time.perf_counter() - t
        t = time.perf_counter()
        x.copy_(h + 1)
        tv.synchronize()
        w = time.perf_counter() - t
        ok = bool((x[::4099].cpu() == h[::4099] + 1).all())
        print(f"{name}: read {h.numel() * 4 / r / 1e9:.2f} GB/s, write (incl. the add on the CPU) "
              f"{h.numel() * 4 / w / 1e9:.2f} GB/s, values ok {ok}, item {x[5].item()}")
    a, b = xs[0][: 1 << 20].view(1024, 1024), xs[-1][: 1 << 20].view(1024, 1024)
    c = a @ b
    print("product of a mapped and an unmapped tensor:", c[0, 0].item(), "expected", 1024.0 * 1.0 * (len(xs)))
del xs
tv.empty_cache()
print("after empty_cache:", tv.memory_stats())
