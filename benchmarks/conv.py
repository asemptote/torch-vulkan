"""Convolutions: milliseconds and GFLOP/s on "vk" (and optionally the CPU).

    python benchmarks/conv.py                  # every case, float32 and float16
    python benchmarks/conv.py --dtype float32  # one dtype
    python benchmarks/conv.py --filter transpose
    python benchmarks/conv.py --cpu            # the CPU too (torch's thread count)

The cases are Kokoro-82M's vocoder convolutions (one sentence: 29,041 audio
samples, 4,840 at the middle rate, 242 frames), its two upsampling transposed
convolutions, and 2-D shapes of an image model (a VAE decoder's stride-2
upsampling, a ResNet block). GFLOP/s counts the useful multiply-adds (2 per
tap that reaches the output, not the zeros a transposed convolution would
insert), so paths that do less work show it as speed.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

import torch_vulkan as tv

SMALL = os.environ.get("TORCH_VK_BENCH_SMALL") == "1"

# name, kind, batch, in channels, out channels, kernel, stride, padding, dilation, groups, spatial
CASES = [
    ("conv1d 128->128 k11 L29041", "conv", 1, 128, 128, 11, 1, 5, 1, 1, (29041,)),
    ("conv1d 128->128 k11 d3", "conv", 1, 128, 128, 11, 1, 15, 3, 1, (29041,)),
    ("conv1d 128->128 k7", "conv", 1, 128, 128, 7, 1, 3, 1, 1, (29041,)),
    ("conv1d 128->128 k3", "conv", 1, 128, 128, 3, 1, 1, 1, 1, (29041,)),
    ("conv1d 256->256 k11 L4840", "conv", 1, 256, 256, 11, 1, 5, 1, 1, (4840,)),
    ("conv1d 256->256 k3", "conv", 1, 256, 256, 3, 1, 1, 1, 1, (4840,)),
    ("conv1d 1024->1024 k3 L242", "conv", 1, 1024, 1024, 3, 1, 1, 1, 1, (242,)),
    ("conv_transpose1d 512->256 k20 s10", "transpose", 1, 512, 256, 20, 10, 5, 1, 1, (484,)),
    ("conv_transpose1d 256->128 k12 s6", "transpose", 1, 256, 128, 12, 6, 3, 1, 1, (4840,)),
    ("conv_transpose1d depthwise k3 s2", "transpose", 1, 640, 640, 3, 2, 1, 1, 640, (121,)),
    ("conv_transpose2d 256->128 k4 s2 64x64", "transpose", 1, 256, 128, 4, 2, 1, 1, 1, (64, 64)),
    ("conv_transpose2d 128->64 k3 s2 op1 b4", "transpose", 4, 128, 64, 3, 2, 1, 1, 1, (32, 32)),
    ("conv2d 64->64 k3 56x56 b8", "conv", 8, 64, 64, 3, 1, 1, 1, 1, (56, 56)),
]


def seconds(fn, sync, budget=0.5):
    fn()
    sync()
    start = time.perf_counter()
    fn()
    sync()
    once = time.perf_counter() - start
    reps = max(3, int(budget / max(once, 1e-6)))
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - start) / reps


def flops(kind, n, cin, cout, k, stride, groups, spatial):
    dims = len(spatial)
    taps = k**dims
    positions = 1
    for s in spatial:
        positions *= s
    if kind == "transpose":
        # every input position reaches K^dims outputs per output channel
        return 2 * n * positions * (cin // groups) * cout * taps
    out = positions  # (stride 1, same padding, in these cases)
    return 2 * n * out * (cin // groups) * cout * taps


def run(case, dtype, device):
    name, kind, n, cin, cout, k, stride, pad, dil, groups, spatial = case
    if SMALL:
        spatial = tuple(min(s, 64) for s in spatial)
    dims = len(spatial)
    g = torch.Generator().manual_seed(0)
    x = torch.randn((n, cin) + spatial, generator=g).to(device, dtype)
    if kind == "transpose":
        w = torch.randn((cin, cout // groups) + (k,) * dims, generator=g).div(k**0.5).to(device, dtype)
        op = 1 if "op1" in name else 0
        fn = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][dims - 1]
        call = lambda: fn(x, w, None, stride, pad, op, groups, dil)  # noqa: E731
    else:
        w = torch.randn((cout, cin // groups) + (k,) * dims, generator=g).div(k**0.5).to(device, dtype)
        fn = [F.conv1d, F.conv2d, F.conv3d][dims - 1]
        call = lambda: fn(x, w, None, stride, pad, dil, groups)  # noqa: E731
    sync = tv.synchronize if device == "vk" else (lambda: None)
    t = seconds(call, sync)
    return t, flops(kind, n, cin, cout, k, stride, groups, spatial) / t / 1e9


def main():
    args = sys.argv[1:]
    dtypes = ["float32", "float16"]
    if "--dtype" in args:
        i = args.index("--dtype")
        dtypes = [args[i + 1]]
        del args[i : i + 2]
    filt = None
    if "--filter" in args:
        i = args.index("--filter")
        filt = args[i + 1]
        del args[i : i + 2]
    cpu = "--cpu" in args
    env = {k: v for k, v in os.environ.items() if k.startswith("TORCH_VK_") and not k.startswith("TORCH_VK_BENCH")}
    print(f"{tv.device_name()} {env or ''}, torch {torch.__version__}")
    for dt in dtypes:
        dtype = getattr(torch, dt)
        for case in CASES:
            if filt is not None and filt not in case[0]:
                continue
            t, gf = run(case, dtype, "vk")
            line = f"{dt:8} {case[0]:42} {t * 1e3:9.2f} ms {gf:7.1f} GFLOP/s"
            if cpu:
                tc, gc = run(case, dtype, "cpu")
                line += f"   CPU {tc * 1e3:9.2f} ms {gc:7.1f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
