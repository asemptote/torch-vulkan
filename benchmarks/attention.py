"""scaled_dot_product_attention: the fused kernel (attention.slang) against
the math path on "vk", milliseconds and peak device memory per call.

    python benchmarks/attention.py                    # every case, float32 and float16
    python benchmarks/attention.py --dtype bfloat16
    python benchmarks/attention.py --filter vit
    python benchmarks/attention.py --backward         # forward and backward

Peak memory is the most device memory the call's tensors (temporaries and
outputs, beyond the inputs) held at once (torch.accelerator's
max_memory_allocated); "fails" where the call raised (an allocation the
device refused, or on the math path a matrix product too large for one
launch, #131). The cases are attention layers of real models: DINOv2's ViT-B/14 at
518 x 518 (1,370 tokens; the TIMM benchmark's batch of 128 too), GPT-2's
causal attention, a Llama-style layer with grouped-query attention, a long
sequence and a small batch of short ones.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import torch_vulkan as tv

# name, batch, query heads, key heads, query length, key length, head dim, causal
CASES = [
    ("vit_base_patch14_dinov2 b8", 8, 12, 12, 1370, 1370, 64, False),
    ("vit_base_patch14_dinov2 b128", 128, 12, 12, 1370, 1370, 64, False),
    ("gpt2 causal b4 L1024", 4, 12, 12, 1024, 1024, 64, True),
    ("llama gqa 32/8 L2048 d128 causal", 1, 32, 8, 2048, 2048, 128, True),
    ("long L8192 h16", 1, 16, 16, 8192, 8192, 64, False),
    ("short b64 L128", 64, 8, 8, 128, 128, 64, False),
]
SMALL_CASES = [
    ("vit b1 L257", 1, 12, 12, 257, 257, 64, False),
    ("causal b2 L512", 2, 8, 8, 512, 512, 64, True),
]


def run(backend, q, k, v, causal, gqa, backward, g):
    with sdpa_kernel([backend]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa)
        if backward:
            torch.autograd.grad(out, (q, k, v), g)


def measure(backend, case, dtype, backward, reps):
    name, B, H, Hk, L, S, E, causal = case
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, L, E, generator=gen, dtype=dtype).to("vk").requires_grad_(backward)
    k = torch.randn(B, Hk, S, E, generator=gen, dtype=dtype).to("vk").requires_grad_(backward)
    v = torch.randn(B, Hk, S, E, generator=gen, dtype=dtype).to("vk").requires_grad_(backward)
    g = torch.randn(B, H, L, E, dtype=dtype).to("vk") if backward else None
    tv.synchronize()
    tv.empty_cache()
    torch.accelerator.reset_peak_memory_stats()
    base = torch.accelerator.memory_allocated()
    try:
        run(backend, q, k, v, causal, H != Hk, backward, g)
        tv.synchronize()
    except (RuntimeError, NotImplementedError):
        del q, k, v, g
        tv.synchronize()
        tv.empty_cache()
        return None, None
    peak = torch.accelerator.max_memory_allocated() - base
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        run(backend, q, k, v, causal, H != Hk, backward, g)
        tv.synchronize()
        best = min(best, time.perf_counter() - start)
    del q, k, v, g
    tv.empty_cache()
    return best * 1e3, peak / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", action="append", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--filter", default="")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--small", action="store_true", help="small cases only (lavapipe)")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    dtypes = [getattr(torch, d) for d in (args.dtype or ["float32", "float16"])]
    cases = [c for c in (SMALL_CASES if args.small else CASES) if args.filter in c[0]]
    print(f"device: {tv.device_name()}")
    print(f"{'forward and backward' if args.backward else 'forward'}, best of {args.reps}\n")
    print("| case | dtype | math ms | fused ms | speed-up | math MiB | fused MiB |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for case in cases:
        for dtype in dtypes:
            fm, fp = measure(SDPBackend.OVERRIDEABLE, case, dtype, args.backward, args.reps)
            mm, mp = measure(SDPBackend.MATH, case, dtype, args.backward, args.reps)
            dt = str(dtype).replace("torch.", "")
            ms = lambda x: "fails" if x is None else f"{x:.2f}"
            mib = lambda x: "fails" if x is None else f"{x:.0f}"
            speed = f"{mm / fm:.2f}x" if mm and fm else "-"
            print(f"| {case[0]} | {dt} | {ms(mm)} | {ms(fm)} | {speed} | {mib(mp)} | {mib(fp)} |", flush=True)


if __name__ == "__main__":
    main()
