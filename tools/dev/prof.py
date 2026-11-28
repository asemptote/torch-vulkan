"""Per-op profile of a model's inference on vk (or cuda), float32 / bfloat16 / amp.

    python tools/dev/prof.py MODEL PRECISION [--device vk] [--batch N] [--ops 30] [--trace]

Times the whole forward (median of --repeat, synchronised), then runs it once
under a TorchDispatchMode that synchronises around each aten op and records
its time and launches, grouped by (op, dtypes). With --trace (needs
TORCH_VK_TRACE=1 in the environment) also the kernel names each op launched.
"""
import argparse
import collections
import os
import statistics
import sys
import tempfile
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

MODELS = {
    # name: (builder, batch)
    "resnet18": ("tv", 16),
    "mobilenet_v2": ("tv", 16),
    "mobilenetv2_100": ("timm", 128),
    "convnextv2_nano.fcmae_ft_in22k_in1k": ("timm", 128),
    "deit_tiny_patch16_224.fb_in1k": ("timm", 128),
    "BertForMaskedLM": ("hf", 16),
}


def build(name, batch, device):
    kind, default_batch = MODELS[name]
    batch = batch or default_batch
    if kind == "tv":
        import torchvision
        m = getattr(torchvision.models, name)()
        x = torch.randn(batch, 3, 224, 224)
        return m, (x,)
    if kind == "timm":
        import timm
        m = timm.create_model(name, pretrained=False)
        cfg = timm.data.resolve_data_config({}, model=m)
        x = torch.randn(batch, *cfg["input_size"])
        return m, (x,)
    if kind == "hf":
        import transformers
        cfg = transformers.BertConfig()
        m = transformers.BertForMaskedLM(cfg)
        ids = torch.randint(0, cfg.vocab_size, (batch, 512))
        return m, (ids,)
    raise ValueError(name)


def sync(device):
    if device == "vk":
        import torch_vulkan
        torch_vulkan.synchronize()
    else:
        torch.cuda.synchronize()


def desc(a):
    if isinstance(a, torch.Tensor):
        return {torch.float32: "f32", torch.bfloat16: "bf16", torch.float16: "f16", torch.int64: "i64",
                torch.bool: "b"}.get(a.dtype, str(a.dtype).replace("torch.", ""))
    if isinstance(a, (list, tuple)) and a and isinstance(a[0], torch.Tensor):
        return "[" + desc(a[0]) + "]"
    return None


class PerOp(TorchDispatchMode):
    def __init__(self, device, trace_fd=None):
        super().__init__()
        self.device = device
        self.time = collections.Counter()
        self.calls = collections.Counter()
        self.launches = collections.Counter()
        self.kernels = collections.defaultdict(collections.Counter)
        self.shapes = {}
        self.trace_fd = trace_fd
        self.native = None
        if device == "vk":
            from torch_vulkan import _native
            self.native = _native

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        sync(self.device)
        before = self.native.dispatch_count() if self.native else 0
        pos = os.lseek(self.trace_fd, 0, os.SEEK_CUR) if self.trace_fd is not None else 0
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        sync(self.device)
        dt = time.perf_counter() - t0
        name = func.overloadpacket.__name__ if hasattr(func, "overloadpacket") else str(func)
        dts = ",".join(d for d in (desc(a) for a in args) if d)
        outd = desc(out) or ""
        key = f"{name}({dts})->{outd}"
        self.time[key] += dt
        self.calls[key] += 1
        if self.native:
            self.launches[key] += self.native.dispatch_count() - before
        if key not in self.shapes:
            self.shapes[key] = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)][:3]
        if self.trace_fd is not None:
            end = os.lseek(self.trace_fd, 0, os.SEEK_END)
            os.lseek(self.trace_fd, pos, os.SEEK_SET)
            text = os.read(self.trace_fd, end - pos).decode(errors="replace")
            os.lseek(self.trace_fd, end, os.SEEK_SET)
            for line in text.splitlines():
                if line.startswith("[torch_vulkan] "):
                    parts = line.split()
                    self.kernels[key][parts[1] + (" " + parts[2] if len(parts) > 2 and parts[2].startswith("[") else "")] += 1
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("precision", choices=["float32", "bfloat16", "amp", "amp_bf16"])
    p.add_argument("--device", default="vk")
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--ops", type=int, default=30)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--no-profile", action="store_true")
    a = p.parse_args()
    if a.device == "vk":
        import torch_vulkan  # noqa: F401
    torch.manual_seed(0)
    m, inputs = build(a.model, a.batch, a.device)
    m.eval()
    if a.precision == "bfloat16":
        m = m.to(torch.bfloat16)
        inputs = tuple(x.to(torch.bfloat16) if x.is_floating_point() else x for x in inputs)
    m = m.to(a.device)
    inputs = tuple(x.to(a.device) for x in inputs)
    torch.backends.cuda.matmul.allow_tf32 = True  # (as the runners: "high")
    torch.set_float32_matmul_precision("high")
    import contextlib
    if a.precision.startswith("amp"):
        ctx = lambda: torch.autocast(a.device, dtype=torch.bfloat16 if a.precision == "amp_bf16" else torch.float16)  # noqa: E731
    else:
        ctx = contextlib.nullcontext

    def run():
        with torch.no_grad(), ctx():
            return m(*inputs)

    for _ in range(3):
        run()
    sync(a.device)
    times = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        run()
        sync(a.device)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times) * 1e3
    print(f"== {a.model} {a.precision} {a.device}: {med:.2f} ms (min {min(times) * 1e3:.2f})", flush=True)
    if a.no_profile:
        return
    trace_fd = None
    if a.trace:
        tf = tempfile.TemporaryFile()
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tf.fileno(), 2)
        trace_fd = 2
    mode = PerOp(a.device, trace_fd)
    with mode:
        run()
    if a.trace:
        os.dup2(saved, 2)
    total = sum(mode.time.values()) * 1e3
    launches = sum(mode.launches.values())
    print(f"   per-op synchronised total {total:.1f} ms, {sum(mode.calls.values())} op calls, {launches} launches")
    by_op = collections.Counter()
    for k, v in mode.time.items():
        by_op[k.split("(")[0]] += v
    print("   by op: " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in by_op.most_common(15)))
    print(f"   {'ms':>7} {'%':>5} {'calls':>5} {'launch':>6}  op(dtypes)  first shapes  [kernels]")
    for k, v in mode.time.most_common(a.ops):
        ks = ""
        if mode.kernels.get(k):
            ks = "  " + ", ".join(f"{n} x{c}" for n, c in mode.kernels[k].most_common(6))
        print(f"   {v * 1e3:7.2f} {100 * v * 1e3 / total:5.1f} {mode.calls[k]:5d} {mode.launches[k]:6d}  {k}  "
              f"{mode.shapes[k]}{ks}")


if __name__ == "__main__":
    main()
