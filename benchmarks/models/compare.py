"""vk's outputs against CUDA's, with the runner's accuracy test.

    python benchmarks/models/compare.py REF CUDA_OUT VK_OUT

REF: run.py --write-reference (the CPU's float64 outputs); CUDA_OUT, VK_OUT:
run.py --outputs. Prints one JSON object (see compare()).
"""

import json
import sys

import torch
from torch._dynamo.utils import same


def float_leaves(obj, out):
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point() or obj.is_complex():
            out.append(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            float_leaves(obj[k], out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            float_leaves(v, out)
    return out


def rel_rms(res, ref):
    """||res - ref|| / ||ref|| over every floating-point output (in float64)."""
    a, b = float_leaves(res, []), float_leaves(ref, [])
    if not a or len(a) != len(b):
        return None
    num = den = 0.0
    for x, y in zip(a, b):
        if x.shape != y.shape:
            return None
        x, y = x.to(torch.float64), y.to(torch.float64)
        num += float(torch.nan_to_num(x - y, nan=1e30, posinf=1e30, neginf=1e30).square().sum())
        den += float(torch.nan_to_num(y).square().sum())
    return (num / den) ** 0.5 if den > 0 else (0.0 if num == 0 else float("inf"))


def compare(ref_path, cuda_path, vk_path):
    """{"pass": the runner's check (vk against CUDA, the CPU's float64 as the
    reference), "err_vk"/"err_cuda": relative RMS error against float64,
    "deterministic_vk"/"deterministic_cuda": the device's two runs the same}."""
    fp64 = torch.load(ref_path, weights_only=False)["fp64"]
    cuda = torch.load(cuda_path, weights_only=False)
    vk = torch.load(vk_path, weights_only=False)
    kw = cuda["same_kwargs"]
    out = {"tol": kw["tol"], "cosine": kw["cos_similarity"], "deterministic_vk": vk["deterministic"],
           "deterministic_cuda": cuda["deterministic"]}

    def check(correct, new, ref):
        try:
            return bool(same(correct, new, ref, **kw))
        except Exception as e:  # (the runner counts this as a failure too)
            out.setdefault("errors", []).append(f"{type(e).__name__}: {e}"[:300])
            return False

    out["pass"] = check(cuda["outputs"], vk["outputs"], fp64)
    out["err_vk"] = rel_rms(vk["outputs"], fp64)
    out["err_cuda"] = rel_rms(cuda["outputs"], fp64)
    return out


if __name__ == "__main__":
    print(json.dumps(compare(*sys.argv[1:4])))
