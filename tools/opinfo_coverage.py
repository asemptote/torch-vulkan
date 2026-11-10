"""Which ops run fully on Vulkan, which fall back to the CPU, and which are wrong.

Runs every OpInfo in PyTorch's test database (torch.testing._internal) on
device "vk" with its sample inputs, for a few dtypes, and records per op:
  - "gpu":      no CPU fallback was used and results match the CPU
  - "fallback": correct, but these aten ops ran on the CPU (listed)
  - "mismatch": result differs from the CPU
  - "error":    raised on vk but not on the CPU
  - "crash":    killed the process (e.g. an access violation)
Writes tools/opinfo_coverage.json and prints a summary.

    python tools/opinfo_coverage.py [--jobs 4] [--dtypes float32,float64] [--samples 3] [--filter substr]

The op list is split across --jobs worker processes (each limited to one
PyTorch CPU thread so they do not oversubscribe the cores). Workers append
one JSON line per op; if a worker dies, the op it was running is recorded as
a crash and a new worker continues after it.
"""

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils._pytree import tree_flatten, tree_map

import torch_vulkan as tv

DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "bool": torch.bool,
    "complex64": torch.complex64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "uint16": torch.uint16,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}


# Not comparable with the CPU: uninitialized or random outputs.
SKIP_COMPARE = ("empty", "new_empty", "rand", "normal", "bernoulli", "multinomial", "dropout", "uniform",
                "exponential", "geometric", "cauchy", "log_normal")
SKIP_ENTIRELY = ("sparse.", "to_sparse", "tensor_split")
# Ops returning (values, indices) where PyTorch leaves the index among equal
# values unspecified (the CPU uses unstable partial sorts): the values must
# match, and the indices must point at them.
TIED_INDICES = ("kthvalue", "median", "nanmedian", "mode", "topk", "sort")
# Ops returning only indices of a (possibly tied) ordering: compared by the
# values they select.
TIED_ORDER = ("argsort",)
# Ops whose vector outputs are unique only up to sign (eigenvectors,
# singular vectors): compared by absolute value.
SIGN_AMBIGUOUS = ("linalg.eigh", "linalg.svd", "svd")
# General eigendecompositions: eigenvalues in no particular order, vectors up
# to a phase; compared as sorted eigenvalues and by A V = V diag(w).
EIG = ("linalg.eig", "linalg.eigvals")
# Batch norm in evaluation mode: the CPU returns empty save_mean/save_invstd,
# other devices per-channel ones (PyTorch's meta kernel): the output only.
# (native_layer_norm: its 16-bit saved statistics are float32 on vk, as
# PyTorch's meta kernel and tests have them, and 16-bit on the CPU)
FIRST_OUTPUT_ONLY = ("native_batch_norm", "_native_batch_norm_legit", "native_layer_norm")


def op_name(info):
    return info.name + (f".{info.variant_test_name}" if info.variant_test_name else "")


def to_vk(x):
    """x on "vk" with the same storage layout: views keep their strides and
    offset into a copy of the whole storage (as_strided samples rely on it)."""
    if not isinstance(x, torch.Tensor):
        return x
    if x.layout != torch.strided or x.is_conj() or x.is_neg() or x.numel() == 0:
        return x.to("vk")
    nelem = x.untyped_storage().nbytes() // x.element_size()
    flat = x.new_empty(0).set_(x.untyped_storage(), 0, (nelem,), (1,))
    return flat.to("vk").as_strided(x.size(), x.stride(), x.storage_offset())


def to_cpu(x):
    return x.cpu() if isinstance(x, torch.Tensor) else x


def run_sample(op, sample):
    args = (sample.input, *sample.args)
    cpu_out = op(*args, **sample.kwargs)
    vk_args, vk_kwargs = tree_map(to_vk, (args, sample.kwargs))
    before = tv.fallback_counts()
    vk_out = tree_map(to_cpu, op(*vk_args, **vk_kwargs))
    after = tv.fallback_counts()
    used = sorted(k for k, v in after.items() if v != before.get(k, 0))
    return cpu_out, vk_out, used


def is_random(op, sample):
    """True if the sample's CPU result depends on the RNG seed (dropout,
    fractional pooling, ...): its vk result comes from the vk generator and
    can't match, so only its fallbacks are recorded. OpInfo wraps such ops in
    wrapper_set_seed (fixed seed, RNG state restored afterwards), so that is
    swapped for one using our seed while we run the op with two seeds."""
    args = (sample.input, *sample.args)
    namespace = getattr(op, "__globals__", {})

    def run(seed):
        def seeded(f, *a, **k):
            torch.manual_seed(seed)
            return f(*a, **k)

        original = namespace.get("wrapper_set_seed")
        if original is not None:
            namespace["wrapper_set_seed"] = seeded
        try:
            torch.manual_seed(seed)
            return op(*args, **sample.kwargs)
        finally:
            if original is not None:
                namespace["wrapper_set_seed"] = original

    try:
        return compare(run(0), run(1)) is not None
    except Exception:
        return False


def _abs(x):
    return x.abs() if isinstance(x, torch.Tensor) and (x.is_floating_point() or x.is_complex()) else x


def compare_tied(name, sample, expected, actual):
    """compare() for TIED_INDICES ops: values, plus indices that select them."""
    problem = compare(expected[0], actual[0])
    if problem:
        return problem
    pos = 1 if name in ("kthvalue", "topk") else 0  # where dim sits among the positional args
    dim = sample.args[pos] if len(sample.args) > pos else sample.kwargs.get("dim", -1)
    x, values, indices = sample.input, actual[0], actual[1]
    if x.dim() == 0:
        return None if indices.item() == 0 else "index of a 0-dim input must be 0"
    if values.dim() < x.dim():
        values, indices = values.unsqueeze(dim), indices.unsqueeze(dim)
    picked = torch.gather(x, dim, indices)
    return compare(values, picked)


def compare_eig(sample, expected, actual):
    def ordered(w):
        key = torch.round(w.real, decimals=3) * 1e6 + torch.round(w.imag, decimals=3)
        return torch.gather(w, -1, torch.argsort(key, -1))

    exp_w = expected[0] if isinstance(expected, tuple) else expected
    act_w = actual[0] if isinstance(actual, tuple) else actual
    problem = compare(ordered(exp_w), ordered(act_w))
    if problem or not isinstance(actual, tuple):
        return problem
    a = sample.input.to(act_w.dtype)
    w, v = actual
    return compare(a @ v, v * w.unsqueeze(-2))


def compare(expected, actual):
    exp_flat, _ = tree_flatten(expected)
    act_flat, _ = tree_flatten(actual)
    if len(exp_flat) != len(act_flat):
        return f"structure differs ({len(exp_flat)} vs {len(act_flat)} leaves)"
    for e, a in zip(exp_flat, act_flat):
        if isinstance(e, torch.Tensor):
            if not isinstance(a, torch.Tensor):
                return f"expected tensor, got {type(a).__name__}"
            # (16-bit floats: a few of their ulps, as results are rounded once from float32)
            tol = max(1e-4, 4 * torch.finfo(e.dtype).eps) if e.dtype in (torch.float16, torch.bfloat16) else 1e-4
            try:
                torch.testing.assert_close(a, e, equal_nan=True, check_device=False, atol=tol, rtol=tol)
            except AssertionError as err:
                return str(err).splitlines()[0][:200]
        elif e != a and not (isinstance(e, float) and e != e and a != a):
            return f"{a!r} != {e!r}"
    return None


def max_error(x, ref):
    """Largest |x - ref| over float leaves (NaN where ref is not NaN: inf)."""
    worst = 0.0
    for a, r in zip(tree_flatten(x)[0], tree_flatten(ref)[0]):
        if not isinstance(a, torch.Tensor) or not (a.is_floating_point() or a.is_complex()):
            if isinstance(a, torch.Tensor) and not torch.equal(a.cpu(), r.cpu()):
                return float("inf")
            continue
        wide = torch.complex128 if a.is_complex() else torch.float64
        a, r = a.cpu().to(wide), r.cpu().to(wide)
        if a.shape != r.shape:
            return float("inf")
        both_nan = a.isnan() & r.isnan()
        same_inf = (a == r) & a.isinf()
        d = (a - r).abs().where(~(both_nan | same_inf), torch.zeros((), dtype=torch.float64))
        d = d.where(~d.isnan(), torch.full((), float("inf"), dtype=torch.float64))
        if d.numel():
            worst = max(worst, d.max().item())
    return worst


def as_accurate_as_cpu(op, sample, dtype, expected, actual):
    """True if the vk result is no further from the op computed in float64 (on
    the CPU) than the CPU's own result in `dtype`. For 16-bit dtypes the CPU
    often rounds to 16 bits at every step where vk computes in float32 and
    rounds once; for float32 and complex64, some vk ops (matrix_exp) compute
    in double precision where the CPU's single-precision result can be off
    in its smaller elements. The two then differ by more than the tolerance
    without vk being less accurate."""
    wide = torch.complex128 if dtype.is_complex else torch.float64

    def up(x):
        return x.to(wide) if isinstance(x, torch.Tensor) and x.dtype == dtype else x
    try:
        precise = op(*tree_map(up, (sample.input, *sample.args)), **tree_map(up, sample.kwargs))
    except Exception:
        return False
    return max_error(actual, precise) <= max_error(expected, precise)


# dtypes whose results may be judged by accuracy against float64 when they
# differ from the CPU's (as_accurate_as_cpu)
LOW_PRECISION = (torch.float16, torch.bfloat16, torch.float32, torch.complex64)


def check_op(info, dtypes, nsamples):
    """Result dict for one OpInfo, or None if it has no usable samples."""
    status, detail, fallbacks = "gpu", "", set()
    tried = 0
    for dtype in dtypes:
        if dtype not in info.supported_dtypes("cpu"):
            continue
        try:
            samples = list(info.sample_inputs("cpu", dtype, requires_grad=False))[:nsamples]
        except Exception:
            continue
        for sample in samples:
            try:
                expected, actual, used = run_sample(info.op, sample)
            except Exception as err:
                try:
                    info.op(sample.input, *sample.args, **sample.kwargs)
                except Exception:
                    continue  # fails on the CPU too: not a vk problem
                return {"status": "error", "detail": f"{dtype}: {type(err).__name__}: {str(err).splitlines()[0][:200]}",
                        "fallbacks": sorted(fallbacks)}
            tried += 1
            fallbacks.update(used)
            if any(s in op_name(info) for s in SKIP_COMPARE) or is_random(info.op, sample):
                continue
            name = op_name(info)
            if name in TIED_INDICES and isinstance(expected, tuple):
                problem = compare_tied(name, sample, expected, actual)
            elif name in TIED_ORDER and isinstance(expected, torch.Tensor) and sample.input.dim() > 0:
                dim = sample.args[0] if sample.args else sample.kwargs.get("dim", -1)
                x = sample.input
                problem = compare(torch.gather(x, dim, expected), torch.gather(x, dim, actual.cpu()))
            elif name in EIG:
                problem = compare_eig(sample, expected, actual)
            elif name in FIRST_OUTPUT_ONLY and isinstance(expected, tuple):
                problem = compare(expected[0], actual[0])
            elif name in SIGN_AMBIGUOUS:
                problem = compare(tree_map(_abs, expected), tree_map(_abs, actual))
            else:
                problem = compare(expected, actual)
            if problem and dtype in LOW_PRECISION and not problem.startswith("The values for attr")                     and as_accurate_as_cpu(info.op, sample, dtype, expected, actual):
                problem = None  # (differs from the CPU, but no less accurate)
            if problem:
                return {"status": "mismatch", "detail": f"{dtype}: {problem}", "fallbacks": sorted(fallbacks)}
    if tried == 0:
        return None
    return {"status": "fallback" if fallbacks else "gpu", "detail": detail, "fallbacks": sorted(fallbacks)}


def worker(args):
    """Runs ops index % N == K with index > --after, appending JSON lines to --out."""
    torch.set_num_threads(1)
    warnings.simplefilter("ignore")
    dtypes = [DTYPES[d] for d in args.dtypes.split(",")]
    shard, nshards = (int(x) for x in args.shard.split("/"))
    with open(args.out, "a", encoding="utf-8") as log:
        for index, info in enumerate(op_db):
            name = op_name(info)
            if index % nshards != shard or index <= args.after or (args.filter and args.filter not in name):
                continue
            if any(s in name for s in SKIP_ENTIRELY):
                continue  # sparse layouts are out of scope; tensor_split takes CPU index tensors
            log.write(json.dumps({"start": name, "index": index}) + "\n")
            log.flush()
            result = check_op(info, dtypes, args.samples)
            log.write(json.dumps({"done": name, "index": index, "result": result}) + "\n")
            log.flush()
            if result is not None:
                print(f"{result['status']:9s} {name}", flush=True)
    return 0


def run_shard(args, k, results):
    """Runs shard k to completion, restarting the worker after each crash."""
    part = pathlib.Path(args.out).with_suffix(f".part{k}.jsonl")
    part.unlink(missing_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    after = -1
    while True:
        cmd = [sys.executable, __file__, "--shard", f"{k}/{args.jobs}", "--after", str(after), "--out", str(part),
               "--dtypes", args.dtypes, "--samples", str(args.samples), "--filter", args.filter]
        code = subprocess.call(cmd, env=env)
        started = None
        for line in part.read_text(encoding="utf-8").splitlines() if part.exists() else []:
            entry = json.loads(line)
            if "start" in entry:
                started = entry
            else:
                started = None
                if entry["result"] is not None:
                    results[entry["done"]] = entry["result"]
        if code == 0 or started is None:
            break
        # The worker died while running `started`: record it and continue after it.
        results[started["start"]] = {"status": "crash", "detail": f"worker exit code {code}", "fallbacks": []}
        print(f"crash     {started['start']}", flush=True)
        with open(part, "a", encoding="utf-8") as log:
            log.write(json.dumps({"done": started["start"], "index": started["index"],
                                  "result": results[started["start"]]}) + "\n")
        after = started["index"]
    part.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtypes", default="float32,float64,int64,bool,complex64")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--filter", default="")
    parser.add_argument("--out", default=str(pathlib.Path(__file__).with_name("opinfo_coverage.json")))
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any op crashes, errors, mismatches or runs an aten op on the CPU")
    parser.add_argument("--shard", default="", help="internal: K/N, run every N-th op starting at K")
    parser.add_argument("--after", type=int, default=-1, help="internal: skip ops up to this index")
    args = parser.parse_args()
    if args.shard:
        return worker(args)

    import threading

    start = time.perf_counter()
    results = {}
    threads = [threading.Thread(target=run_shard, args=(args, k, results)) for k in range(max(1, args.jobs))]
    args.jobs = len(threads)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    report(results, args.out, time.perf_counter() - start)
    if args.strict and any(r["status"] != "gpu" for r in results.values()):
        return 1
    return 0


def report(results, out, seconds):
    counts = collections.Counter(r["status"] for r in results.values())
    missing = collections.Counter(f for r in results.values() for f in r["fallbacks"])
    ordered = dict(sorted(results.items()))
    pathlib.Path(out).write_text(json.dumps({"results": ordered, "missing_kernels": missing.most_common()}, indent=1))
    print(f"\n{len(results)} ops in {seconds:.0f} s: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    for status in ("crash", "error", "mismatch"):
        bad = [f"{n} ({r['detail'][:80]})" for n, r in ordered.items() if r["status"] == status]
        if bad:
            print(f"{status}: " + "; ".join(bad))
    print(f"{len(missing)} distinct aten ops ran on the CPU; most common:")
    for op, n in missing.most_common(40):
        print(f"  {n:4d}  {op}")


if __name__ == "__main__":
    sys.exit(main())
