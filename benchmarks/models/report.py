"""Tables from sweep.py's results: Markdown, CSV and JSON.

    python benchmarks/models/report.py OUT/results.jsonl [...] --md results.md \\
        --csv results.csv --json results.json [--title TEXT] [--notes FILE]

A model "runs on vk" when every vk step that ran (the accuracy run, the
latency run) finished with the CPU fallback off. Models whose CUDA run
fails too (out of memory, a broken benchmark) are listed apart and left out
of the counts. Blockers: for each model that does not run on vk, the aten
ops that went to the CPU in its run with TORCH_VK_CPU_FALLBACK=1 (or, where
that run failed too, its error), ranked by the number of models each blocks.
"""

import argparse
import collections
import csv
import json
import math
import re


def last_by_key(paths):
    """One row per (suite, model, mode, precision); a later row's fields
    replace an earlier one's (a rerun of the accuracy steps alone, say)."""
    rows = {}
    for path in paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows.setdefault((r["suite"], r["model"], r["mode"], r["precision"]), {}).update(r)
    return list(rows.values())


def short_error(err):
    if not err:
        return ""
    m = re.search(r"(aten::[\w.]+) has no Vulkan kernel", err)
    if m:
        return f"{m.group(1)} (no Vulkan kernel)"
    err = re.sub(r"^RuntimeError: Eager run failed\s*", "", err)
    return err.replace("|", "/")[:160]


def op_name(key):
    # fallback_counts() keys: "aten::op", "aten::op (raised)", or with
    # TORCH_VK_FALLBACK_DETAIL=1 "aten::op [dtypes and shapes]"
    return key.split(" [")[0].replace(" (refused)", "")


CUDA_ONLY_BENCHMARKS = {
    "moco": "its benchmark raises NotImplementedError for any device but cuda (DistributedDataParallel)",
    "nvidia_deeprecommender": "its benchmark builds the model only for cpu and cuda",
}


def status(rec):
    return (rec or {}).get("status")


def summarise(r):
    s = {k: r[k] for k in ("suite", "model", "mode", "precision")}
    acc_cuda, acc_vk, perf_cuda, perf_vk = (r.get(k) for k in ("acc_cuda", "acc_vk", "perf_cuda", "perf_vk"))
    # (the accuracy run's fallback rerun, or the latency run's where only it failed)
    fb = (r.get("acc_vk_fallback") if status(r.get("acc_vk")) != "captured" else None) or (
        r.get("perf_vk_fallback") if r.get("vk_ms") is None else None)
    def ok(x, ms):
        return status(x) in ("captured", "pass") and not (ms in r and r.get(ms) is None)

    # each vk step beside its CUDA step: a step that fails on CUDA too (a
    # model that does not fit, a benchmark broken on this release) says
    # nothing about vk
    pairs = [(v, c, ok(v, ms) if v else None, ok(c, ms) if c else None)
             for v, c, ms in ((acc_vk, acc_cuda, None), (perf_vk, perf_cuda, "vk_ms"))]
    pairs = [(v, c, okv, ok(c, "cuda_ms") if (c and c is perf_cuda) else okc) for v, c, okv, okc in pairs]
    cuda_steps = [c for _, c, _, _ in pairs if c]
    cuda_ok = [okc for _, c, _, okc in pairs if c]
    s["cuda_fails"] = bool(cuda_steps) and not any(cuda_ok)
    # (TorchBench models whose benchmark builds them only for "cpu" or "cuda")
    s["cuda_only_benchmark"] = s["model"] in CUDA_ONLY_BENCHMARKS
    vk_steps = [v for v, c, _, okc in pairs if v and (c is None or okc)]
    vk_ok = [okv for v, c, okv, okc in pairs if v and (c is None or okc)]
    s["runs_on_vk"] = bool(vk_steps) and all(vk_ok)
    first_fail = next((x for x, ok in zip(vk_steps, vk_ok) if not ok), None)
    s["vk_error"] = short_error((first_fail or {}).get("error") or (status(first_fail) if first_fail else ""))
    s["vk_error_full"] = (first_fail or {}).get("error")
    counts = (fb or {}).get("fallback_counts") or {}
    if not counts and first_fail:
        counts = first_fail.get("fallback_counts") or {}
    ops = collections.Counter()
    for k, v in counts.items():
        ops[op_name(k)] += v
    s["cpu_ops"] = dict(ops)
    s["cpu_op_signatures"] = counts
    s["fallback_run"] = ("ok" if status(fb) in ("captured", "pass") else status(fb)) if fb else None
    s["fallback_error"] = short_error((fb or {}).get("error")) if fb and s["fallback_run"] != "ok" else ""
    cmp = r.get("compare") or (r.get("compare_fallback") if not s["runs_on_vk"] else None) or {}
    s["accuracy"] = None if not cmp or "pass" not in cmp else ("pass" if cmp["pass"] else "fail")
    s["accuracy_with_fallback"] = bool(r.get("compare_fallback")) and not r.get("compare")
    for k in ("err_vk", "err_cuda", "deterministic_vk", "tol"):
        s[k] = cmp.get(k)
    s["cuda_ms"], s["vk_ms"] = r.get("cuda_ms"), r.get("vk_ms")
    s["vk_ms_fallback"] = r.get("vk_ms_fallback")
    s["ratio"] = s["vk_ms"] / s["cuda_ms"] if s["vk_ms"] and s["cuda_ms"] else None
    # (the HF LLMs' latency is model.generate's, whose length ends at an
    # end-of-sequence token: the devices can generate different lengths)
    s["generate"] = s["suite"] == "huggingface" and "/" in s["model"]
    s["gpu"] = r.get("gpu") or "NVIDIA L40S"  # (the rows from before sweep.py recorded it)
    s["batch_size"] = ((perf_vk or {}).get("row") or (perf_cuda or {}).get("row") or {}).get("batch_size")
    s["cuda_error"] = short_error(next((x.get("error") or status(x) for x, ok in zip(cuda_steps, cuda_ok)
                                        if not ok), ""))
    return s


def blockers(rows):
    by = collections.defaultdict(list)
    for s in rows:
        if s["cuda_fails"] or s["runs_on_vk"] or s["cuda_only_benchmark"]:
            continue
        keys = [f"{op} (no Vulkan kernel)" for op in s["cpu_ops"] if "(raised)" not in op]
        if s["fallback_run"] != "ok":
            keys.append(s["fallback_error"] or s["vk_error"] or "unknown")
        if not keys:
            keys.append(s["vk_error"] or "unknown")
        for k in dict.fromkeys(keys):
            by[k].append(f"{s['suite']}/{s['model']}")
    return sorted(by.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def fmt(x, spec):
    return "" if x is None else format(x, spec)


def geomean(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def markdown(rows, title, notes):
    out = [f"# {title}", ""]
    if notes:
        out += [open(notes).read().strip(), ""]
    configs = sorted({(s["mode"], s["precision"]) for s in rows})
    out += ["## Summary", "",
            "Models: those tried; left out: models that fail on CUDA too, or whose benchmark builds them only "
            "for cpu and cuda. The geometric mean is over the models that run fully on vk.", "",
            "| mode | precision | GPU | suite | models | left out | run fully on vk "
            "| accuracy ok (of those compared) | vk time / CUDA time, geometric mean (models) |",
            "|---|---|---|---|--:|--:|--:|--:|--:|"]
    keys = sorted({(s["mode"], s["precision"], s["gpu"]) for s in rows})
    for mode, prec, gpu in keys:
      for suite in ("torchbench", "timm", "huggingface", "all"):
        sel = [s for s in rows if (s["mode"], s["precision"], s["gpu"]) == (mode, prec, gpu)
               and suite in (s["suite"], "all")]
        if not sel:
            continue
        tried = [s for s in sel if not s["cuda_fails"] and not s["cuda_only_benchmark"]]
        ok = [s for s in tried if s["runs_on_vk"]]
        compared = [s for s in ok if s["accuracy"]]
        acc = [s for s in compared if s["accuracy"] == "pass"]
        ratios = [s["ratio"] for s in ok if s["ratio"] and not s["generate"]]
        g = geomean(ratios)
        b = "**" if suite == "all" else ""
        out.append(f"| {mode} | {prec} | {gpu.replace('NVIDIA ', '')} | {b}{suite}{b} | {b}{len(tried)}{b} | "
                   f"{len(sel) - len(tried)} | {b}{len(ok)}{b} | {len(acc)} / {len(compared)} | "
                   f"{b}{fmt(g, '.2f')}{b} ({len(ratios)}) |")
    out.append("")
    for mode, prec in configs:
        sel = [s for s in rows if (s["mode"], s["precision"]) == (mode, prec)]
        bl = blockers(sel)
        if bl:
            out += [f"## Blockers: {mode}, {prec}", "",
                    "What stops the models that do not run on vk with the CPU fallback off, by the number of "
                    "models each blocks (a model can have several).", "",
                    "| blocker | models | which |", "|---|--:|---|"]
            for k, models in bl:
                out.append(f"| {k} | {len(models)} | {', '.join(m.split('/', 1)[1] for m in models)} |")
            out.append("")
    for (mode, prec) in configs:
        for suite in ("torchbench", "timm", "huggingface"):
            sel = sorted((s for s in rows if (s["mode"], s["precision"], s["suite"]) == (mode, prec, suite)),
                         key=lambda s: s["model"].lower())
            if not sel:
                continue
            gpus = ", ".join(sorted({s["gpu"] for s in sel}))
            out += [f"## {suite}: {mode}, {prec} ({gpus})", "",
                    "| model | batch | runs on vk | first error on vk (fallback off) | ops on the CPU (fallback on) "
                    "| accuracy vs CUDA | rel. error vs fp64: vk / CUDA | CUDA ms | vk ms | vk / CUDA |",
                    "|---|--:|---|---|---|---|---|--:|--:|--:|"]
            for s in sel:
                if s["cuda_only_benchmark"]:
                    runs = "(left out: " + CUDA_ONLY_BENCHMARKS[s["model"]] + ")"
                elif s["cuda_fails"]:
                    runs = "(CUDA fails: " + (s["cuda_error"] or "?")[:80] + ")"
                else:
                    runs = "yes" if s["runs_on_vk"] else "**no**"
                ops = ", ".join(f"{k.replace('aten::', '')} ({v})" for k, v in sorted(s["cpu_ops"].items()))
                if s["runs_on_vk"]:
                    ops = ""
                if s["fallback_run"] and s["fallback_run"] != "ok" and not s["runs_on_vk"]:
                    ops = (ops + "; " if ops else "") + "fails with the fallback too: " + s["fallback_error"][:80]
                acc = s["accuracy"] or ""
                if acc and s["accuracy_with_fallback"]:
                    acc += " (fallback on)"
                err = ""
                if s["err_vk"] is not None or s["err_cuda"] is not None:
                    err = f"{fmt(s['err_vk'], '.1e')} / {fmt(s['err_cuda'], '.1e')}"
                vk_ms = fmt(s["vk_ms"], ".2f")
                if not vk_ms and s["vk_ms_fallback"]:
                    vk_ms = f"({s['vk_ms_fallback']:.2f}, fallback on)"
                out.append(f"| {s['model']}{' (generate)' if s['generate'] else ''} | {s['batch_size'] or ''} | {runs} | {s['vk_error'] if not s['runs_on_vk'] else ''} "
                           f"| {ops} | {acc} | {err} | {fmt(s['cuda_ms'], '.2f')} | {vk_ms} | {fmt(s['ratio'], '.2f')} |")
            out.append("")
    return "\n".join(out)


COLUMNS = ["suite", "model", "mode", "precision", "gpu", "batch_size", "cuda_fails", "runs_on_vk", "vk_error",
           "cpu_ops", "fallback_run", "fallback_error", "accuracy", "accuracy_with_fallback", "err_vk", "err_cuda",
           "deterministic_vk", "tol", "cuda_ms", "vk_ms", "vk_ms_fallback",
           "ratio", "cuda_error"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="+")
    p.add_argument("--md")
    p.add_argument("--csv")
    p.add_argument("--json")
    p.add_argument("--title", default="Model benchmarks: eager PyTorch on vk and CUDA")
    p.add_argument("--notes", help="a Markdown file to put under the title")
    a = p.parse_args()
    rows = [summarise(r) for r in last_by_key(a.results)]
    rows.sort(key=lambda s: (s["mode"], s["precision"], s["suite"], s["model"].lower()))
    md = markdown(rows, a.title, a.notes)
    if a.md:
        with open(a.md, "w", newline="\n") as f:
            f.write(md + "\n")
    else:
        print(md)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(COLUMNS)
            for s in rows:
                w.writerow([json.dumps(s[c]) if isinstance(s[c], dict) else s[c] for c in COLUMNS])
    if a.json:
        with open(a.json, "w", newline="\n") as f:
            json.dump({"rows": rows, "blockers": {f"{m}/{pr}": blockers(
                [s for s in rows if (s["mode"], s["precision"]) == (m, pr)])
                for m, pr in sorted({(s["mode"], s["precision"]) for s in rows})}}, f, indent=1)


if __name__ == "__main__":
    main()
