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
    rows = {}
    for path in paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                rows[(r["suite"], r["model"], r["mode"], r["precision"])] = r
    return list(rows.values())


def short_error(err):
    if not err:
        return ""
    m = re.search(r"(aten::[\w.]+) has no Vulkan kernel", err)
    if m:
        return f"{m.group(1)}: no Vulkan kernel"
    err = re.sub(r"^RuntimeError: Eager run failed\s*", "", err)
    return err.replace("|", "/")[:160]


def op_name(key):
    # fallback_counts() keys: "aten::op", "aten::op (raised)", or with
    # TORCH_VK_FALLBACK_DETAIL=1 "aten::op [dtypes and shapes]"
    return key.split(" [")[0]


def status(rec):
    return (rec or {}).get("status")


def summarise(r):
    s = {k: r[k] for k in ("suite", "model", "mode", "precision")}
    acc_cuda, acc_vk, perf_cuda, perf_vk = (r.get(k) for k in ("acc_cuda", "acc_vk", "perf_cuda", "perf_vk"))
    # (the accuracy run's fallback rerun, or the latency run's where only it failed)
    fb = r.get("acc_vk_fallback") or (r.get("perf_vk_fallback") if r.get("vk_ms") is None else None)
    vk_steps = [x for x in (acc_vk, perf_vk) if x]
    cuda_steps = [x for x in (acc_cuda, perf_cuda) if x]
    vk_ok = [status(x) in ("captured", "pass") and not (x is perf_vk and r.get("vk_ms") is None)
             for x in vk_steps]
    cuda_ok = [status(x) in ("captured", "pass") and not (x is perf_cuda and r.get("cuda_ms") is None)
               for x in cuda_steps]
    s["cuda_fails"] = bool(cuda_steps) and not any(cuda_ok)
    if acc_vk and acc_cuda and status(acc_cuda) != "captured" and status(acc_vk) != "captured":
        s["cuda_fails"] = True
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
    s["batch_size"] = ((perf_vk or {}).get("row") or (perf_cuda or {}).get("row") or {}).get("batch_size")
    s["cuda_error"] = short_error(next((x.get("error") or status(x) for x, ok in zip(cuda_steps, cuda_ok)
                                        if not ok), ""))
    return s


def blockers(rows):
    by = collections.defaultdict(list)
    for s in rows:
        if s["cuda_fails"] or s["runs_on_vk"]:
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
            "| mode | precision | models | CUDA fails too | run fully on vk | accuracy ok (of those compared) "
            "| vk time / CUDA time, geometric mean (models) |",
            "|---|---|--:|--:|--:|--:|--:|"]
    for mode, prec in configs:
        sel = [s for s in rows if (s["mode"], s["precision"]) == (mode, prec)]
        tried = [s for s in sel if not s["cuda_fails"]]
        ok = [s for s in tried if s["runs_on_vk"]]
        compared = [s for s in ok if s["accuracy"]]
        acc = [s for s in compared if s["accuracy"] == "pass"]
        ratios = [s["ratio"] for s in ok if s["ratio"]]
        g = geomean(ratios)
        out.append(f"| {mode} | {prec} | {len(tried)} | {len(sel) - len(tried)} | {len(ok)} | "
                   f"{len(acc)} / {len(compared)} | {fmt(g, '.2f')} ({len(ratios)}) |")
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
            out += [f"## {suite}: {mode}, {prec}", "",
                    "| model | batch | runs on vk | first error on vk (fallback off) | ops on the CPU (fallback on) "
                    "| accuracy vs CUDA | rel. error vs fp64: vk / CUDA | CUDA ms | vk ms | vk / CUDA |",
                    "|---|--:|---|---|---|---|---|--:|--:|--:|"]
            for s in sel:
                if s["cuda_fails"]:
                    runs = "(CUDA fails: " + (s["cuda_error"] or "?")[:80] + ")"
                else:
                    runs = "yes" if s["runs_on_vk"] else "**no**"
                ops = ", ".join(f"{k.replace('aten::', '')} ({v})" for k, v in sorted(s["cpu_ops"].items()))
                if s["fallback_run"] and s["fallback_run"] != "ok":
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
                out.append(f"| {s['model']} | {s['batch_size'] or ''} | {runs} | {s['vk_error'] if not s['runs_on_vk'] else ''} "
                           f"| {ops} | {acc} | {err} | {fmt(s['cuda_ms'], '.2f')} | {vk_ms} | {fmt(s['ratio'], '.2f')} |")
            out.append("")
    return "\n".join(out)


COLUMNS = ["suite", "model", "mode", "precision", "batch_size", "cuda_fails", "runs_on_vk", "vk_error",
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
        open(a.md, "w").write(md + "\n")
    else:
        print(md)
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for s in rows:
                w.writerow([json.dumps(s[c]) if isinstance(s[c], dict) else s[c] for c in COLUMNS])
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"rows": rows, "blockers": {f"{m}/{pr}": blockers(
                [s for s in rows if (s["mode"], s["precision"]) == (m, pr)])
                for m, pr in sorted({(s["mode"], s["precision"]) for s in rows})}}, f, indent=1)


if __name__ == "__main__":
    main()
