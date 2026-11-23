"""Runs run.py over a suite's models, each step in its own process, and
appends one JSON line per model and configuration to OUT/results.jsonl.

    python benchmarks/models/sweep.py --phase reference --suite timm --out OUT
    python benchmarks/models/sweep.py --phase gpu --suite timm --out OUT \\
        --cuda-python CUDA_VENV/bin/python [--precision float32] [--mode inference]

Phases:
  reference  (CPU only) the runner's accuracy-mode model and inputs, saved
             with the outputs of the model in float64: OUT/ref/SUITE/MODEL.pt.
  gpu        for each model: on CUDA, then on vk with the CPU fallback off
             (the default), the runner's accuracy-mode run on the reference's
             weights and inputs (outputs saved, then compared: compare.py);
             if vk fails, again with TORCH_VK_CPU_FALLBACK=1, to list the ops
             that went to the CPU; then the runner's eager latency
             (--nothing --performance) on CUDA and vk.

The vk interpreter is this one (--vk-python to change); the CUDA one is
--cuda-python, an environment with the same torch release built for CUDA.
PyTorch's benchmarks/dynamo/ comes from PYTORCH_BENCH_DIR, TorchBench from
TORCHBENCH_DIR (see run.py). Models already in results.jsonl (same suite,
model, mode, precision) are skipped, so an interrupted sweep resumes.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run.py")
PRECISION_FLAGS = {
    "float32": ["--float32"],
    "bfloat16": ["--bfloat16"],
    "amp": ["--amp"],  # (autocast's default type on both devices: float16)
    "amp_bf16": ["--amp", "--amp-dtype", "bfloat16"],
}


def model_names(suite, pytorch_dir):
    """The suite's list, less the models its runner skips on every device."""
    import yaml

    d = os.path.join(pytorch_dir, "benchmarks", "dynamo")
    yml = {"timm": "timm_models.yaml", "huggingface": "huggingface.yaml", "torchbench": "torchbench.yaml"}[suite]
    with open(os.path.join(d, yml)) as f:
        skip = set(x for x in (yaml.safe_load(f).get("skip", {}).get("all") or []) if isinstance(x, str))
    return [n for n in listed_models(suite, d) if n not in skip]


def listed_models(suite, d):
    if suite == "timm":
        lines = open(os.path.join(d, "timm_models_list.txt")).read().split("\n")
        return [line.split()[0] for line in lines if line.strip()]
    if suite == "huggingface":
        lines = open(os.path.join(d, "huggingface_models_list.txt")).read().split("\n")
        names = [line.split(",")[0].strip() for line in lines if line.strip()]
        # (huggingface.py's EXTRA_MODELS, not in the list file)
        return names + [n for n in ("AllenaiLongformerBase", "T5Small", "DistillGPT2", "GoogleFnet",
                                    "YituTechConvBert") if n not in names]
    lines = open(os.path.join(d, "all_torchbench_models_list.txt")).read().split("\n")
    return [line.strip() for line in lines if line.strip()]


def step(name, cmd, log_dir, timeout, env=None):
    """Runs cmd with --record; returns the record (or a status for a timeout
    or a crash without one)."""
    rec_path = os.path.join(log_dir, name + ".json")
    log_path = os.path.join(log_dir, name + ".log")
    if os.path.exists(rec_path):
        os.remove(rec_path)
    t0 = time.time()
    with open(log_path, "w") as log:
        try:
            p = subprocess.run(cmd + ["--record", rec_path], stdout=log, stderr=subprocess.STDOUT,
                               timeout=timeout, env=env)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    rec = {}
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            rec = json.load(f)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["returncode"] = code
    if code == "timeout":
        rec["status"] = "timeout"
    elif not rec.get("status") or (code != 0 and rec.get("status") in ("pass", None)):
        rec["status"] = rec.get("status") or "crashed"
        if code != 0 and not rec.get("error"):
            tail = open(log_path, errors="replace").read().strip().splitlines()[-3:]
            rec["error"] = f"exit {code}: " + " | ".join(tail)[-500:]
    rec.pop("traceback", None)
    rec.pop("argv", None)
    return rec


def latency_ms(rec):
    if rec.get("status") != "pass":
        return None
    row = rec.get("row") or {}
    try:
        return float(row["abs_latency"])
    except (KeyError, TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("reference", "gpu"), required=True)
    p.add_argument("--suite", choices=("timm", "huggingface", "torchbench"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", help="comma-separated (default: the suite's list)")
    p.add_argument("--precision", default="float32", choices=sorted(PRECISION_FLAGS))
    p.add_argument("--mode", default="inference", choices=("inference", "training"))
    p.add_argument("--cuda-python")
    p.add_argument("--vk-python", default=sys.executable)
    p.add_argument("--timeout", type=int, default=1800, help="seconds per step")
    p.add_argument("--repeat", type=int, default=30, help="the runner's timed calls")
    p.add_argument("--no-accuracy", action="store_true", help="latency only")
    p.add_argument("--no-perf", action="store_true", help="accuracy only")
    p.add_argument("--delete-reference", action="store_true",
                   help="delete each model's CPU reference once compared (they are large: for the last "
                        "configuration that needs them)")
    p.add_argument("--perf-with-fallback", action="store_true",
                   help="where vk's latency run fails with the fallback off, run it with the fallback on (for "
                        "the ops that went to the CPU, and the time)")
    a = p.parse_args()
    pytorch_dir = os.environ.get("PYTORCH_BENCH_DIR")
    if not pytorch_dir:
        sys.exit("set PYTORCH_BENCH_DIR (see run.py)")
    names = a.models.split(",") if a.models else model_names(a.suite, pytorch_dir)
    out = os.path.abspath(a.out)
    ref_dir = os.path.join(out, "ref", a.suite)
    os.makedirs(ref_dir, exist_ok=True)
    mode_flag = "--" + a.mode

    if a.phase == "reference":
        for name in names:
            safe = name.replace("/", "--")
            ref = os.path.join(ref_dir, safe + ".pt")
            if os.path.exists(ref):
                continue
            log_dir = os.path.join(out, "logs", a.suite, safe)
            os.makedirs(log_dir, exist_ok=True)
            rec = step("reference", [a.vk_python, RUN, a.suite, "--devices", "cpu", "--accuracy", mode_flag,
                                     "--float32", "--only", name, "--output", os.path.join(log_dir, "ref.csv"),
                                     "--write-reference", ref], log_dir, a.timeout)
            print(f"{a.suite} {name}: reference {rec['status']} {rec.get('error') or ''} ({rec['seconds']} s)",
                  flush=True)
        return

    results = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(results):
        for line in open(results):
            r = json.loads(line)
            done.add((r["suite"], r["model"], r["mode"], r["precision"]))
    env_off = dict(os.environ, TORCH_VK_CPU_FALLBACK="0")
    # (each fallback's dtypes in fallback_counts(), its shapes and caller in the log)
    env_on = dict(os.environ, TORCH_VK_CPU_FALLBACK="1", TORCH_VK_FALLBACK_DETAIL="1", TORCH_VK_FALLBACK_TRACE="1")
    prec = PRECISION_FLAGS[a.precision]
    for name in names:
        if (a.suite, name, a.mode, a.precision) in done:
            continue
        safe = name.replace("/", "--")
        log_dir = os.path.join(out, "logs", a.suite, safe, f"{a.mode}-{a.precision}")
        os.makedirs(log_dir, exist_ok=True)
        ref = os.path.join(ref_dir, safe + ".pt")
        r = {"suite": a.suite, "model": name, "mode": a.mode, "precision": a.precision}

        def run(dev, python, kind, env, tag=""):
            cmd = [python, RUN, a.suite, "--devices", dev, mode_flag, *prec, "--only", name]
            if kind == "accuracy":
                outputs = os.path.join(log_dir, f"{dev}{tag}.out.pt")
                cmd += ["--accuracy", "--output", os.path.join(log_dir, f"acc-{dev}{tag}.csv"),
                        "--reference", ref, "--outputs", outputs]
            else:
                cmd += ["--performance", "--nothing", "--repeat", str(a.repeat),
                        "--output", os.path.join(log_dir, f"perf-{dev}{tag}.csv")]
            return step(f"{kind}-{dev}{tag}", cmd, log_dir, a.timeout, env)

        accuracy = not a.no_accuracy and a.mode == "inference" and os.path.exists(ref)
        if not a.no_accuracy and a.mode == "inference" and not accuracy:
            r["accuracy_note"] = "no CPU reference"
        if accuracy:
            r["acc_cuda"] = run("cuda", a.cuda_python, "accuracy", env_off) if a.cuda_python else None
            r["acc_vk"] = run("vk", a.vk_python, "accuracy", env_off)
            vk_ok = r["acc_vk"]["status"] == "captured"
            if not vk_ok:
                r["acc_vk_fallback"] = run("vk", a.vk_python, "accuracy", env_on, "-fallback")
            cuda_out = os.path.join(log_dir, "cuda.out.pt")
            for tag in ("", "-fallback"):
                vk_out = os.path.join(log_dir, f"vk{tag}.out.pt")
                if os.path.exists(cuda_out) and os.path.exists(vk_out) and (r.get("acc_cuda") or {}).get(
                        "status") == "captured":
                    c = subprocess.run([a.vk_python, os.path.join(HERE, "compare.py"), ref, cuda_out, vk_out],
                                       capture_output=True, text=True, timeout=a.timeout)
                    try:
                        r["compare" + tag.replace("-", "_")] = json.loads(c.stdout.strip().splitlines()[-1])
                    except (ValueError, IndexError):
                        r["compare" + tag.replace("-", "_")] = {"error": (c.stderr or c.stdout)[-500:]}
            for f in os.listdir(log_dir):  # (outputs can be large)
                if f.endswith(".out.pt"):
                    os.remove(os.path.join(log_dir, f))
        if not a.no_perf:
            if a.cuda_python:
                r["perf_cuda"] = run("cuda", a.cuda_python, "performance", env_off)
                r["cuda_ms"] = latency_ms(r["perf_cuda"])
            r["perf_vk"] = run("vk", a.vk_python, "performance", env_off)
            r["vk_ms"] = latency_ms(r["perf_vk"])
            if r["vk_ms"] is None and a.perf_with_fallback and (
                    (r.get("acc_vk_fallback") or r.get("acc_vk") or {}).get("status") == "captured"):
                r["perf_vk_fallback"] = run("vk", a.vk_python, "performance", env_on, "-fallback")
                r["vk_ms_fallback"] = latency_ms(r["perf_vk_fallback"])
        r["torch_vulkan_commit"] = os.environ.get("TORCH_VULKAN_COMMIT")
        with open(results, "a") as f:
            f.write(json.dumps(r) + "\n")
        if a.delete_reference and os.path.exists(ref):
            os.remove(ref)
        summary = {k: r.get(k) for k in ("cuda_ms", "vk_ms")}
        vk_status = (r.get("acc_vk") or r.get("perf_vk") or {}).get("status")
        err = (r.get("acc_vk") or {}).get("error") or (r.get("perf_vk") or {}).get("error") or ""
        cmp = (r.get("compare") or {}).get("pass")
        print(f"{a.suite} {name} {a.mode} {a.precision}: vk {vk_status} acc {cmp} {summary} {err[:200]}",
              flush=True)


if __name__ == "__main__":
    main()
