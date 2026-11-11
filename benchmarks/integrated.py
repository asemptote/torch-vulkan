"""Benchmarks on one machine's GPU against the same machine's CPU, written up
as a results page (benchmarks/results/NAME.md).

    python benchmarks/integrated.py hd530-mesa matmul fft linalg eigh elementwise
    python benchmarks/integrated.py uhd620-windows matmul fft division
    python benchmarks/integrated.py --small test-run matmul division   # a quick check of the harness

Workloads (each in its own process, as each script prints it):
  matmul       benchmarks/matmul.py (GFLOP/s, vk and CPU)
  fft          benchmarks/fft.py
  linalg       benchmarks/linalg.py (solve, cholesky, inv)
  eigh         benchmarks/eigh.py on the GPU, then with TORCH_VK_LINALG=cpu
               (the CPU's LAPACK through the same calls) as the baseline
  elementwise  benchmarks/elementwise.py (all of it)
  division     benchmarks/elementwise.py's division alone

--small sets TORCH_VK_BENCH_SMALL=1: every script runs small sizes only, to
check the harness, not to measure. The page names the GPU, its Vulkan driver,
the CPU and the versions; never the host.
"""

import datetime
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WORKLOADS = {
    "matmul": [("matmul.py", [], {})],
    "fft": [("fft.py", [], {})],
    "linalg": [("linalg.py", [], {})],
    "eigh": [("eigh.py", [], {}), ("eigh.py", [], {"TORCH_VK_LINALG": "cpu"})],
    "elementwise": [("elementwise.py", [], {})],
    "division": [("elementwise.py", ["division"], {})],
}


def run(script, args, env_extra, timeout):
    env = dict(os.environ, **env_extra)
    cmd = [sys.executable, os.path.join(HERE, script), *args]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        out = r.stdout.rstrip()
        if r.returncode != 0:
            out += f"\n(exit {r.returncode})\n" + "\n".join(r.stderr.rstrip().splitlines()[-5:])
        return out
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout} s)"


def vulkan_driver(device):
    """The Vulkan driver's name and version for `device`, from vulkaninfo."""
    try:
        text = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "unknown (no vulkaninfo)"
    for block in re.split(r"^GPU\d+:", text, flags=re.M)[1:]:
        fields = dict(re.findall(r"^\s*(\w+)\s*=\s*(.+?)\s*$", block, re.M))
        if fields.get("deviceName") == device:
            name, info, version = (fields.get(k, "") for k in ("driverName", "driverInfo", "driverVersion"))
            return ", ".join(x for x in (name, info, version and f"version {version}") if x) or "unknown"
    return "unknown"


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                               capture_output=True, text=True, timeout=60)
            if r.stdout.strip():
                return r.stdout.strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or "unknown"


def main():
    args = sys.argv[1:]
    small = "--small" in args
    args = [a for a in args if a != "--small"]
    if len(args) < 2 or any(w not in WORKLOADS for w in args[1:]):
        sys.exit(__doc__)
    name, workloads = args[0], args[1:]
    if small:
        os.environ["TORCH_VK_BENCH_SMALL"] = "1"

    import torch
    import torch_vulkan as tv

    device = tv.device_name()
    # (TORCH_VK_BENCH_COMMIT names it where the tree is not a git checkout)
    commit = os.environ.get("TORCH_VK_BENCH_COMMIT", "")
    if not commit:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip()
        except OSError:
            pass
    lines = [
        f"# {device}: torch-vulkan against the CPU",
        "",
        f"- GPU: {device}, Vulkan driver {vulkan_driver(device)}",
        f"- CPU baseline: {cpu_model()}, {torch.get_num_threads()} threads, PyTorch {torch.__version__}",
        f"- OS: {platform.system()} {platform.release()}",
        f"- torch-vulkan commit {commit or 'unknown'}, {datetime.date.today().isoformat()}",
        f"- `python benchmarks/integrated.py {' '.join(sys.argv[1:])}`",
    ]
    if small:
        lines.append("- **Small sizes (TORCH_VK_BENCH_SMALL=1): a check of the harness, not measurements.**")
    for w in workloads:
        lines += ["", f"## {w}", ""]
        for script, sargs, env in WORKLOADS[w]:
            label = " ".join([script, *sargs] + [f"{k}={v}" for k, v in env.items()])
            print(f"== {label}", flush=True)
            out = run(script, sargs, env, timeout=3600)
            print(out, flush=True)
            lines += [f"`{label}`:", "", "```", out, "```", ""]
    path = os.path.join(HERE, "results", f"{name}.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
