"""Regenerates the files derived from the installed PyTorch, after a change
of tools/torch-version (CONTRIBUTING.md, "Upgrading PyTorch", step 2), and
checks the result. From the repo root, in the venv with the new torch:

    python tools/dev/regenerate.py [RUN_DIR...]

1. tools/check_torch_version.py: the installed torch, tools/torch-version
   and pyproject.toml agree (else nothing is regenerated);
2. pytorch-src/ (PyTorch's tests, for tools/pytorch_tests.py) is at the
   installed torch's commit (tools/cluster/setup.sh checks it out);
3. tools/port_special.py: crates/vkcore/shaders/_special.slang from ATen's
   Math.h;
4. tools/dev/cpu_dtype_errors.py [RUN_DIR...]: the CPU's dtype errors that vk
   raises itself (crates/pyvk/csrc/cpu_dtype_errors.inc,
   python/torch_vulkan/_cpu_dtype_errors.py and the examples), probed again
   on the new CPU, with the raised fallbacks of RUN_DIRs (tools/pytorch_tests.py
   runs with TORCH_VK_FALLBACK_DETAIL=1) added; then --check --all.

It ends with the generated files git sees changed: review their diffs, then
rebuild (the .inc is compiled in) and go on with step 3.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATED = ["crates/vkcore/shaders/_special.slang", "crates/pyvk/csrc/cpu_dtype_errors.inc",
             "python/torch_vulkan/_cpu_dtype_errors.py", "tools/dev/cpu_dtype_errors_examples.json"]


def run(*args):
    print("$", " ".join(args), flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode


def main():
    if run("tools/check_torch_version.py") != 0:
        raise SystemExit("install tools/torch-version's torch (and update pyproject.toml) first")
    import torch

    src = ROOT / "pytorch-src"
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != torch.version.git_version:
        print(f"note: pytorch-src/ is at {head or 'nothing'}, not torch {torch.__version__}'s commit "
              f"{torch.version.git_version}: tools/cluster/setup.sh (or tools/pytorch_tests.py's docstring) "
              "checks it out, before PyTorch's test files are run")
    failed = [name for name, code in [
        ("port_special.py", run("tools/port_special.py")),
        ("cpu_dtype_errors.py", run("tools/dev/cpu_dtype_errors.py", *sys.argv[1:])),
        ("cpu_dtype_errors.py --check --all", run("tools/dev/cpu_dtype_errors.py", "--check", "--all")),
    ] if code != 0]
    status = subprocess.run(["git", "status", "--short", "--", *GENERATED], cwd=ROOT, capture_output=True, text=True)
    if status.returncode != 0:  # (a copy of the tree, as tools/cluster/sync.sh makes)
        print("\nnot a git checkout: compare these with the repository's:\n  " + "\n  ".join(GENERATED))
    else:
        print("\nchanged:\n" + (status.stdout or "  (nothing)\n"))
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))
    print("review the diffs, rebuild (maturin develop --release), then CONTRIBUTING.md's step 3")


if __name__ == "__main__":
    main()
