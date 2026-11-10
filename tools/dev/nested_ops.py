"""Lists the aten ops whose NestedTensorCPU kernel is also PyTorch's
NestedTensorCUDA kernel (one function for both: device-generic), and those
with a CPU-only kernel, from native_functions.yaml in pytorch-src/ (a sparse
checkout of PyTorch at the installed torch's commit; the file is read with
git show, so only test/ needs to be checked out):

    python tools/dev/nested_ops.py

crates/pyvk/csrc/nested.cpp's kNestedOps is the first list (plus a few
generic CPU-only kernels). Python 3 standard library only.
"""

import pathlib
import re
import subprocess

src = pathlib.Path(__file__).resolve().parents[2] / "pytorch-src"
text = subprocess.run(["git", "-C", str(src), "show", "HEAD:aten/src/ATen/native/native_functions.yaml"],
                      capture_output=True, text=True, check=True).stdout
shared, cpu_only = [], []
for entry in re.split(r"\n(?=- func:)", text):
    m = re.match(r"- func: ([\w.]+)\(", entry)
    d = re.search(r"\n  dispatch:\n((?:    .*\n?)+)", entry)
    if not (m and d):
        continue
    for line in d.group(1).splitlines():
        keys, _, kernel = line.strip().partition(":")
        names = [k.strip() for k in keys.split(",")]
        if "NestedTensorCPU" in names:
            (shared if "NestedTensorCUDA" in names else cpu_only).append(f"{m.group(1)}  ({kernel.strip()})")
print("shared with CUDA:", *shared, sep="\n  ")
print("CPU only:", *cpu_only, sep="\n  ")
