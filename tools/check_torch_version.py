"""Checks that the PyTorch versions the project names agree with each other.

tools/torch-version is the one PyTorch release the extension is compiled and
tested against (CI and the release workflow install exactly it). This script
fails if the installed torch is not that release, if pyproject.toml's torch
requirement excludes it, or if the requirement admits another minor release
(the wheels are tied to one release's C++ ABI). Run by CI after installing
torch; see CONTRIBUTING.md, "Upgrading PyTorch".

    python tools/check_torch_version.py
"""

import pathlib
import re
import sys
import tomllib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
       "<": lambda a, b: a < b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


def Version(text):
    """(major, minor, patch) of a release such as "2.14.1" or "2.14" (local parts dropped)."""
    parts = [int(x) for x in re.match(r"\d+(?:\.\d+)*", text.strip()).group(0).split(".")]
    return tuple((parts + [0, 0, 0])[:3])


class SpecifierSet:
    """The comparison clauses of a requirement such as ">=2.14,<2.15" (no wildcards)."""

    def __init__(self, text):
        self.clauses = [(m.group(1), Version(m.group(2)))
                        for m in re.finditer(r"(>=|<=|==|!=|>|<)\s*([\d.]+)", text)]
        assert self.clauses, f"no version clauses in {text!r}"

    def __contains__(self, v):
        return all(OPS[op](v, bound) for op, bound in self.clauses)


def fmt(v):
    return ".".join(map(str, v))


def main():
    pinned = Version((ROOT / "tools" / "torch-version").read_text(encoding="utf-8").strip())
    installed = Version(torch.__version__.split("+")[0])
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    torch_dep = next(d for d in deps if re.match(r"torch\b", d))
    spec = SpecifierSet(torch_dep[len("torch"):].strip())
    problems = []
    if installed != pinned:
        problems.append(f"installed torch {fmt(installed)} is not tools/torch-version's {fmt(pinned)}")
    if pinned not in spec:
        problems.append(f"pyproject.toml's '{torch_dep}' excludes {fmt(pinned)}")
    next_minor = (pinned[0], pinned[1] + 1, 0)
    if next_minor in spec:
        problems.append(f"pyproject.toml's '{torch_dep}' admits {fmt(next_minor)}: pin it to {pinned[0]}.{pinned[1]}")
    for p in problems:
        print("torch version mismatch:", p)
    if not problems:
        print(f"torch {fmt(pinned)}: tools/torch-version, pyproject.toml ('{torch_dep}') and the installed package agree")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
