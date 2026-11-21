"""Bitwise A/B runs: compare everything the host read from "vk" between runs.

With TORCH_VK_READ_LOG=DIR, each process logs a hash of every read of vk
memory by the host (copies to the CPU, item()) to DIR/reads-<pid>.txt. Runs
of the same tests under different settings (e.g. TORCH_VK_BARRIERS unset
and =full) should then read the same bits wherever the computation is
deterministic. A missing barrier shows up as a test whose reads differ
(docs/log/M37).

    # as a pytest plugin: a section per test
    PYTHONPATH=tools/dev TORCH_VK_READ_LOG=/tmp/a python -m pytest -p barrier_ab tests/python
    TORCH_VK_BARRIERS=full PYTHONPATH=tools/dev TORCH_VK_READ_LOG=/tmp/b python -m pytest -p barrier_ab tests/python
    # tools/opinfo_coverage.py marks a section per op itself
    python tools/dev/barrier_ab.py /tmp/a /tmp/b [--noise /tmp/a2 ...]

The comparison lists the sections (tests, ops) whose reads differ. Sections
that also differ between two runs with the same settings (--noise: e.g. a
second default run) are nondeterministic anyway (uninitialized memory,
atomics, timing) and are reported apart.
"""
import argparse
import collections
import pathlib
import sys


def pytest_runtest_setup(item):
    from torch_vulkan import _native

    _native.read_log_mark(item.nodeid)


def load(directory):
    """{section: [hash lines]} over all of a run's per-process logs."""
    sections = collections.defaultdict(list)
    for path in sorted(pathlib.Path(directory).glob("reads-*.txt")):
        current = "(before any section)"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("## "):
                current = line[3:]
            else:
                sections[current].append(line)
    return sections


def differing(a, b):
    return {s for s in a.keys() | b.keys() if a.get(s) != b.get(s)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--noise", nargs="*", default=[], help="runs with a's settings: sections differing are noise")
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    noise = set()
    for other in args.noise:
        noise |= differing(a, load(other))
    diff = differing(a, b)
    reads = sum(len(v) for v in a.values())
    print(f"{len(a)} sections, {reads} reads in {args.a}; {len(b)} sections in {args.b}")
    print(f"nondeterministic (differ between runs with the same settings): {len(noise)}")
    for s in sorted(noise):
        print(f"  noise {s}")
    real = sorted(diff - noise)
    print(f"differ between {args.a} and {args.b} only: {len(real)}")
    for s in real:
        la, lb = a.get(s, []), b.get(s, [])
        first = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
        print(f"  DIFF {s}: {len(la)} vs {len(lb)} reads, first difference at read {first}")
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
