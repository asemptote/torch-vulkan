"""Plans a queued run's shards longest first, from an earlier run's results.

    python3 plan_shards.py FROM_DIR... --out RUN_DIR [--minutes 5] [--ids-from DIR]

The test ids are those an earlier run recorded (FROM_DIR: its *.jsonl results;
several directories merge, later ones winning), and each test's expected time
is the wall time it took there (elapsed, recorded by tools/pytorch_tests.py),
else a prior for tests known to be slow (KNOWN_SLOW), else DEFAULT seconds.
With --ids-from (e.g. the ids tools/pytorch_tests.py --collect wrote), the
tests are that directory's instead, timed from the FROM_DIRs where they ran.
Each shard holds tests of one file (one pytest process imports and collects
one file), packed longest first up to --minutes of expected time, so a test
longer than that gets a shard of its own. The shards are numbered from the
longest: the cluster's workers take the queue from the start, so the longest
work starts first, and the workstations' backfill workers from the end.

Writes RUN_DIR/shards/shard-NNNN.ids (test ids, one per line), the queue
(RUN_DIR/queue/shard-NNNN), shard_count, and plan.txt (each shard's expected
seconds and test count). Python 3 standard library only.
"""

import argparse
import collections
import json
import pathlib
import re
import sys

DEFAULT = 0.5  # seconds, for a test without a recorded time
# Priors (seconds, on lavapipe) for tests known to be slow, until a run has
# recorded their times.
KNOWN_SLOW = [
    (r"test_linalg\.py::.*::test_pca_lowrank_vk$", 2400),
    (r"test_linalg\.py::.*::test_svd_lowrank_vk", 1200),
    (r"::TestCompositeComplianceVK::test_backward_grid_sampler_3d_vk", 1800),
    (r"grid_sampler_3d_vk", 300),
    (r"(pca|svd)_lowrank_vk", 300),
    (r"::test_reduction_vectorize_along_input_corner_vk", 600),
    (r"::test_quantile_large_input_vk", 300),
    (r"::test_CTCLoss_long_targets_vk", 150),
    (r"::test_index_put_large_indices_vk", 130),
    (r"::test_inverse_errors_large_vk", 60),
    (r"::test_cross_entropy_loss_precision$", 60),
    (r"::test_interpolate_buffer_overflow$", 60),
    (r"::test_argminmax_large_axis_vk", 60),
]


def recorded(dirs):
    """{nodeid: elapsed seconds or None} over the runs' results."""
    times = {}
    for d in dirs:
        for path in sorted(pathlib.Path(d).glob("*.jsonl")):
            if path.name.endswith(".cpu.jsonl"):
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if "nodeid" in entry:
                    elapsed = entry.get("elapsed")
                    if elapsed is not None or entry["nodeid"] not in times:
                        times[entry["nodeid"]] = elapsed
    return times


def expected(nodeid, elapsed):
    if elapsed is not None:
        return max(float(elapsed), 0.01)
    for pattern, seconds in KNOWN_SLOW:
        if re.search(pattern, nodeid):
            return float(seconds)
    return DEFAULT


def plan(times, limit):
    """Shards [(seconds, [nodeids])], longest first: per file, tests longest
    first into the shard with the least time while it stays under limit
    (a new shard otherwise)."""
    by_file = collections.defaultdict(list)
    for nodeid, elapsed in times.items():
        by_file[nodeid.split("::")[0]].append((expected(nodeid, elapsed), nodeid))
    shards = []
    for _, tests in sorted(by_file.items()):
        tests.sort(reverse=True)
        bins = []  # [seconds, [nodeids]]
        for seconds, nodeid in tests:
            best = min(bins, key=lambda b: b[0], default=None)
            if best is None or (best[0] + seconds > limit and best[0] > 0):
                bins.append([seconds, [nodeid]])
            else:
                best[0] += seconds
                best[1].append(nodeid)
        shards.extend((s, ids) for s, ids in bins)
    shards.sort(key=lambda b: -b[0])
    return shards


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("from_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--ids-from", default="", help="plan these tests (a directory of *.jsonl), not FROM_DIRs'")
    args = ap.parse_args()
    times = recorded(args.from_dirs)
    if args.ids_from:
        times = {nodeid: times.get(nodeid) for nodeid in recorded([args.ids_from])}
    if not times:
        sys.exit(f"no results in {args.from_dirs}")
    shards = plan(times, args.minutes * 60)
    out = pathlib.Path(args.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    (out / "queue").mkdir(exist_ok=True)
    lines = []
    for i, (seconds, ids) in enumerate(shards):
        name = f"shard-{i:04d}"
        (out / "shards" / f"{name}.ids").write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
        (out / "queue" / name).touch()
        lines.append(f"{name}\t{seconds:.0f}\t{len(ids)}\t{ids[0].split('::')[0]}")
    (out / "shard_count").write_text(f"{len(shards)}\n", encoding="utf-8")
    (out / "plan.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    measured = sum(1 for e in times.values() if e is not None)
    total = sum(s for s, _ in shards)
    print(f"{len(times)} tests ({measured} with recorded times), {len(shards)} shards, "
          f"{total / 3600:.1f} h expected in all, longest shard {shards[0][0] / 60:.0f} min")


if __name__ == "__main__":
    main()
