"""Runs PyTorch's own device-generic test suites on device "vk" (M15).

PyTorch's tests (test/test_ops.py, test_torch.py, ...) are written once and
instantiated per device type by instantiate_device_type_tests; a PrivateUse1
backend gets its own instance (TestFooPRIVATEUSE1) when the backend is
importable before torch.testing._internal.common_device_type computes its
device list. This script imports torch_vulkan first, restricts the
instantiation to "vk", and runs pytest on the given files:

    python tools/pytorch_tests.py [--tests DIR] [--out DIR] [--shard INDEX/COUNT] test_ops.py [...] [-- pytest args]
    python tools/pytorch_tests.py --merge [--out DIR]
    python tools/pytorch_tests.py --ids FAILURES.txt [--name NAME] [--out DIR] [--shard INDEX/COUNT]

DIR defaults to $PYTORCH_TEST_DIR, the test/ directory of a PyTorch checkout
at the installed torch's commit (torch.version.git_version):

    git init pytorch-src && cd pytorch-src
    git remote add origin https://github.com/pytorch/pytorch.git
    git config core.sparseCheckout true && echo test/ > .git/info/sparse-checkout
    git fetch --depth 1 --filter=blob:none origin <git_version> && git checkout FETCH_HEAD

Each file (or shard: every COUNT-th of the file's collected tests, starting
at INDEX) runs in a pytest process that appends each test's outcome to
--out/<run>.jsonl as it finishes. If the process dies (a segfault in a
kernel, say), the test it was running is recorded as "crashed" and pytest
starts again with the tests not yet run. Failing tests are then run again
on the CPU (the same test instantiated for "cpu"): those that fail there
too are counted as "fails on CPU too", not as vk problems, and failures
listed in EXPECTED (with the reason) are counted as "expected". A summary
(outcome counts and the most common messages of the vk-only failures) is
printed and the vk-only failures are listed in --out/<run>.failures.txt.
--merge summarizes every run in --out (e.g. shards from several machines).
"""

import argparse
import collections
import json
import os
import pathlib
import re
import subprocess
import sys

# pytest with torch_vulkan imported first (it registers "vk" before
# common_device_type lists the device types), test ids from a file, and a
# plugin recording each test's start and outcome (flushed, so they survive a
# crash). With TORCH_VK_VALIDATE set, a test during which the validation
# layer reported anything gets the outcome "validation" (with the message).
RUNNER = r"""
import json, os, sys
import torch_vulkan  # noqa: F401
from torch_vulkan import _native
import pytest

VALIDATE = os.environ.get("TORCH_VK_VALIDATE", "0") not in ("", "0")

class Recorder:
    def __init__(self, path):
        self.out = open(path, "a", encoding="utf-8")
        self.issues = 0

    def write(self, **entry):
        self.out.write(json.dumps(entry) + "\n")
        self.out.flush()

    def pytest_runtest_logstart(self, nodeid, location):
        self.write(start=nodeid)

    def pytest_runtest_teardown(self, item):
        # An out-of-range index is flagged by the kernel and raised at the next
        # synchronization (as CUDA's device asserts): charge it to this test,
        # not to whichever test synchronizes next.
        try:
            _native.synchronize()
        except IndexError as e:
            self.write(nodeid=item.nodeid, outcome="failed", message=f"IndexError at teardown: {e}")
        if VALIDATE:
            issues = _native.validation_issues()
            if issues != self.issues:
                message = _native.last_validation_message().strip().splitlines()
                self.write(nodeid=item.nodeid, outcome="validation",
                           message=f"{issues - self.issues} validation messages, last: {message[0][:200] if message else ''}")
                self.issues = issues

    def pytest_runtest_logreport(self, report):
        failed_outside_call = report.when != "call" and not report.passed
        if report.when != "call" and not failed_outside_call:
            return
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        message = ""
        if report.failed:
            crash = getattr(report.longrepr, "reprcrash", None)
            message = crash.message if crash is not None else str(report.longrepr)
            message = message.strip().splitlines()[0][:200] if message.strip() else ""
        self.write(nodeid=report.nodeid, outcome=outcome, message=message)

args = sys.argv[2:]
if args and args[0].startswith("@"):
    args = open(args[0][1:], encoding="utf-8").read().split() + args[1:]
sys.exit(pytest.main(args, plugins=[Recorder(sys.argv[1])] if sys.argv[1] != "-" else []))
"""


def message_key(message):
    """Failure messages grouped up to numbers."""
    return re.sub(r"\d+(\.\d+)?(e[-+]?\d+)?", "N", message)


def read_results(path):
    """{nodeid: (outcome, message)} and the last test started without an outcome."""
    results, started = {}, None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if "start" in entry:
                started = entry["start"]
            else:
                # A test reports once; a failure in teardown (or a validation
                # report) after a pass wins.
                if entry["nodeid"] not in results or entry["outcome"] in ("failed", "validation"):
                    results[entry["nodeid"]] = (entry["outcome"], entry["message"])
                if entry["nodeid"] == started:
                    started = None
    if started in results:
        started = None
    return results, started


def cpu_twin(nodeid):
    """The CPU instance of a vk test: TestFooPRIVATEUSE1::test_x_vk_float32 ->
    TestFooCPU::test_x_cpu_float32."""
    path, cls, name = nodeid.split("::", 2)
    return f"{path}::{cls.replace('PRIVATEUSE1', 'CPU')}::{re.sub(r'_vk(?=_|$)', '_cpu', name)}"


def run_tests(tests, ids, env, results_path, log, extra, timeout):
    """Runs `ids`, restarting after crashes; returns the last exit code."""
    listing = results_path.with_suffix(".ids")
    remaining = list(ids)
    code = 0
    with open(log, "a", encoding="utf-8") as fh:
        while remaining:
            listing.write_text("\n".join(remaining), encoding="utf-8")
            cmd = [sys.executable, "-c", RUNNER, str(results_path), f"@{listing}", "-q", "-p", "no:cacheprovider",
                   *extra]
            start = fh.tell()
            try:
                code = subprocess.call(cmd, cwd=tests, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.TimeoutExpired:
                code = "timeout"
            if code == 4:
                # pytest runs nothing if any id is not found (e.g. a CPU
                # twin's, not collected for "vk"): record those, run the rest.
                fh.flush()
                with open(log, encoding="utf-8") as text:
                    text.seek(start)
                    missing = {line.split("::", 1)[1].strip() for line in text if line.startswith("ERROR: not found:")}
                missing = {nodeid for nodeid in remaining if nodeid.split("::", 1)[1] in missing}
                if missing:
                    with open(results_path, "a", encoding="utf-8") as out:
                        for nodeid in sorted(missing):
                            out.write(json.dumps({"nodeid": nodeid, "outcome": "not found", "message": ""}) + "\n")
                    remaining = [nodeid for nodeid in remaining if nodeid not in missing]
                    continue
            results, started = read_results(results_path)
            if started is None:
                break  # finished, or died between tests
            # A test started and never reported: the process died in it (a
            # crash, or pytest-timeout's thread method ending a hung test).
            with open(results_path, "a", encoding="utf-8") as out:
                out.write(json.dumps({"nodeid": started, "outcome": "crashed",
                                      "message": f"pytest process died (exit {code}) in this test"}) + "\n")
            fh.write(f"\n[runner] {started} crashed the process (exit {code}); continuing after it\n")
            fh.flush()
            done = set(results) | {started}
            remaining = [nodeid for nodeid in remaining if nodeid not in done]
    listing.unlink(missing_ok=True)
    return code


def collect(tests, file, env, extra):
    """The test ids pytest collects from `file` (with the -k/-m filters in extra)."""
    filters = []
    for i, arg in enumerate(extra):
        if arg in ("-k", "-m") and i + 1 < len(extra):
            filters += [arg, extra[i + 1]]
    cmd = [sys.executable, "-c", RUNNER, "-", file, "--collect-only", "-q", "-p", "no:cacheprovider", *filters]
    out = subprocess.run(cmd, cwd=tests, env=env, capture_output=True, text=True).stdout
    return sorted(line.strip() for line in out.splitlines() if "::" in line and not line.startswith(" "))


FAILING = ("failed", "crashed", "xpassed", "validation")

# Failures that are expected on "vk" by design, as (test id pattern, message
# pattern, reason); counted as "expected", not as failures. Every entry needs
# a reason a reader can check.
EXPECTED = [
    (r"_sparse_|_to_sparse_|test_errors_sparse", r"",
     "sparse layouts are out of scope for this backend (dense tensors only)"),
    (r"", r"Sparsevk' backend",
     "sparse layouts are out of scope for this backend (dense tensors only)"),
    (r"", r"Quantizedvk' backend",
     "quantized tensors are out of scope for this backend (as sparse layouts: dense tensors only)"),
    (r"", r"not implemented for 'ComplexHalf'",
     "complex32 arithmetic: PyTorch's CPU backend has no kernels for it either (only copies and casts)"),
    (r"test_dtypes_nn_functional_conv", r"",
     "int32 convolution works on vk as on the CPU, but the OpInfo does not list it"),
    (r"test_dtypes_nn_functional_max_pool1d_", r"",
     "integer max_pool1d: vk takes PyTorch's max_pool2d path (integers work, as in max_pool2d on the CPU); "
     "only the CPU's no-grad shortcut is float-only"),
    (r"test_upsamplingBiMode2d_nonsupported_dtypes_.*_mode_(bilinear|bicubic)_uint8", r"RuntimeError not raised",
     "uint8 bilinear/bicubic upsampling works on vk as on the CPU (the test expects only the CPU to support it)"),
    (r"test_float_to_int_conversion_finite_", r"",
     "float -> integer conversion of out-of-range values (float max) is undefined: GPUs saturate, x86 (NumPy's reference) does not"),
    (r"test_linalg_lstsq_input_checks_", r"`driver` other than `gels` is not supported on CUDA",
     "the test expects CUDA's lstsq drivers on any non-CPU device; vk supports all of the CPU's"),
    (r"test_trivial_fancy_out_of_bounds_", r"",
     "out-of-range indices are flagged by the kernel and raised at the next synchronization, as CUDA's "
     "device asserts (CUDA skips this test)"),
    (r"test_pow_cuda_complex_extremal_passing_", r"Scalars are not close",
     "(-1 + inf i)^2: the CPU's complex pow gives inf + nan i from its special-value path; vk squares exactly "
     "(-inf - inf i)"),
    (r"test_eig(vals)?_numpy_", r"Tensor-likes are not close",
     "for batched inputs the test applies one batch's eigenvalue sort to another ([::-1] reverses the batch "
     "axis), so it requires LAPACK's eigenvalue order; eigenvalues come in no particular order"),
    (r"test_index_put_large_indices_|test_index_reduce_reduce_(mean|prod)_vk_(bfloat16|float16)", r"Tensor-likes are not close",
     "vk accumulates 16-bit floats in float32 and rounds once; the CPU (the test's reference) rounds its running "
     "sum or product to 16 bits at every step (for bfloat16 index_put_: error ~1 on vk, ~230 on the CPU)"),
    (r"test_transformerencoderlayer_vk_", r"False is not true",
     "TransformerEncoderLayer takes its fast path on PrivateUse1 devices too (torch.nn's _supported_device_type), "
     "which gives NaN for fully masked rows; the test expects the fast path only on the CPU and CUDA"),
    (r"test_TransformerEncoderLayer_empty_", r"NestedTensorvk",
     "nested tensors are out of scope (as sparse ones)"),
    (r"test_broadcast_fn_map2?_", r"map2?_ is only implemented on CPU tensors",
     "Tensor.map_/map2_ are CPU-only Python callbacks; the test skips them only on CUDA"),
    (r"test_grad_scaler_deprecated_warning_|test_grad_scaling_state_dict_", r"",
     "the test assumes a device without torch.<device>.amp.GradScaler is the CPU (torch.cpu.amp, "
     "torch.FloatTensor)"),
    (r"test_grad_scaling_autocast_.*_Adam(W)?_", r"AssertionError not raised",
     "the test expects Adam under float16 autocast to diverge from float32; \"vk\" has no lower-precision "
     "autocast policy (PyTorch registers none for PrivateUse1), so both run in float32 and agree"),
    (r"test_python_ref_errors__refs_where_", r"is not on the expected device",
     "PyTorch's where reference words the mixed-device error its own way (an expected failure on CUDA and MPS)"),
    (r"test_(nan)?sum_vs_numpy_vk_float32", r"Tensor-likes are not close",
     "float32 sums compared with NumPy's pairwise sums within tolerances tighter than NumPy's own rounding "
     "error (even the exactly rounded sum misses them in ~0.5% of random inputs); only the CPU's summation "
     "order reliably passes"),
    (r"test_fmod_remainder_by_zero_integral_", r"",
     "integer x % 0: vk returns its own value (the CPU raises ZeroDivisionError, the test expects CUDA's bit patterns)"),
    (r"test_histc_min_max_corner_cases_device_vk_(u?int)", r"",
     "integer histc: vk follows the CPU, which has no integer kernel (the test expects CUDA's)"),
    (r"test_out_warning_mode_", r"Unexpected success",
     "vk's mode out= warns when it resizes a non-empty out= (PyTorch's composite kernel did not; the test expects that)"),
    (r"test_out_(cholesky_inverse|geqrf|ormqr|triangular_solve)_", r"Unexpected success",
     "vk passes an out= test PyTorch marks as failing for these ops (their CPU kernels change out='s strides)"),
    (r"", r"^KeyError: 'vk(\W|$)",
     "the test looks up per-device settings (drivers, tolerances) that have no entry for vk"),
    (r"_int4|int4pack|_dyn_quant_|_weight_int4", r"",
     "int4 packed-weight matmuls: packed layouts specific to x86/ARM CPUs and CUDA (out of scope)"),
    (r"", r"InductorError: KeyError: 'vk'",
     "torch.compile: inductor has no code generator for vk (out of scope; eager mode only)"),
]


def expected_reason(nodeid, message):
    for id_pattern, message_pattern, reason in EXPECTED:
        if re.search(id_pattern, nodeid) and re.search(message_pattern, message):
            return reason
    return None


def failing_on_cpu(tests, nodeids, out, stem, timeout):
    """The subset of vk test ids whose CPU twins also fail."""
    if not nodeids:
        return set()
    env = dict(os.environ, PYTORCH_TESTING_DEVICE_ONLY_FOR="cpu", PYTHONUNBUFFERED="1")
    twins = {cpu_twin(nodeid): nodeid for nodeid in nodeids}
    results_path = out / f"{stem}.cpu.jsonl"
    results_path.unlink(missing_ok=True)
    run_tests(tests, list(twins), env, results_path, out / f"{stem}.cpu.log", ["--timeout", "300"], timeout)
    return cpu_failures(results_path, nodeids)


def cpu_failures(results_path, nodeids):
    """The vk test ids among `nodeids` whose CPU twins (in results_path) fail
    or are expected to fail: PyTorch knows those tests fail on the CPU."""
    twins = {cpu_twin(nodeid): nodeid for nodeid in nodeids}
    results, _ = read_results(results_path)
    return {twins[cpu_id] for cpu_id, (outcome, _) in results.items()
            if (outcome in FAILING or outcome == "xfailed") and cpu_id in twins}


def describe(counts):
    return ", ".join(f"{outcome} {count}" for outcome, count in sorted(counts.items()) if count)


def summarize(out, stems):
    """Prints and returns outcome totals over the runs `stems` in out."""
    total = collections.Counter()
    messages = collections.Counter()
    for stem in stems:
        results, _ = read_results(out / f"{stem}.jsonl")
        both_file = out / f"{stem}.cpu-too.txt"
        both = set(both_file.read_text(encoding="utf-8").split()) if both_file.exists() else set()
        if (out / f"{stem}.cpu.jsonl").exists():  # (recomputed: the rule may have changed since the run)
            both = cpu_failures(out / f"{stem}.cpu.jsonl", list(results))
        counts = collections.Counter()
        vk_only = []
        expected = []
        for nodeid, (outcome, message) in sorted(results.items()):
            if outcome in FAILING and nodeid in both:
                counts["fails on CPU too"] += 1
            elif outcome in FAILING and expected_reason(nodeid, message):
                counts["expected"] += 1
                expected.append((nodeid, expected_reason(nodeid, message)))
            else:
                counts[outcome] += 1
                if outcome in FAILING:
                    vk_only.append((nodeid, outcome, message))
        messages.update(message_key(f"[{outcome}] {message}") for _, outcome, message in vk_only)
        with open(out / f"{stem}.failures.txt", "w", encoding="utf-8") as fh:
            for nodeid, outcome, message in vk_only:
                fh.write(f"{nodeid}\t{outcome}\t{message}\n")
        with open(out / f"{stem}.expected.txt", "w", encoding="utf-8") as fh:
            for nodeid, reason in expected:
                fh.write(f"{nodeid}\t{reason}\n")
        total.update(counts)
    print("\ntotal:", describe(total))
    if messages:
        print("\nmost common failure messages (vk only):")
        for message, count in messages.most_common(40):
            print(f"{count:6d}  {message}")
    return total


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--tests", default=os.environ.get("PYTORCH_TEST_DIR", ""))
    parser.add_argument("--out", default=str(pathlib.Path(__file__).with_name("pytorch_tests")))
    parser.add_argument("--shard", default="", help="INDEX/COUNT: run every COUNT-th collected test, starting at INDEX")
    parser.add_argument("--timeout", type=int, default=6 * 3600, help="seconds per pytest process")
    parser.add_argument("--merge", action="store_true", help="summarize every run in --out")
    parser.add_argument("--ids", default="", help="run these test ids (one per line) instead of collecting a file's")
    parser.add_argument("--name", default="", help="run name for --ids (default: the ids file's name)")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.merge:
        stems = sorted(path.name[:-6] for path in out.glob("*.jsonl") if not path.name.endswith(".cpu.jsonl"))
        summarize(out, stems)
        return
    tests = pathlib.Path(args.tests)
    if not (tests / "test_ops.py").exists():
        sys.exit(f"{tests} is not PyTorch's test/ directory (set --tests or PYTORCH_TEST_DIR)")
    env = dict(os.environ, PYTORCH_TESTING_DEVICE_ONLY_FOR="vk", PYTHONUNBUFFERED="1")
    # The -k / -m filters are applied when collecting the ids.
    run_extra = [arg for i, arg in enumerate(extra)
                 if arg not in ("-k", "-m") and (i == 0 or extra[i - 1] not in ("-k", "-m"))]
    stems = []
    runs = [(test_file, None) for test_file in args.files]
    if args.ids:
        listed = [line.split()[0] for line in pathlib.Path(args.ids).read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        # One run per test file (one pytest process finds the device-generic
        # tests of only one file).
        name = args.name or pathlib.Path(args.ids).stem
        by_file = collections.defaultdict(set)
        for nodeid in listed:
            by_file[nodeid.split("::")[0]].add(nodeid)
        runs = [(name if len(by_file) == 1 else f"{name}-{pathlib.Path(test_file).stem}", sorted(nodeids))
                for test_file, nodeids in sorted(by_file.items())]
    for test_file, listed in runs:
        ids = listed if listed is not None else collect(tests, test_file, env, extra)
        stem = pathlib.Path(test_file).stem
        if args.shard:
            index, count = (int(x) for x in args.shard.split("/"))
            ids = ids[index::count]
            stem += f".shard{index}-{count}"
        results_path = out / f"{stem}.jsonl"
        results_path.unlink(missing_ok=True)
        code = run_tests(tests, ids, env, results_path, out / f"{stem}.log", run_extra, args.timeout)
        results, _ = read_results(results_path)
        failing = [nodeid for nodeid, (outcome, _) in results.items() if outcome in FAILING]
        both = failing_on_cpu(tests, failing, out, stem, args.timeout)
        (out / f"{stem}.cpu-too.txt").write_text("\n".join(sorted(both)), encoding="utf-8")
        counts = collections.Counter(outcome for outcome, _ in results.values())
        print(f"{stem}: {len(ids)} tests, exit {code}; {describe(counts)}", flush=True)
        stems.append(stem)
    summarize(out, stems)


if __name__ == "__main__":
    main()
