"""PyTorch's model benchmark suites in eager mode on "vk" (and CUDA, the CPU).

    python benchmarks/models/run.py SUITE [this script's options] [runner options]

SUITE is timm, huggingface or torchbench: PyTorch's own runners,
benchmarks/dynamo/{timm_models,huggingface,torchbench}.py from the PyTorch
source at the installed torch's commit (--pytorch-dir, or PYTORCH_BENCH_DIR:
a checkout holding benchmarks/dynamo/), driven unchanged through their
main(). The runner options are theirs; for example, eager latency (the
runner's "abs_latency": the median of --repeat synchronised calls, after its
warm-up) of one model on vk:

    python benchmarks/models/run.py timm --devices vk --nothing --performance \\
        --inference --float32 --only resnet50 --output out.csv

(`--nothing` is the runner's experiment without torch.compile: eager
against eager.) TorchBench models come from a pytorch/benchmark checkout
(--torchbench-dir, or TORCHBENCH_DIR).

The adapter for "vk" (the runners know cuda, xpu, hpu and the CPU) is three
patches of benchmarks/dynamo/common.py, made only when "vk" is among the
--devices: synchronize() (which the runner's timer calls; a no-op without
CUDA) becomes torch_vulkan.synchronize, empty_gpu_cache() accepts "vk", and
init_optimizer() gives "vk" the optimizer it gives "cuda" (Adam, or SGD for
some models; without it, training on any other device steps no optimizer).

Accuracy across devices. The runner's --accuracy mode compares two runs on
one device (eager against torch.compile, with a float64 run on that device
as the reference for the error), so it cannot compare vk with CUDA. With
--write-reference, --reference and --outputs, this script keeps the
runner's accuracy mode (its small batch sizes, eval mode, determinism
settings, casts and AMP) and replaces only the comparison
(BenchmarkRunner.check_accuracy):

  --write-reference REF   (on the CPU) saves the model's float32 weights,
                          the example inputs, and the outputs of the model
                          cast to float64 (the runner's fp64 reference).
  --reference REF         loads those weights and inputs into the model the
                          runner loaded (on its device, as it would), so
                          that every device runs the same model on the same
                          inputs.
  --outputs OUT           runs the model twice (eager), and saves the first
                          run's outputs, whether the second was bitwise the
                          same, and the runner's tolerance settings for it.

compare.py then checks vk's outputs against CUDA's with the runner's own
test (torch._dynamo.utils.same, its tolerance, CUDA as the "correct" result
and the CPU's float64 outputs as the reference for the error).

--record FILE writes one JSON object: the run's status, the first error (the
innermost exception: on vk with the CPU fallback off, usually the op with no
Vulkan kernel), the runner's output row (latency), and on vk the ops that ran
on the CPU (torch_vulkan.fallback_counts(), with TORCH_VK_CPU_FALLBACK=1).
"""

import argparse
import atexit
import importlib
import json
import logging
import os
import sys
import time
import traceback

SUITES = {
    "timm": ("timm_models", "TimmRunner", "timm_main"),
    "huggingface": ("huggingface", "HuggingfaceRunner", "huggingface_main"),
    "torchbench": ("torchbench", "TorchBenchmarkRunner", "torchbench_main"),
}

RECORD = {"status": None, "error": None, "traceback": None, "row": None}


def innermost(exc):
    """The exception at the bottom of a chain ("Eager run failed" from ...)."""
    seen = set()
    while id(exc) not in seen:
        seen.add(id(exc))
        nxt = exc.__cause__ or exc.__context__
        if nxt is None:
            break
        exc = nxt
    return exc


def note_error(exc, where):
    if RECORD["error"] is not None:
        return
    inner = innermost(exc)
    msg = str(inner).strip().splitlines()
    RECORD["error"] = f"{type(inner).__name__}: {msg[0] if msg else ''}"[:2000]
    RECORD["error_where"] = where
    RECORD["traceback"] = "".join(traceback.format_exception(exc))[-20000:]


class _ErrorLog(logging.Handler):
    """The runners log some failures (log.exception) instead of raising them."""

    def emit(self, record):
        if record.exc_info and record.exc_info[1] is not None:
            note_error(record.exc_info[1], f"log: {record.getMessage()[:200]}")


def to_cpu(obj):
    """Outputs as plain containers of CPU tensors (comparable across processes)."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):  # (transformers' ModelOutput is a dict)
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*(to_cpu(v) for v in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def like(ref, native):
    """ref's tensors on native's devices, in native's structure; raises if the
    two differ in structure or in a tensor's shape or dtype."""
    import torch

    if isinstance(native, torch.Tensor):
        # (in the native dtype: the runner casts some suites' inputs as it loads them, --bfloat16 say)
        if (not isinstance(ref, torch.Tensor) or ref.shape != native.shape
                or ref.is_floating_point() != native.is_floating_point()):
            raise ValueError(f"reference input {getattr(ref, 'shape', type(ref))} != {native.shape} {native.dtype}")
        return ref.to(native.device, native.dtype).requires_grad_(native.requires_grad)
    if isinstance(native, dict):
        if not isinstance(ref, dict) or set(ref) != set(native):
            raise ValueError("reference inputs: different keys")
        out = {k: like(ref[k], v) for k, v in native.items()}
        return type(native)(out) if type(native) is not dict else out
    if isinstance(native, (list, tuple)):
        if not isinstance(ref, (list, tuple)) or len(ref) != len(native):
            raise ValueError("reference inputs: different structure")
        items = [like(r, n) for r, n in zip(ref, native)]
        if isinstance(native, tuple) and hasattr(native, "_fields"):
            return type(native)(*items)
        return type(native)(items)
    return native


def install_vk_adapter(common):
    import torch_vulkan

    common.synchronize = torch_vulkan.synchronize
    empty = common.empty_gpu_cache

    def empty_gpu_cache(device):
        if device == "vk":
            torch_vulkan.empty_cache()
        else:
            empty(device)

    common.empty_gpu_cache = empty_gpu_cache
    init_optimizer = common.BenchmarkRunner.init_optimizer

    def init_optimizer_vk(self, name, device, params):
        return init_optimizer(self, name, "cuda" if device == "vk" else device, params)

    common.BenchmarkRunner.init_optimizer = init_optimizer_vk


def install_capture(common, runner_cls, opts):
    import torch
    from torch._dynamo.testing import reset_rng_state
    from torch._dynamo.utils import clone_inputs, same

    def check_accuracy(self, name, model, example_inputs, optimize_ctx, experiment, tag):
        device = common.current_device
        training = self.args.training
        try:
            with self.pick_grad(name, training):
                if opts.write_reference:
                    ref = {"state": to_cpu(model.state_dict()), "inputs": to_cpu(example_inputs)}
                    model64, inputs64 = common.cast_to_fp64(
                        self.deepcopy_and_maybe_parallelize(model), clone_inputs(example_inputs))
                    self.init_optimizer(name, device, model64.parameters())
                    reset_rng_state()
                    t0 = time.perf_counter()
                    ref["fp64"] = to_cpu(self.run_n_iterations(model64, inputs64, self.model_iter_fn))
                    RECORD["fp64_seconds"] = time.perf_counter() - t0
                    torch.save(ref, opts.write_reference + ".tmp")  # (sweeps read it as soon as it exists)
                    os.replace(opts.write_reference + ".tmp", opts.write_reference)
                    RECORD["status"] = "reference_saved"
                    return "reference_saved"
                if opts.reference:
                    ref = torch.load(opts.reference, weights_only=False)
                    model.load_state_dict(ref["state"])
                    example_inputs = like(ref["inputs"], example_inputs)
                    del ref
                tol, cosine = self.get_tolerance_and_cosine_flag(training, device, name)
                model, example_inputs = self.maybe_cast(model, example_inputs)
                runs = []
                for _ in range(2):
                    reset_rng_state()
                    copy = self.deepcopy_and_maybe_parallelize(model)
                    self.init_optimizer(name, device, copy.parameters())
                    t0 = time.perf_counter()
                    out = self.run_n_iterations(copy, clone_inputs(example_inputs), self.model_iter_fn)
                    common.synchronize()
                    RECORD.setdefault("accuracy_run_seconds", []).append(time.perf_counter() - t0)
                    runs.append(to_cpu(out))
                    del copy, out
                    self.optimizer = None
                try:
                    deterministic = bool(same(runs[0], runs[1], fp64_ref=None, cos_similarity=False, tol=0,
                                              equal_nan=self.equal_nan))
                except Exception:
                    deterministic = False
                torch.save({
                    "outputs": runs[0],
                    "deterministic": deterministic,
                    "same_kwargs": {
                        "tol": tol,
                        "cos_similarity": cosine,
                        "equal_nan": self.equal_nan,
                        "use_larger_multiplier_for_smaller_tensor":
                            self.use_larger_multiplier_for_smaller_tensor(name),
                        "use_iou_for_bool": self.use_iou_for_bool_accuracy(name),
                        "iou_threshold": self.get_iou_threshold(name),
                    },
                    "non_deterministic_model": name in self.non_deterministic_models
                    or name in self.skip_accuracy_check_as_eager_non_deterministic,
                }, opts.outputs)
                RECORD["status"] = "captured"
                RECORD["deterministic"] = deterministic
                return "captured"
        except Exception as e:
            note_error(e, "accuracy run")
            RECORD["status"] = "fail_to_run"
            log = logging.getLogger("run.py")
            log.exception("accuracy run failed")
            return "fail_to_run"

    runner_cls.check_accuracy = check_accuracy


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("suite", choices=sorted(SUITES))
    p.add_argument("--pytorch-dir", default=os.environ.get("PYTORCH_BENCH_DIR"))
    p.add_argument("--torchbench-dir", default=os.environ.get("TORCHBENCH_DIR"))
    p.add_argument("--record")
    p.add_argument("--write-reference")
    p.add_argument("--reference")
    p.add_argument("--outputs")
    opts, rest = p.parse_known_args()
    if not opts.pytorch_dir:
        sys.exit("set --pytorch-dir or PYTORCH_BENCH_DIR: a PyTorch checkout with benchmarks/dynamo/")
    dynamo_dir = os.path.join(os.path.abspath(opts.pytorch_dir), "benchmarks", "dynamo")
    devices = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a in ("--devices", "--device", "-d")]

    if opts.record:
        RECORD.update(argv=sys.argv[1:], devices=devices)

        def write_record():
            if "vk" in devices:
                import torch_vulkan

                RECORD["cpu_fallback"] = torch_vulkan.cpu_fallback_enabled()
                RECORD["fallback_counts"] = dict(torch_vulkan.fallback_counts())
            with open(opts.record, "w") as f:
                json.dump(RECORD, f, indent=1, default=str)

        atexit.register(write_record)

    if "vk" in devices:
        import torch_vulkan  # noqa: F401  (registers "vk" before the runner parses its options)
    if opts.suite == "torchbench":
        if not opts.torchbench_dir:
            sys.exit("set --torchbench-dir or TORCHBENCH_DIR: a pytorch/benchmark checkout")
        # (the runner looks for torchbenchmark/ beside the working directory,
        # and imports it as a package)
        tb = os.path.abspath(opts.torchbench_dir)
        os.chdir(tb)
        sys.path.insert(0, tb)
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [tb, os.environ.get("PYTHONPATH")]))
    sys.path.insert(0, dynamo_dir)
    import common

    logging.getLogger().addHandler(_ErrorLog())
    if "vk" in devices:
        install_vk_adapter(common)

    write_outputs = common.write_outputs

    def write_outputs_rec(filename, headers, row, *args, **kwargs):
        if "abs_latency" in headers or RECORD["row"] is None:  # (the latency row, not compile times)
            RECORD["row"] = dict(zip(headers, [str(x) for x in row]))
        return write_outputs(filename, headers, row, *args, **kwargs)

    common.write_outputs = write_outputs_rec
    module_name, cls_name, main_name = SUITES[opts.suite]
    module = importlib.import_module(module_name)
    if opts.write_reference or opts.reference or opts.outputs:
        install_capture(common, getattr(module, cls_name), opts)

    runner_cls = getattr(module, cls_name)
    load_model = runner_cls.load_model

    def load_model_rec(self, *args, **kwargs):
        try:
            return load_model(self, *args, **kwargs)
        except Exception as e:
            note_error(e, "load_model (the runner's first eager run)")
            RECORD["status"] = "eager_fail_to_run"
            raise

    runner_cls.load_model = load_model_rec
    sys.argv = [os.path.join(dynamo_dir, module_name + ".py")] + rest
    try:
        getattr(module, main_name)()
    except SystemExit as e:
        RECORD["exit"] = e.code
        raise
    except BaseException as e:
        note_error(e, "main")
        RECORD["status"] = RECORD["status"] or "crashed"
        raise
    if RECORD["status"] is None:  # (the runner reports some failures without raising them)
        RECORD["status"] = "fail" if RECORD["error"] is not None else "pass"


if __name__ == "__main__":
    main()
