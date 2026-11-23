# Whole models: PyTorch's benchmark suites on vk and CUDA (M51)

Issue #128. `benchmarks/vs_cuda.py` compares single ops with CUDA on the same
GPU; this compares whole models, with the suites PyTorch's own dashboards
use: TorchBench, TIMM and HuggingFace (`benchmarks/dynamo/` in the PyTorch
repository), in eager mode, on one NVIDIA L40S, once on CUDA and once on vk.
The tables are in [benchmarks/models/results.md](../../benchmarks/models/results.md)
(and `.csv`, `.json`); the scripts are in `benchmarks/models/`.

## The runners, unchanged

`benchmarks/models/run.py SUITE ...` imports PyTorch's runner for the suite
(`timm_models.py`, `huggingface.py`, `torchbench.py`, from a sparse checkout
at the installed torch's commit) and calls its `main()` with the runner's own
options. Nothing is copied. Where "vk" is among `--devices`, three functions
of `common.py` are patched, the smallest adapter that made the runners work:

- `synchronize()`, which the runner's timer calls after the timed calls, is
  `torch.cuda.synchronize` only on a CUDA build and a no-op otherwise; it
  becomes `torch_vulkan.synchronize`. (Without it, vk's times are the time
  to record the work, not to do it.)
- `empty_gpu_cache()` warns for any device but cuda, xpu and mps; "vk"
  calls `torch_vulkan.empty_cache()`.
- `init_optimizer()` gives an optimizer (Adam, or SGD for a few models) only
  to "cuda"; elsewhere training steps no optimizer at all. "vk" gets CUDA's.

Everything else the runners need from a device ("vk" in `--devices`,
`torch.manual_seed` seeding it, `torch.autocast("vk")` for `--amp`,
`torch.cuda.OutOfMemoryError` in `except` clauses) already works. The
runners load each model on the device they are given, as on CUDA.

## Latency

The runner's own: `--nothing --performance` is its speed-up experiment with
no compiler, eager against eager. After loading (which runs the model once)
and five warm-up calls, it times `--repeat` (30) calls, each followed by
`synchronize()`; its `abs_latency` is their median, in ms. Batch sizes are
the suites' (`timm_models_list.txt`, `huggingface_models_list.txt` and
`huggingface.yaml`'s divisors, TorchBench's per-model defaults and
`torchbench.yaml`), the same on both devices.

Precision is what the runners set for performance runs: `common.py` sets
`torch.backends.cuda.matmul.allow_tf32 = True` at import, which makes
`torch.get_float32_matmul_precision()` "high" in both processes. CUDA's
float32 products and cuDNN's convolutions then use TF32; vk's float32
products use three bfloat16 products each (M41). The accuracy runs below are
at "highest" on both (the runner's `setup_determinism`).

## Accuracy against CUDA

The runner's `--accuracy` mode compares two runs on one device (eager
against `torch.compile`), with the model cast to float64 on that device as
the reference for the error, so it cannot compare vk with CUDA as it
stands. `run.py` keeps everything the mode sets up (its small batch sizes,
eval mode, determinism settings, casts, autocast) and replaces only the
comparison (`BenchmarkRunner.check_accuracy`):

1. On the CPU (no GPU needed), the runner loads the model; its float32
   weights, its example inputs and the outputs of the model cast to float64
   are saved (`--write-reference`).
2. On CUDA and on vk, the runner loads the model on the device, as usual,
   and the reference's weights and inputs replace the ones it made
   (`--reference`): the two devices' random numbers differ, and the runners
   generate inputs on the device. The model runs twice, eager; the first
   run's outputs are saved, with whether the second was the same
   (`--outputs`).
3. `compare.py` applies the runner's own test, `torch._dynamo.utils.same`
   with the runner's tolerance for the model, CUDA's outputs as the correct
   result, vk's as the new one and the CPU's float64 outputs as the
   reference for the error. It also reports each device's relative RMS
   error against float64.

## What runs on vk

With the CPU fallback off (the default since M45), a model "runs on vk"
when both the accuracy run and the latency run finish. Where one fails, the
sweep reruns it with `TORCH_VK_CPU_FALLBACK=1` and
`TORCH_VK_FALLBACK_DETAIL=1` and records `fallback_counts()`: every op that
went to the CPU, with its arguments' types and shapes. The blockers in
results.md are those ops (or the error, where the fallback does not help),
ranked by the number of models each stops.

## Results

RESULTS

## Reproducing

```sh
# once: the suites' sources, pins and models, in a large directory, for the
# vk environment and one with the same torch release built for CUDA
benchmarks/models/setup.sh BIG_DIR .venv/bin/python CUDA_VENV/bin/python
export PYTORCH_BENCH_DIR=BIG_DIR/pytorch-bench TORCHBENCH_DIR=BIG_DIR/torchbench HF_HOME=BIG_DIR/hf
python benchmarks/models/sweep.py --phase reference --suite timm --out OUT   # CPU
python benchmarks/models/sweep.py --phase gpu --suite timm --out OUT \
    --cuda-python CUDA_VENV/bin/python --perf-with-fallback                   # GPU
python benchmarks/models/report.py OUT/results.jsonl --md results.md --csv results.csv --json results.json
```

`tools/cluster/pbs-models.sh` runs the two phases as PBS jobs. A single
model through the runner alone:

```sh
python benchmarks/models/run.py timm --devices vk --nothing --performance \
    --inference --float32 --only resnet50 --output out.csv
```
