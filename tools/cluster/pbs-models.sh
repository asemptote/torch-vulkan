#!/bin/bash
# benchmarks/models/sweep.py on one GPU node: PyTorch's model suites in eager
# mode on vk and CUDA (the same GPU), after the CPU reference phase (which
# needs no GPU: run it first as a CPU job, PHASE=reference). From the head node:
#   qsub -q CPU_QUEUE -l select=1:ncpus=16:mem=128gb -l walltime=12:00:00 \
#        -v TORCH_VULKAN=DIR,MODELS_DIR=BIG_DIR,OUT=OUT_DIR,PHASE=reference tools/cluster/pbs-models.sh
#   qsub -q GPU_QUEUE -l select=1:ncpus=16:ngpus=1:mem=96gb -l walltime=24:00:00 \
#        -v TORCH_VULKAN=DIR,MODELS_DIR=BIG_DIR,OUT=OUT_DIR tools/cluster/pbs-models.sh
# TORCH_VULKAN: a built copy of this repo (its .venv has torch_vulkan and the
# suites' dependencies); MODELS_DIR: where benchmarks/models/setup.sh put
# PyTorch's benchmarks and TorchBench, and the downloads' caches (large);
# CUDA_PYTHON (cluster.env): the same torch release built for CUDA, set up
# by setup.sh too. SUITES (default "timm huggingface torchbench"), PRECISION
# (float32), MODE (inference) and SWEEP_ARGS (more sweep.py options) choose
# the runs. Results: OUT/results.jsonl; tables: benchmarks/models/report.py.
#PBS -N tv-models
#PBS -j oe
set -u
cd "${TORCH_VULKAN:?}" || exit 1
CFG=tools/cluster/cluster.env
if [ -f "$CFG" ]; then . "$CFG"; fi
M=${MODELS_DIR:?}
export PYTORCH_BENCH_DIR=$M/pytorch-bench TORCHBENCH_DIR=$M/torchbench
export HF_HOME=$M/hf TORCH_HOME=$M/torch-home XDG_CACHE_HOME=$M/xdg-cache TOKENIZERS_PARALLELISM=false
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PHASE=${PHASE:-gpu}
if [ "$PHASE" = gpu ]; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; fi
for suite in ${SUITES:-timm huggingface torchbench}; do
  .venv/bin/python benchmarks/models/sweep.py --phase "$PHASE" --suite "$suite" --out "${OUT:?}" \
    --cuda-python "${CUDA_PYTHON:?set CUDA_PYTHON: see cluster.env.example}" \
    --precision "${PRECISION:-float32}" --mode "${MODE:-inference}" ${SWEEP_ARGS:-}
done
