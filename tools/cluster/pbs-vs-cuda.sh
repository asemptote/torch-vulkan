#!/bin/bash
# benchmarks/vs_cuda.py on one GPU node: vk (NVIDIA's Vulkan driver) against
# CUDA, the same PyTorch release in two virtual environments. From the head
# node, with this site's settings in tools/cluster/cluster.env:
#   qsub -q GPU_QUEUE -l select=1:ncpus=4:ngpus=1:mem=32gb -l walltime=00:30:00 \
#        -v TORCH_VULKAN=DEV_DIR,OUT=NAME.md tools/cluster/pbs-vs-cuda.sh
# TORCH_VULKAN: a built copy of this repo (its .venv has torch_vulkan);
# CUDA_PYTHON (cluster.env): an interpreter with torch built for CUDA; OUT:
# the table's file, relative to TORCH_VULKAN/benchmarks/results.
#PBS -N tv-vs-cuda
#PBS -j oe
set -u
cd "${TORCH_VULKAN:?}" || exit 1
CFG=tools/cluster/cluster.env
if [ -f "$CFG" ]; then . "$CFG"; fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json OMP_NUM_THREADS=4
.venv/bin/python benchmarks/vs_cuda.py --vk-python "$PWD/.venv/bin/python" \
  --cuda-python "${CUDA_PYTHON:?set CUDA_PYTHON: see cluster.env.example}" \
  --timeout "${TIMEOUT:-90}" --out "benchmarks/results/${OUT:?}"
