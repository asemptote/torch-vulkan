#!/bin/bash
# One GPU job running SHARDS_PER_JOB shards side by side on the node's NVIDIA
# GPU (the h100 and l40s queues run one job per user at a time):
#   qsub -q h100 -l select=1:ncpus=16:ngpus=1:mem=48gb \
#        -v FIRST_SHARD=0,SHARDS_PER_JOB=16,SHARD_COUNT=16,OUT_DIR=...,TEST_FILES=test_ops.py tools/cluster/pbs-gpu.sh
#PBS -N torch-vulkan-gpu
#PBS -l walltime=12:00:00
#PBS -j oe
nvidia-smi -L
# With RUN_DIR (queue-run.sh): WORKERS workers claiming shards from the run's queue.
if [ -n "${RUN_DIR:-}" ]; then
  for i in $(seq 1 "${WORKERS:-16}"); do
    "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/worker.sh" nvidia "$RUN_DIR" &
  done
  wait
  exit 0
fi
for i in $(seq 0 $((SHARDS_PER_JOB - 1))); do
  "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/run-shard.sh" $((FIRST_SHARD + i)) "$SHARD_COUNT" nvidia \
    "$OUT_DIR" $TEST_FILES &
done
wait
