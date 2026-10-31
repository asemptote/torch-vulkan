#!/bin/bash
# One GPU job running SHARDS_PER_JOB shards side by side on the node's NVIDIA
# GPU (the h100 and l40s queues run one job per user at a time):
#   qsub -q h100 -l select=1:ncpus=16:ngpus=1:mem=48gb \
#        -v FIRST_SHARD=0,SHARDS_PER_JOB=16,SHARD_COUNT=16,OUT_DIR=...,TEST_FILES=test_ops.py tools/cluster/pbs-gpu.sh
#PBS -N torch-vulkan-gpu
#PBS -l walltime=12:00:00
#PBS -j oe
nvidia-smi -L
for i in $(seq 0 $((SHARDS_PER_JOB - 1))); do
  "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/run-shard.sh" $((FIRST_SHARD + i)) "$SHARD_COUNT" nvidia \
    "$OUT_DIR" $TEST_FILES &
done
wait
