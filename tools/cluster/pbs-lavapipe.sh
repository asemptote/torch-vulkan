#!/bin/bash
# PBS job array of lavapipe shards, one single-core job per shard (the cluster's
# queue runs up to 48 CPUs per user at once):
#   qsub -J 0-47 -v FIRST_SHARD=0,SHARD_COUNT=62,OUT_DIR=...,TEST_FILES=test_ops.py tools/cluster/pbs-lavapipe.sh
# (job i runs shard FIRST_SHARD + i).
#PBS -N torch-vulkan-tests
#PBS -q workq
#PBS -l select=1:ncpus=1:mem=6gb
#PBS -l walltime=12:00:00
#PBS -j oe
export LP_NUM_THREADS=1  # one core per job
# With RUN_DIR (queue-run.sh): a worker claiming shards from the run's queue.
if [ -n "${RUN_DIR:-}" ]; then
  exec "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/worker.sh" lavapipe "$RUN_DIR"
fi
SHARD=$((FIRST_SHARD + PBS_ARRAY_INDEX))
exec "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/run-shard.sh" "$SHARD" "$SHARD_COUNT" lavapipe "$OUT_DIR" \
  $TEST_FILES
