#!/bin/bash
# PBS job array of lavapipe shards, one single-core job per shard (the cluster's
# queue runs up to 48 CPUs per user at once):
#   qsub -J 0-47 -v FIRST_SHARD=0,SHARD_COUNT=62,OUT_DIR=...,TEST_FILES=test_ops.py tools/cluster/pbs-lavapipe.sh
# (job i runs shard FIRST_SHARD + i).
#PBS -N torch-vulkan-tests
#PBS -q workq
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=12:00:00
#PBS -j oe
SHARD=$((FIRST_SHARD + PBS_ARRAY_INDEX))
export LP_NUM_THREADS=1  # one core per job
exec "${TORCH_VULKAN:-$HOME/torch-vulkan}/tools/cluster/run-shard.sh" "$SHARD" "$SHARD_COUNT" lavapipe "$OUT_DIR" \
  $TEST_FILES
