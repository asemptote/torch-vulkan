#!/bin/bash
# Submits a full run of PyTorch's device-generic suites: 48 single-core PBS
# jobs on lavapipe, one shard each, and with --gpu also one H100 job running
# 16 shards on the NVIDIA driver (a separate queue: no extra wall time).
# Run on the head node:
#   bash tools/cluster/full-run.sh NAME [--gpu]   (results in $RUNS/NAME, $RUNS/NAME-h100)
# Merge afterwards (on any machine that sees $RUNS):
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
# Settings (environment, or tools/cluster/cluster.env): RUNS (default
# ~/torch-vulkan-runs), LAVAPIPE_QUEUE, GPU_QUEUES (--gpu uses the first).
set -e
NAME=${1:?usage: full-run.sh NAME [--gpu]}
CFG="$(dirname "$0")/cluster.env"
if [ -f "$CFG" ]; then . "$CFG"; fi
RUNS=${RUNS:-$HOME/torch-vulkan-runs}
cd "${TORCH_VULKAN:-$HOME/torch-vulkan}"
FILES=$(grep -v '^#' tools/cluster/test_files.txt | tr '\n' ' ')
mkdir -p "$RUNS/$NAME" "$RUNS/pbs-logs"
qsub ${LAVAPIPE_QUEUE:+-q $LAVAPIPE_QUEUE} -J 0-47 -o "$RUNS/pbs-logs" \
  -v "FIRST_SHARD=0,SHARD_COUNT=48,OUT_DIR=$RUNS/$NAME,TEST_FILES=$FILES" tools/cluster/pbs-lavapipe.sh
if [ "${2:-}" = "--gpu" ]; then
  mkdir -p "$RUNS/$NAME-h100"
  Q=${GPU_QUEUES:?set GPU_QUEUES for --gpu}
  qsub -q "${Q%% *}" -l select=1:ncpus=16:ngpus=1:mem=32gb -o "$RUNS/pbs-logs" \
    -v "FIRST_SHARD=0,SHARDS_PER_JOB=16,SHARD_COUNT=16,OUT_DIR=$RUNS/$NAME-h100,TEST_FILES=$FILES" tools/cluster/pbs-gpu.sh
fi
