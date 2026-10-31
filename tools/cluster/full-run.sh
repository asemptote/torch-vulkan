#!/bin/bash
# Submits a full run of PyTorch's device-generic suites: 48 single-core PBS
# jobs on lavapipe, one shard each, and with --gpu also one H100 job running
# 16 shards on the NVIDIA driver (a separate queue: no extra wall time).
# Run on the head node:
#   bash tools/cluster/full-run.sh NAME [--gpu]   (results in $RUNS/NAME, $RUNS/NAME-h100)
# Merge afterwards (on ws3):
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
set -e
NAME=${1:?usage: full-run.sh NAME [--gpu]}
RUNS=${RUNS:-/shared/scratch/me/torch-vulkan-runs}
FILES="test_ops.py test_torch.py test_binary_ufuncs.py test_unary_ufuncs.py test_nn.py test_linalg.py \
test_reductions.py test_indexing.py test_tensor_creation_ops.py test_shape_ops.py test_type_promotion.py \
test_sort_and_select.py test_view_ops.py"
cd "${TORCH_VULKAN:-$HOME/torch-vulkan}"
mkdir -p "$RUNS/$NAME" "$RUNS/pbs-logs"
qsub -J 0-47 -o "$RUNS/pbs-logs" \
  -v "FIRST_SHARD=0,SHARD_COUNT=48,OUT_DIR=$RUNS/$NAME,TEST_FILES=$FILES" tools/cluster/pbs-lavapipe.sh
if [ "${2:-}" = "--gpu" ]; then
  mkdir -p "$RUNS/$NAME-h100"
  qsub -q h100 -l select=1:ncpus=16:ngpus=1:mem=64gb -o "$RUNS/pbs-logs" \
    -v "FIRST_SHARD=0,SHARDS_PER_JOB=16,SHARD_COUNT=16,OUT_DIR=$RUNS/$NAME-h100,TEST_FILES=$FILES" tools/cluster/pbs-gpu.sh
fi
