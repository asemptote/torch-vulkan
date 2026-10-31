#!/bin/bash
# Submits a full run of PyTorch's device-generic suites on lavapipe: 48
# single-core PBS jobs, one shard each. Run on the head node:
#   bash tools/cluster/full-run.sh NAME        (results in $RUNS/NAME)
# Merge afterwards (on ws3):
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
set -e
NAME=${1:?usage: full-run.sh NAME}
RUNS=${RUNS:-/shared/scratch/me/torch-vulkan-runs}
FILES="test_ops.py test_torch.py test_binary_ufuncs.py test_unary_ufuncs.py test_nn.py test_linalg.py \
test_reductions.py test_indexing.py test_tensor_creation_ops.py test_shape_ops.py test_type_promotion.py \
test_sort_and_select.py test_view_ops.py"
cd "${TORCH_VULKAN:-$HOME/torch-vulkan}"
mkdir -p "$RUNS/$NAME" "$RUNS/pbs-logs"
qsub -J 0-47 -o "$RUNS/pbs-logs" \
  -v "FIRST_SHARD=0,SHARD_COUNT=48,OUT_DIR=$RUNS/$NAME,TEST_FILES=$FILES" tools/cluster/pbs-lavapipe.sh
