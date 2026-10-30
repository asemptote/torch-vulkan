#!/bin/bash
# Runs one shard of PyTorch's test files on "vk" (tools/pytorch_tests.py), at
# low priority, on the given Vulkan driver:
#   run-shard.sh SHARD SHARD_COUNT DRIVER OUT_DIR TEST_FILE...
# SHARD is 0 .. SHARD_COUNT - 1; DRIVER is lavapipe (Mesa's CPU driver) or
# nvidia. Expects the repo at $TORCH_VULKAN (default ~/torch-vulkan, built
# into its .venv) and PyTorch's tests at $PYTORCH_TEST_DIR (default
# ~/pytorch-src/test).
set -uo pipefail
SHARD=$1; SHARD_COUNT=$2; DRIVER=$3; OUT_DIR=$4; shift 4
case "$DRIVER" in
  lavapipe) ICD=lvp ;;  # Mesa's file name for lavapipe
  nvidia) ICD=nvidia ;;
  *) echo "unknown driver $DRIVER (lavapipe or nvidia)" >&2; exit 2 ;;
esac
cd "${TORCH_VULKAN:-$HOME/torch-vulkan}"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/${ICD}_icd.x86_64.json
export LP_NUM_THREADS=${LP_NUM_THREADS:-2}  # lavapipe's threads per process
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_TEST_DIR=${PYTORCH_TEST_DIR:-$HOME/pytorch-src/test}
mkdir -p "$OUT_DIR"
exec nice -n 19 .venv/bin/python tools/pytorch_tests.py --out "$OUT_DIR" --shard "$SHARD/$SHARD_COUNT" "$@" \
  -- --timeout 600 --timeout-method thread >"$OUT_DIR/runner.shard$SHARD-$SHARD_COUNT.$(hostname -s).log" 2>&1
