#!/bin/bash
# A full run spread over every machine that can take part, from the local
# repo root:
#   tools/cluster/queue-run.sh start NAME [TEST_FILE...]
#   tools/cluster/queue-run.sh status NAME
#
# Two runs, one per driver family, each a queue of small shards on the
# shared file system ($RUNS/NAME for lavapipe, $RUNS/NAME-nvidia):
#   - the cluster: PBS_JOBS single-core lavapipe jobs, and an H100 and an
#     L40S job (separate queues) of GPU_WORKERS workers each;
#   - backfill on the workstations (pool.sh): ws0's and ws3's cores on the
#     lavapipe run, ws3's Tesla M4 on the NVIDIA run, and ws2's free cores.
# The cluster's workers take shards from the start of each queue and the
# workstations' from its end (worker.sh), so they meet in the middle; a
# workstation worker also stops when its machine gets busy. Merge as for
# fixed shards:
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
#
# Settings (environment): DEV_SSH (ssh options, default "-J
# gateway.example.edu"), RUNS (/shared/scratch/me/torch-vulkan-runs),
# LAVAPIPE_SHARDS (192), NVIDIA_SHARDS (64), PBS_JOBS (48), GPU_WORKERS (16),
# WS0_WORKERS (28), WS3_WORKERS (4), WS3_GPU_WORKERS (2), WS2_WORKERS (auto).
set -euo pipefail
CMD=${1:?usage: queue-run.sh start|status NAME [TEST_FILE...]}
NAME=${2:?usage: queue-run.sh start|status NAME [TEST_FILE...]}
shift 2
SSH_OPTS=${DEV_SSH:--J gateway.example.edu}
RUNS=${RUNS:-/shared/scratch/me/torch-vulkan-runs}
LAV="$RUNS/$NAME"
NV="$RUNS/$NAME-nvidia"
TV='~/torch-vulkan/tools/cluster'
on() { ssh -o ServerAliveInterval=30 $SSH_OPTS "me@$1" "${@:2}"; }

case "$CMD" in
  start)
    FILES=${*:-test_ops.py test_torch.py test_binary_ufuncs.py test_unary_ufuncs.py test_nn.py test_linalg.py \
test_reductions.py test_indexing.py test_tensor_creation_ops.py test_shape_ops.py test_type_promotion.py \
test_sort_and_select.py test_view_ops.py}
    queue() {  # queue RUN_DIR COUNT
      echo "mkdir -p $1/queue $1/pbs-logs && echo $2 > $1/shard_count && echo '$FILES' > $1/test_files &&
        for i in \$(seq 0 $(($2 - 1))); do touch $1/queue/shard-\$(printf %04d \$i); done"
    }
    on ws3 "set -e; [ ! -e $LAV ] && [ ! -e $NV ] || { echo '$NAME exists'; exit 1; }
      $(queue "$LAV" "${LAVAPIPE_SHARDS:-192}"); $(queue "$NV" "${NVIDIA_SHARDS:-64}")"
    on headnode "cd ~/torch-vulkan
      qsub -J 0-$((${PBS_JOBS:-48} - 1)) -o $LAV/pbs-logs -v RUN_DIR=$LAV tools/cluster/pbs-lavapipe.sh
      for q in h100 l40s; do
        qsub -q \$q -l select=1:ncpus=16:ngpus=1:mem=32gb -o $NV/pbs-logs \
          -v RUN_DIR=$NV,WORKERS=${GPU_WORKERS:-16} tools/cluster/pbs-gpu.sh
      done"
    on ws0 "bash $TV/pool.sh lavapipe $LAV ${WS0_WORKERS:-28}"
    on ws3 "bash $TV/pool.sh lavapipe $LAV ${WS3_WORKERS:-4}; bash $TV/pool.sh nvidia $NV ${WS3_GPU_WORKERS:-2}"
    on ws2 "bash $TV/pool.sh lavapipe $LAV ${WS2_WORKERS:-auto}"
    ;;
  status)
    on ws3 "for r in $LAV $NV; do
      echo \"\$(basename \$r): \$(ls \$r/queue | wc -l) queued, \$(ls \$r/claimed | wc -l) running, \$(ls \$r/done | wc -l) done\"
      ls \$r/claimed | sed 's/^shard-[0-9]*\.\([^.]*\)\..*/  \1/' | sort | uniq -c
    done"
    ;;
  *)
    echo "unknown command $CMD (start or status)" >&2
    exit 2
    ;;
esac
