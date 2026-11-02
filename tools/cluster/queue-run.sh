#!/bin/bash
# A full run spread over every machine that can take part, from the local
# repo root:
#   tools/cluster/queue-run.sh start NAME [--from PREV] [TEST_FILE...]
#   tools/cluster/queue-run.sh status NAME
#
# Two runs, one per driver family, each a queue of small shards on the
# shared file system ($RUNS/NAME for lavapipe, $RUNS/NAME-nvidia):
#   - the cluster: PBS_JOBS single-core lavapipe jobs, and an H100 and an
#     L40S job (separate queues) of GPU_WORKERS workers each;
#   - backfill on the workstations (pool.sh): one ranked worker per core on
#     ws0, ws2 and ws3 for the lavapipe run (each running only while
#     its rank fits beside the machine's other users' load, see worker.sh),
#     and ws3's Tesla M4 on the NVIDIA run.
# The cluster's workers take shards from the start of each queue and the
# workstations' from its end, so they meet in the middle. With --from PREV
# (an earlier run's name), plan_shards.py plans each queue longest first from
# that run's tests and their recorded times ($RUNS/PREV for lavapipe,
# $RUNS/PREV-nvidia or PREV-h100 for the GPUs): one-file shards of about
# PLAN_MINUTES (5) each, numbered from the longest, so the cluster starts the
# longest work first. Without it, shards are fixed (every COUNT-th test of
# each file). Merge as for fixed shards:
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
#
# Settings (environment): DEV_SSH (ssh options, default "-J
# gateway.example.edu"), RUNS (/shared/scratch/me/torch-vulkan-runs),
# LAVAPIPE_SHARDS (192), NVIDIA_SHARDS (64), PBS_JOBS (48), GPU_WORKERS (16),
# WS0_WORKERS, WS2_WORKERS, WS3_WORKERS (one per core), WS3_GPU_WORKERS (2).
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
    PREV=""
    if [ "${1:-}" = --from ]; then
      PREV=${2:?--from needs a run name}
      shift 2
    fi
    FILES=${*:-test_ops.py test_torch.py test_binary_ufuncs.py test_unary_ufuncs.py test_nn.py test_linalg.py \
test_reductions.py test_indexing.py test_tensor_creation_ops.py test_shape_ops.py test_type_promotion.py \
test_sort_and_select.py test_view_ops.py}
    queue() {  # queue RUN_DIR COUNT
      echo "mkdir -p $1/queue $1/pbs-logs && echo $2 > $1/shard_count && echo '$FILES' > $1/test_files &&
        for i in \$(seq 0 $(($2 - 1))); do touch $1/queue/shard-\$(printf %04d \$i); done"
    }
    if [ -n "$PREV" ]; then
      P="python3 ~/torch-vulkan/tools/cluster/plan_shards.py --minutes ${PLAN_MINUTES:-5}"
      on ws3 "set -e; [ ! -e $LAV ] && [ ! -e $NV ] || { echo '$NAME exists'; exit 1; }
        mkdir -p $LAV/pbs-logs $NV/pbs-logs
        $P $RUNS/$PREV --out $LAV
        $P \$(ls -d $RUNS/$PREV-nvidia $RUNS/$PREV-h100 2>/dev/null | head -1) --out $NV"
    else
      on ws3 "set -e; [ ! -e $LAV ] && [ ! -e $NV ] || { echo '$NAME exists'; exit 1; }
        $(queue "$LAV" "${LAVAPIPE_SHARDS:-192}"); $(queue "$NV" "${NVIDIA_SHARDS:-64}")"
    fi
    on headnode "cd ~/torch-vulkan
      qsub -J 0-$((${PBS_JOBS:-48} - 1)) -o $LAV/pbs-logs -v RUN_DIR=$LAV tools/cluster/pbs-lavapipe.sh
      for q in h100 l40s; do
        qsub -q \$q -l select=1:ncpus=16:ngpus=1:mem=32gb -o $NV/pbs-logs \
          -v RUN_DIR=$NV,WORKERS=${GPU_WORKERS:-16} tools/cluster/pbs-gpu.sh
      done"
    on ws0 "bash $TV/pool.sh lavapipe $LAV ${WS0_WORKERS:-}"
    on ws2 "bash $TV/pool.sh lavapipe $LAV ${WS2_WORKERS:-}"
    # (ws3's GPU workers rank first: lavapipe workers there rank after them)
    G=${WS3_GPU_WORKERS:-2}
    on ws3 "bash $TV/pool.sh nvidia $NV $G 0; bash $TV/pool.sh lavapipe $LAV ${WS3_WORKERS:-\$((\$(nproc) - $G))} $G"
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
