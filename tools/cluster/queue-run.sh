#!/bin/bash
# A full run spread over every machine that can take part, from the local
# repo root:
#   tools/cluster/queue-run.sh start NAME [--from PREV [--ids DIR]] [TEST_FILE...]
#   tools/cluster/queue-run.sh status NAME
#
# Two runs, one per driver family, each a queue of small shards on the
# shared file system ($RUNS/NAME for lavapipe, $RUNS/NAME-nvidia):
#   - the cluster (through PBS_HEAD): PBS_JOBS single-core lavapipe jobs,
#     and one job per GPU queue (GPU_QUEUES) of GPU_WORKERS workers each;
#   - backfill on the workstations (pool.sh): one ranked worker per core on
#     each of BACKFILL_HOSTS for the lavapipe run (each running only while
#     its rank fits beside the machine's other users' load, see worker.sh),
#     and GPU_HOST's own GPU on the NVIDIA run if BACKFILL_GPU_WORKERS > 0.
# The cluster's workers take shards from the start of each queue and the
# workstations' from its end, so they meet in the middle. With --from PREV
# (an earlier run's name), the files' tests are collected (into $RUNS/NAME/collected)
# and plan_shards.py plans each queue longest first from
# that run's recorded times ($RUNS/PREV for lavapipe,
# $RUNS/PREV-nvidia or PREV-h100 for the GPUs): one-file shards of about
# PLAN_MINUTES (5) each, numbered from the longest, so the cluster starts the
# longest work first. Without it, shards are fixed (every COUNT-th test of
# each file). Merge as for fixed shards:
#   .venv/bin/python tools/pytorch_tests.py --merge --out $RUNS/NAME
#
# Settings (environment, or tools/cluster/cluster.env: see
# cluster.env.example): CLUSTER_SSH (ssh options), CLUSTER_USER, RUNS (the
# shared run directory), PBS_HEAD, LAVAPIPE_QUEUE, GPU_QUEUES,
# BACKFILL_HOSTS, GPU_HOST, QUEUE_HOST (where the queues are made and read:
# the first backfill host), LAVAPIPE_SHARDS (192), NVIDIA_SHARDS (64),
# PBS_JOBS (48), GPU_WORKERS (16), <HOST>_WORKERS (one per core; HOST in
# capitals), BACKFILL_GPU_WORKERS (0: a small GPU runs out of memory in full
# runs; it stays for quick checks, dev.sh DRIVER=nvidia-host).
set -euo pipefail
CMD=${1:?usage: queue-run.sh start|status NAME [TEST_FILE...]}
NAME=${2:?usage: queue-run.sh start|status NAME [TEST_FILE...]}
shift 2
CFG="$(dirname "$0")/cluster.env"
if [ -f "$CFG" ]; then . "$CFG"; fi
SSH_OPTS=${CLUSTER_SSH:-}
RUNS=${RUNS:?set RUNS (a directory every machine shares): see tools/cluster/cluster.env.example}
BACKFILL_HOSTS=${BACKFILL_HOSTS:-}
PBS_HEAD=${PBS_HEAD:-}
QUEUE_HOST=${QUEUE_HOST:-${BACKFILL_HOSTS%% *}}
QUEUE_HOST=${QUEUE_HOST:-${PBS_HEAD:?set QUEUE_HOST, BACKFILL_HOSTS or PBS_HEAD: see tools/cluster/cluster.env.example}}
LAV="$RUNS/$NAME"
NV="$RUNS/$NAME-nvidia"
TV='~/torch-vulkan/tools/cluster'
on() { ssh -o ServerAliveInterval=30 $SSH_OPTS "${CLUSTER_USER:+$CLUSTER_USER@}$1" "${@:2}"; }

case "$CMD" in
  start)
    PREV=""
    if [ "${1:-}" = --from ]; then
      PREV=${2:?--from needs a run name}
      shift 2
    fi
    IDS=""  # (--ids DIR: plan the tests in DIR's *.jsonl instead of collecting the files')
    if [ -n "$PREV" ] && [ "${1:-}" = --ids ]; then
      IDS=${2:?--ids needs a directory}
      shift 2
    fi
    FILES=${*:-$(grep -v '^#' "$(dirname "$0")/test_files.txt" | tr '\n' ' ')}
    queue() {  # queue RUN_DIR COUNT
      echo "mkdir -p $1/queue $1/pbs-logs && echo $2 > $1/shard_count && echo '$FILES' > $1/test_files &&
        for i in \$(seq 0 $(($2 - 1))); do touch $1/queue/shard-\$(printf %04d \$i); done"
    }
    if [ -n "$PREV" ]; then
      # (the files' tests collected now, on lavapipe, or those in --ids; their times from PREV)
      COLLECT="cd ~/torch-vulkan && VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json LP_NUM_THREADS=1 \
        .venv/bin/python tools/pytorch_tests.py --collect --out $LAV/collected $FILES"
      [ -z "$IDS" ] || COLLECT=true
      P="python3 ~/torch-vulkan/tools/cluster/plan_shards.py --minutes ${PLAN_MINUTES:-5} --ids-from ${IDS:-$LAV/collected}"
      on "$QUEUE_HOST" "set -e; [ ! -e $LAV ] && [ ! -e $NV ] || { echo '$NAME exists'; exit 1; }
        mkdir -p $LAV/pbs-logs $NV/pbs-logs
        $COLLECT
        $P $RUNS/$PREV --out $LAV
        $P \$(ls -d $RUNS/$PREV-nvidia $RUNS/$PREV-h100 2>/dev/null | head -1) --out $NV"
    else
      on "$QUEUE_HOST" "set -e; [ ! -e $LAV ] && [ ! -e $NV ] || { echo '$NAME exists'; exit 1; }
        $(queue "$LAV" "${LAVAPIPE_SHARDS:-192}"); $(queue "$NV" "${NVIDIA_SHARDS:-64}")"
    fi
    if [ -n "$PBS_HEAD" ]; then
      on "$PBS_HEAD" "cd ~/torch-vulkan
        qsub ${LAVAPIPE_QUEUE:+-q $LAVAPIPE_QUEUE} -J 0-$((${PBS_JOBS:-48} - 1)) -o $LAV/pbs-logs -v RUN_DIR=$LAV \
          tools/cluster/pbs-lavapipe.sh
        for q in ${GPU_QUEUES:-}; do
          qsub -q \$q -l select=1:ncpus=16:ngpus=1:mem=32gb -o $NV/pbs-logs \
            -v RUN_DIR=$NV,WORKERS=${GPU_WORKERS:-16} tools/cluster/pbs-gpu.sh
        done"
    fi
    G=${BACKFILL_GPU_WORKERS:-0}
    for h in $BACKFILL_HOSTS; do
      v=$(echo "$h" | tr a-z.- A-Z__)_WORKERS  # (WS1_WORKERS for host ws1)
      w=${!v:-}
      if [ "$h" = "${GPU_HOST:-}" ] && [ "$G" -gt 0 ]; then
        # (the GPU workers rank first: lavapipe workers there rank after them)
        on "$h" "bash $TV/pool.sh nvidia $NV $G 0; bash $TV/pool.sh lavapipe $LAV ${w:-\$((\$(nproc) - $G))} $G"
      else
        on "$h" "bash $TV/pool.sh lavapipe $LAV $w"
      fi
    done
    ;;
  status)
    on "$QUEUE_HOST" "for r in $LAV $NV; do
      echo \"\$(basename \$r): \$(ls \$r/queue | wc -l) queued, \$(ls \$r/claimed | wc -l) running, \$(ls \$r/done | wc -l) done\"
      ls \$r/claimed | sed 's/^shard-[0-9]*\.\([^.]*\)\..*/  \1/' | sort | uniq -c
    done"
    ;;
  *)
    echo "unknown command $CMD (start or status)" >&2
    exit 2
    ;;
esac
