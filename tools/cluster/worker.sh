#!/bin/bash
# Claims shards of a queued run until none are left, running each with
# run-shard.sh:
#   worker.sh DRIVER RUN_DIR
# RUN_DIR (on a file system every machine shares) holds queue/shard-NNNN
# files and either shards/shard-NNNN.ids (the shard's test ids, planned
# longest first by plan_shards.py) or shard_count and test_files (fixed
# shards: every COUNT-th test of each file). A shard is
# claimed by renaming its file into claimed/ (one rename succeeds, on NFS as
# locally), and moved to done/ after its run. Results go to RUN_DIR as with
# fixed shards, so tools/pytorch_tests.py --merge works unchanged.
#
# Cluster workers take shards from the start of the queue. BACKFILL=1 workers
# (the workstations, one per core, ranked by RANK = 0, 1, ...) take them from
# its end, so the two sides meet in the middle, and give way to the
# machine's other users: worker k runs only while
#   others + k + 1 <= cores
# where others is the load average minus our own running test processes, so
# the highest ranks step aside first as other users' load rises and come back
# in order as it falls (resuming needs one spare core more, so they do not
# flap). A worker that loses its slot mid-shard stops the shard's processes
# (SIGSTOP; they keep their memory) and continues them when the slot frees;
# after GIVE_BACK seconds stopped (1200) it kills the shard and returns it to
# the queue for another machine.
set -u
DRIVER=$1
RUN=$2
COUNT=$(cat "$RUN/shard_count")
FILES=$(cat "$RUN/test_files" 2>/dev/null)
HERE=$(cd "$(dirname "$0")" && pwd)
ME="$(hostname -s).$$"
RANK=${RANK:-0}
GIVE_BACK=${GIVE_BACK:-1200}
mkdir -p "$RUN/claimed" "$RUN/done"

# Exit status 0 if this worker's rank fits beside the other users' load; $1
# extra spare cores required (1 to resume, 0 to keep running).
fits() {
  local ours
  # (our running test processes: python -c RUNNER RESULTS.jsonl @LISTING, not stopped)
  ours=$(ps -u "$(id -un)" -o stat=,args= | grep -E '\.jsonl @' | grep -vc '^T')
  awk -v o="$ours" -v r="$RANK" -v c="$(nproc)" -v h="$1" \
    '{ others = $1 - o; if (others < 0) others = 0; exit !(others + r + 1 + h <= c) }' /proc/loadavg
}

queue_empty() { [ -z "$(ls "$RUN/queue" 2>/dev/null)" ]; }

# Runs shard $1 (shard-NNNN) in the foreground.
run_shard() {
  if [ -e "$RUN/shards/$1.ids" ]; then
    LOG_TAG=$1 "$HERE/run-shard.sh" 0 1 "$DRIVER" "$RUN" --ids "$RUN/shards/$1.ids" --name "$1"
  else
    "$HERE/run-shard.sh" "$((10#${1#shard-}))" "$COUNT" "$DRIVER" "$RUN" $FILES
  fi
}

while :; do
  if [ "${BACKFILL:-0}" = 1 ] && ! fits 1; then
    queue_empty && break
    sleep 60
    continue
  fi
  if [ "${BACKFILL:-0}" = 1 ]; then
    next=$(ls "$RUN/queue" 2>/dev/null | sort | tail -n 1)
  else
    next=$(ls "$RUN/queue" 2>/dev/null | sort | head -n 1)
  fi
  [ -z "$next" ] && break
  claimed="$RUN/claimed/$next.$ME"
  mv "$RUN/queue/$next" "$claimed" 2>/dev/null || continue
  if [ "${BACKFILL:-0}" != 1 ]; then
    run_shard "$next"
    mv "$claimed" "$RUN/done/$next"
    continue
  fi
  # (its own process group, so the whole shard can be stopped, continued or killed)
  export -f run_shard
  export RUN DRIVER HERE COUNT FILES
  setsid bash -c 'run_shard "$0"' "$next" &
  pid=$!
  stopped=0
  gave_back=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    if [ "$stopped" = 0 ] && ! fits 0; then
      kill -STOP -- "-$pid" 2>/dev/null
      stopped=$(date +%s)
    elif [ "$stopped" != 0 ] && fits 1; then
      kill -CONT -- "-$pid" 2>/dev/null
      stopped=0
    elif [ "$stopped" != 0 ] && [ $(($(date +%s) - stopped)) -ge "$GIVE_BACK" ]; then
      kill -KILL -- "-$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      mv "$claimed" "$RUN/queue/$next"
      echo "$ME: gave $next back after $GIVE_BACK s stopped"
      gave_back=1
      break
    fi
  done
  wait "$pid" 2>/dev/null
  [ "$gave_back" = 0 ] && mv "$claimed" "$RUN/done/$next"
done
