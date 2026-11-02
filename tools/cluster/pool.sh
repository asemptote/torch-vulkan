#!/bin/bash
# Starts backfill workers (worker.sh, BACKFILL=1) on this machine, detached,
# at low priority:
#   pool.sh DRIVER RUN_DIR [N [FIRST_RANK]]
# N defaults to one per core (less FIRST_RANK). Worker i gets RANK
# FIRST_RANK + i (another pool on the machine holds the ranks before): each runs only
# while its rank fits in the cores the machine's other users leave free, so
# the pool shrinks and grows with their load (see worker.sh). Backfill
# workers take shards from the end of the queue (the cluster's take them from
# the start). For lavapipe each worker uses one core (LP_NUM_THREADS=1).
set -u
DRIVER=$1
RUN=$2
FIRST=${4:-0}
N=${3:-$(($(nproc) - FIRST))}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RUN/pbs-logs"
for i in $(seq "$FIRST" $((FIRST + N - 1))); do
  RANK=$i BACKFILL=1 LP_NUM_THREADS=1 setsid nohup "$HERE/worker.sh" "$DRIVER" "$RUN" \
    > "$RUN/pbs-logs/worker.$(hostname -s).$DRIVER.$i.log" 2>&1 < /dev/null &
done
echo "$(hostname -s): $N $DRIVER workers"
