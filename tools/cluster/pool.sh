#!/bin/bash
# Starts N workers (worker.sh) on this machine, detached, at low priority:
#   pool.sh DRIVER RUN_DIR N
# N = auto: the cores not in use (nproc - load average - 2), for shared
# workstations. For lavapipe each worker uses one core (LP_NUM_THREADS=1).
# Workers backfill (BACKFILL=1): they take shards from the end of the queue
# (the cluster's take them from the start) and wait while the machine is busy.
set -u
DRIVER=$1
RUN=$2
N=$3
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$N" = auto ]; then
  load=$(cut -d' ' -f1 /proc/loadavg)
  N=$(awk -v c="$(nproc)" -v l="$load" 'BEGIN { n = int(c - l - 2); print (n > 0 ? n : 0) }')
fi
mkdir -p "$RUN/pbs-logs"
for i in $(seq 1 "$N"); do
  BACKFILL=1 LP_NUM_THREADS=1 setsid nohup "$HERE/worker.sh" "$DRIVER" "$RUN" \
    > "$RUN/pbs-logs/worker.$(hostname -s).$DRIVER.$i.log" 2>&1 < /dev/null &
done
echo "$(hostname -s): $N $DRIVER workers"
