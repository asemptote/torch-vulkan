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
#
# The pool also starts the machine's CPU monitor (one per user and machine,
# while any worker runs): every 30 s it writes the cores in use by processes
# other than our test shards (top over 5 s) to /tmp/tv-others.USER, which the
# workers compare their ranks against. (The load average overstates it:
# processes waiting on the shared file system count too.)
set -u
DRIVER=$1
RUN=$2
FIRST=${4:-0}
N=${3:-$(($(nproc) - FIRST))}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$RUN/pbs-logs"

ME=$(id -un)
OTHERS=/tmp/tv-others.$ME
monitor() {
  sleep 5  # (the workers start first)
  while pgrep -u "$ME" -f "tools/cluster/worker.sh" > /dev/null; do
    ours=$(ps -u "$ME" -o pid=,args= | grep -E '\.jsonl @|run-shard\.sh|worker\.sh|pytorch_tests\.py' | awk '{print $1}' | tr '\n' ' ')
    top -b -n 2 -d 5 -w 200 | awk -v ours=" $ours " '/^ *PID/ { n++; next }
      n == 2 && index(ours, " " $1 " ") == 0 { s += $9 } END { printf "%.2f\n", s / 100 }' > "$OTHERS.tmp" &&
      mv "$OTHERS.tmp" "$OTHERS"
    sleep 25
  done
}
if ! pgrep -u "$ME" -f "tv-cpu-monitor" > /dev/null; then
  export -f monitor
  export ME OTHERS
  setsid nohup bash -c 'monitor # tv-cpu-monitor' > /dev/null 2>&1 < /dev/null &
fi

for i in $(seq "$FIRST" $((FIRST + N - 1))); do
  RANK=$i BACKFILL=1 LP_NUM_THREADS=1 setsid nohup "$HERE/worker.sh" "$DRIVER" "$RUN" \
    > "$RUN/pbs-logs/worker.$(hostname -s).$DRIVER.$i.log" 2>&1 < /dev/null &
done
echo "$(hostname -s): $N $DRIVER workers"
