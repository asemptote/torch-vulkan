#!/bin/bash
# Claims shards of a queued run until none are left, running each with
# run-shard.sh:
#   worker.sh DRIVER RUN_DIR
# RUN_DIR (on a file system every machine shares) holds queue/shard-NNNN
# files, shard_count and test_files (queue-run.sh writes them). A shard is
# claimed by renaming its file into claimed/ (one rename succeeds, on NFS as
# locally), and moved to done/ after its run. Results go to RUN_DIR as with
# fixed shards, so tools/pytorch_tests.py --merge works unchanged.
# Cluster workers take shards from the start of the queue; BACKFILL=1
# workers (the workstations) from its end, so the two sides meet in the
# middle without competing for the same shards. A backfill worker also stops
# when its machine gets busy (load average above its cores): its users come
# first, and the cluster takes the rest.
set -u
DRIVER=$1
RUN=$2
COUNT=$(cat "$RUN/shard_count")
FILES=$(cat "$RUN/test_files")
HERE=$(cd "$(dirname "$0")" && pwd)
ME="$(hostname -s).$$"
mkdir -p "$RUN/claimed" "$RUN/done"
while :; do
  if [ "${BACKFILL:-0}" = 1 ] && awk -v c="$(nproc)" '{ exit !($1 > c) }' /proc/loadavg; then
    echo "$ME: machine busy, stopping"
    break
  fi
  if [ "${BACKFILL:-0}" = 1 ]; then
    next=$(ls "$RUN/queue" 2>/dev/null | sort | tail -n 1)
  else
    next=$(ls "$RUN/queue" 2>/dev/null | sort | head -n 1)
  fi
  [ -z "$next" ] && break
  mv "$RUN/queue/$next" "$RUN/claimed/$next.$ME" 2>/dev/null || continue
  shard=$((10#${next#shard-}))
  "$HERE/run-shard.sh" "$shard" "$COUNT" "$DRIVER" "$RUN" $FILES
  mv "$RUN/claimed/$next.$ME" "$RUN/done/$next"
done
