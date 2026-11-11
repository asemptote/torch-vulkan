#!/bin/bash
# Copies this working tree (the files git tracks or would track, so
# uncommitted changes too, and the site's tools/cluster/cluster.env) to
# ~/torch-vulkan on HOST, and optionally builds there:
#   tools/cluster/sync.sh HOST [--build] [ssh options, e.g. -J GATEWAY]
# Without ssh options, cluster.env's CLUSTER_SSH (and CLUSTER_USER). Run
# from the repo root.
set -euo pipefail
HOST=$1; shift
BUILD=0
if [ "${1:-}" = "--build" ]; then BUILD=1; shift; fi
if [ $# = 0 ]; then
  if [ -f tools/cluster/cluster.env ]; then . tools/cluster/cluster.env; fi
  set -- ${CLUSTER_SSH:-}
  case "$HOST" in *@*) ;; *) HOST=${CLUSTER_USER:+$CLUSTER_USER@}$HOST ;; esac
fi
archive=$(mktemp -t torch-vulkan-src.XXXXXX)
trap 'rm -f "$archive"' EXIT
{ git ls-files -co --exclude-standard; ls tools/cluster/cluster.env 2>/dev/null || true; } |
  tar czf "$archive" --force-local -T -
# (-m: the files dated now, not with their local times, which can be older
# than the host's last build: cargo would then see nothing to rebuild)
ssh "$@" "$HOST" "mkdir -p ~/torch-vulkan && tar xzmf - -C ~/torch-vulkan" < "$archive"
if [ $BUILD = 1 ]; then
  ssh "$@" "$HOST" "bash ~/torch-vulkan/tools/cluster/build.sh; tail -1 \${LOG:-\$HOME/torch-vulkan-build.log}"
fi
