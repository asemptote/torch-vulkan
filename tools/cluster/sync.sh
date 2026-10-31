#!/bin/bash
# Copies this working tree (the files git tracks or would track, so
# uncommitted changes too) to ~/torch-vulkan on HOST, and optionally builds
# there:
#   tools/cluster/sync.sh HOST [--build] [ssh options, e.g. -J GATEWAY]
# Run from the repo root.
set -euo pipefail
HOST=$1; shift
BUILD=0
if [ "${1:-}" = "--build" ]; then BUILD=1; shift; fi
archive=$(mktemp -t torch-vulkan-src.XXXXXX)
trap 'rm -f "$archive"' EXIT
git ls-files -co --exclude-standard | tar czf "$archive" --force-local -T -
ssh "$@" "$HOST" "mkdir -p ~/torch-vulkan && tar xzf - -C ~/torch-vulkan" < "$archive"
if [ $BUILD = 1 ]; then
  ssh "$@" "$HOST" "bash ~/torch-vulkan/tools/cluster/build.sh; tail -1 \${LOG:-\$HOME/torch-vulkan-build.log}"
fi
