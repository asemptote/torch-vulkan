#!/bin/bash
# A second copy of the repo on the cluster for experiments ("the dev copy"),
# with its own venv and Cargo target directory, so it can be rebuilt while
# a full run uses ~/torch-vulkan. Run from the repo root on the local machine:
#
#   tools/cluster/dev.sh init             one-time: copy the tree and ~/torch-vulkan/.venv, build
#   tools/cluster/dev.sh sync             copy the working tree (uncommitted changes too)
#   tools/cluster/dev.sh build            sync, rebuild; prints the time and "build exit N"
#   tools/cluster/dev.sh run ID...        run test ids through the runner on lavapipe
#   tools/cluster/dev.sh rss ID...        each id alone: outcome, seconds, peak memory
#   tools/cluster/dev.sh pt FILE [ARGS]   pytest FILE on vk with tracebacks (tools/dev/pt.py)
#   tools/cluster/dev.sh py SCRIPT [ARGS] run a local Python script there on lavapipe
#
# ID is a pytest node id (test_linalg.py::TestLinalgPRIVATEUSE1::test_foo_vk);
# run and pt pass the runner's / pytest's remaining options through.
# Settings (environment): DEV_HOST (ws3), DEV_SSH (ssh options, default
# "-J gateway.example.edu"), DEV_DIR (/shared/scratch/me/tv-dev),
# LP_NUM_THREADS (lavapipe threads, 4), TIMEOUT (seconds per test, 3600).
set -euo pipefail
HOST=${DEV_HOST:-ws3}
SSH_OPTS=${DEV_SSH:--J gateway.example.edu}
DIR=${DEV_DIR:-/shared/scratch/me/tv-dev}
THREADS=${LP_NUM_THREADS:-4}
TIMEOUT=${TIMEOUT:-3600}
CMD=${1:?usage: dev.sh init|sync|build|run|rss|pt|py ...}; shift

remote() { ssh $SSH_OPTS "$HOST" "$@"; }

# Environment for running on lavapipe in the dev copy (a remote shell prefix).
LAVAPIPE="cd $DIR && export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
LP_NUM_THREADS=$THREADS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_TEST_DIR=\$HOME/pytorch-src/test"

sync_tree() {
  git ls-files -co --exclude-standard | tar czf - --force-local -T - | remote "mkdir -p $DIR && tar xzf - -C $DIR"
}

# Writes the arguments, one per line, to a file on the host; prints its name.
ids_file() {
  printf '%s\n' "$@" | remote 'f=$(mktemp /tmp/$USER-dev-ids.XXXXXX); cat > $f; echo $f'
}

case "$CMD" in
  init)
    sync_tree
    # (The venv's scripts name its interpreter by path: point them here.)
    remote "set -e; rm -rf $DIR/.venv; cp -a \$HOME/torch-vulkan/.venv $DIR/.venv
      grep -rlI \"\$HOME/torch-vulkan/.venv\" $DIR/.venv/bin | xargs -r sed -i \"s#\$HOME/torch-vulkan/.venv#$DIR/.venv#g\""
    "$0" build
    ;;
  sync)
    sync_tree
    ;;
  build)
    sync_tree
    remote "s=\$(date +%s); TORCH_VULKAN=$DIR LOG=$DIR/build.log CARGO_TARGET_DIR=/tmp/\$USER-tv-dev-target \
      bash $DIR/tools/cluster/build.sh; code=\$?; echo \"build \$(( \$(date +%s) - s )) s\"
      [ \$code = 0 ] || grep -E 'error|Error' $DIR/build.log | head -20; tail -1 $DIR/build.log"
    ;;
  run)
    ids=() opts=()
    for a in "$@"; do if [[ $a == *::* ]]; then ids+=("$a"); else opts+=("$a"); fi; done
    f=$(ids_file "${ids[@]}")
    remote "$LAVAPIPE; nice -n 19 .venv/bin/python tools/pytorch_tests.py --out $DIR/runs --ids $f \
      --name dev ${opts[*]:-} -- --timeout $TIMEOUT --timeout-method thread 2>&1 | tail -40;
      cat $DIR/runs/dev.failures.txt 2>/dev/null | cut -c1-400"
    ;;
  rss)
    for id in "$@"; do
      f=$(ids_file "$id")
      remote "$LAVAPIPE; s=\$(date +%s); r=\$(nice -n 19 /usr/bin/time -v .venv/bin/python tools/pytorch_tests.py \
        --out $DIR/runs --ids $f --name rss -- --timeout $TIMEOUT --timeout-method thread 2>&1 |
        grep -E '^total|Maximum resident' | tr -s ' \t\n' ' '); echo \"$id \$(( \$(date +%s) - s )) s: \$r\""
    done
    ;;
  pt)
    remote "$LAVAPIPE; nice -n 19 timeout $TIMEOUT .venv/bin/python tools/dev/pt.py $(printf '%q ' "$@")"
    ;;
  py)
    script=${1:?usage: dev.sh py SCRIPT [ARGS]}; shift
    remote "cat > /tmp/\$USER-dev-script.py" < "$script"
    remote "$LAVAPIPE; nice -n 19 timeout $TIMEOUT .venv/bin/python /tmp/\$USER-dev-script.py $(printf '%q ' "$@")"
    ;;
  *)
    echo "unknown command $CMD (init, sync, build, run, rss, pt or py)" >&2
    exit 2
    ;;
esac
