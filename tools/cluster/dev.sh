#!/bin/bash
# A second copy of the repo on the cluster for experiments ("the dev copy"),
# with its own venv and Cargo target directory, so it can be rebuilt while
# a full run uses ~/torch-vulkan. Run from the repo root on the local machine:
#
#   tools/cluster/dev.sh init             one-time: copy the tree and ~/torch-vulkan/.venv, build
#   tools/cluster/dev.sh sync             copy the working tree (uncommitted changes too)
#   tools/cluster/dev.sh build            sync, rebuild; prints the time and "build exit N"
#   tools/cluster/dev.sh run ID...        run test ids through the runner
#   tools/cluster/dev.sh rss ID...        each id alone: outcome, seconds, peak memory
#   tools/cluster/dev.sh pt FILE [ARGS]   pytest FILE on vk with tracebacks (tools/dev/pt.py)
#   tools/cluster/dev.sh py SCRIPT [ARGS] run a local Python script there
#
# run, rss, pt and py use lavapipe on DEV_HOST; with DRIVER=nvidia they run
# on the NVIDIA driver instead, as a PBS job on a GPU node (DEV_GPU_QSUB),
# submitted through DEV_HEAD and waited for, with its output printed; with
# DRIVER=nvidia-host, on DEV_HOST's own NVIDIA GPU (ws3: a Tesla M4, 4 GB),
# directly: no queue, for quick checks of the NVIDIA driver.
#
# ID is a pytest node id (test_linalg.py::TestLinalgPRIVATEUSE1::test_foo_vk);
# run and pt pass the runner's / pytest's remaining options through.
# Settings (environment): DEV_HOST (ws3), DEV_HEAD (headnode), DEV_SSH (ssh
# options, default "-J gateway.example.edu"), DEV_DIR
# (/shared/scratch/me/tv-dev, on a file system the GPU nodes share),
# LP_NUM_THREADS (lavapipe threads, 4), TIMEOUT (seconds per test, 3600).
set -euo pipefail
HOST=${DEV_HOST:-ws3}
HEAD=${DEV_HEAD:-headnode}
SSH_OPTS=${DEV_SSH:--J gateway.example.edu}
DIR=${DEV_DIR:-/shared/scratch/me/tv-dev}
THREADS=${LP_NUM_THREADS:-4}
TIMEOUT=${TIMEOUT:-3600}
DRIVER=${DRIVER:-lavapipe}
GPU_QSUB=${DEV_GPU_QSUB:--q h100 -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=02:00:00}
CMD=${1:?usage: dev.sh init|sync|build|run|rss|pt|py ...}; shift

remote() { ssh -o ServerAliveInterval=30 $SSH_OPTS "$HOST" "$@"; }

case "$DRIVER" in
  lavapipe) ICD=lvp ;;  # Mesa's file name for lavapipe
  nvidia | nvidia-host) ICD=nvidia ;;
  *) echo "unknown DRIVER $DRIVER (lavapipe, nvidia or nvidia-host)" >&2; exit 2 ;;
esac

# Environment for running in the dev copy (a shell prefix).
ENV="cd $DIR && export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/${ICD}_icd.x86_64.json \
LP_NUM_THREADS=$THREADS OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_TEST_DIR=\$HOME/pytorch-src/test"

# Runs shell command $1 in the dev copy on DRIVER: through ssh on HOST, or
# as a job script on a GPU node (the same text: ssh's remote shell and the
# job's bash each parse it once).
on_driver() {
  if [ "$DRIVER" = nvidia ]; then
    printf '#!/bin/bash\n%s\n' "$ENV; $1" | remote "cat > $DIR/.dev-gpu-job.sh"
    # (PBS copies the output file back after qsub -W block returns.)
    ssh -o ServerAliveInterval=30 $SSH_OPTS "$HEAD" "rm -f $DIR/.dev-gpu-out.txt; qsub $GPU_QSUB -N tv-dev -W block=true -j oe \
      -o $DIR/.dev-gpu-out.txt $DIR/.dev-gpu-job.sh > /dev/null
      for i in \$(seq 60); do [ -f $DIR/.dev-gpu-out.txt ] && break; sleep 1; done; cat $DIR/.dev-gpu-out.txt"
  else
    remote "$ENV; $1"
  fi
}

sync_tree() {
  git ls-files -co --exclude-standard | tar czf - --force-local -T - | remote "mkdir -p $DIR && tar xzf - -C $DIR"
}

# Writes the arguments, one per line, to a file in the dev copy; prints its name.
ids_file() {
  printf '%s\n' "$@" | remote "mkdir -p $DIR/runs && f=\$(mktemp $DIR/runs/ids.XXXXXX) && cat > \$f && echo \$f"
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
    on_driver "nice -n 19 .venv/bin/python tools/pytorch_tests.py --out $DIR/runs --ids $f --name dev-$DRIVER \
      ${opts[*]:-} -- --timeout $TIMEOUT --timeout-method thread 2>&1 | tail -40;
      cat $DIR/runs/dev-$DRIVER.failures.txt 2>/dev/null | cut -c1-400"
    ;;
  rss)
    for id in "$@"; do
      f=$(ids_file "$id")
      on_driver "s=\$(date +%s); r=\$(nice -n 19 /usr/bin/time -v .venv/bin/python tools/pytorch_tests.py \
        --out $DIR/runs --ids $f --name rss-$DRIVER -- --timeout $TIMEOUT --timeout-method thread 2>&1 |
        grep -E '^total|Maximum resident' | tr -s ' \t\n' ' '); echo \"$id \$(( \$(date +%s) - s )) s: \$r\""
    done
    ;;
  pt)
    on_driver "nice -n 19 timeout $TIMEOUT .venv/bin/python tools/dev/pt.py $(printf '%q ' "$@")"
    ;;
  py)
    script=${1:?usage: dev.sh py SCRIPT [ARGS]}; shift
    remote "cat > $DIR/.dev-script.py" < "$script"
    on_driver "nice -n 19 timeout $TIMEOUT .venv/bin/python $DIR/.dev-script.py $(printf '%q ' "$@")"
    ;;
  *)
    echo "unknown command $CMD (init, sync, build, run, rss, pt or py)" >&2
    exit 2
    ;;
esac
