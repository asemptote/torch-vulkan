#!/bin/bash
# The release's hardware check (RELEASING.md, before tagging): the paths that
# CI's lavapipe cannot run, on the GPUs that take them, with one pass/fail
# table at the end. From the repo root of the release candidate:
#
#   tools/cluster/hw-check.sh [intel] [nvidia] [mali]
#
# (default: intel and nvidia, and mali when HW_MALI_HOST is set)
#
#   intel   An Intel GPU under Linux (Mesa's ANV), HW_INTEL_HOST over ssh:
#           16-lane kernels (gemm4s, conv4s, float16 arithmetic) and the
#           handling of i915's 20 s request watchdog
#           (tests/python/test_long_submissions.py, ~2 minutes on an HD 530).
#   nvidia  An NVIDIA GPU with float16 and bfloat16 cooperative matrices
#           (an L40S), as a PBS job on HW_NVIDIA_QUEUE, built on DEV_HOST:
#           gemmcm, float32 matmul precision by bfloat16 parts, split k.
#   mali    An Arm Mali GPU in Termux, HW_MALI_HOST over ssh, built there
#           by tools/build-termux.sh from the intel build's SPIR-V (so it
#           needs intel in the same run): gemm4s and conv4s as Mali's
#           defaults, and no float64.
#
# Each machine gets a copy of this working tree (uncommitted changes too)
# in ~/$HW_DIR, built there (tools/cluster/setup.sh the first time), and
# runs, in that copy:
#   fast paths   the kernels its kind of GPU must take do run there (the tests
#                of a missing feature skip, so a driver that stopped exposing
#                one would otherwise pass unnoticed);
#   tests        python -m pytest tests/python, with TORCH_VK_VALIDATE=1
#                where the SDK's validation layer is in ~/opt (HW_VALIDATE=0:
#                without);
#   long         (intel) TORCH_VK_LONG_TESTS=1 test_long_submissions.py;
#   smoke        tools/release_smoke.py, with the CPU fallback off.
# Logs: $HW_LOGS (default a new directory under /tmp), one file per machine.
#
# Settings (environment, or tools/cluster/cluster.env: see
# cluster.env.example): HW_INTEL_HOST (default LAB_HOST), HW_INTEL_SSH
# (its ssh options), HW_NVIDIA_QUEUE (default the first of GPU_QUEUES),
# HW_NVIDIA_QSUB (qsub resources), HW_MALI_HOST, HW_MALI_SSH, HW_DIR
# (torch-vulkan-hwcheck), HW_BUILD_JOBS (4), HW_VALIDATE (1),
# HW_VALIDATION_LIBS, and the cluster's CLUSTER_SSH, CLUSTER_USER, DEV_HOST
# (where the NVIDIA build runs) and PBS_HEAD.
#
# Internally, `hw-check.sh --run KIND` runs one machine's checks in its copy
# (the PBS job sets HW_RUN and HW_TREE instead), printing one
# "HW|KIND|CHECK|PASS or FAIL|detail" line per check.
set -uo pipefail

# ---- on the machine: one kind's checks, in the copy of the tree ----

# The Vulkan driver manifest named $1 (intel, nvidia), wherever the
# distribution puts it.
icd() {
  ls /usr/share/vulkan/icd.d/"$1"_icd*.json /etc/vulkan/icd.d/"$1"_icd*.json 2>/dev/null | head -1
}

# check NAME COMMAND...: runs the command (output to .hw-check/NAME.log) and
# prints its result line; the detail is the output's last line matching
# DETAIL (a regex).
DETAIL='passed|failed|error|SMOKE|FAST PATHS'
check() {
  local name=$1 log=.hw-check/$1.log status detail
  shift
  "$@" > "$log" 2>&1
  status=$?
  detail=$(grep -aE "$DETAIL" "$log" | tail -1 | tr -d '=|' | sed 's/^ *//; s/ *$//; s/ in [0-9.]*s.*//; s/, [0-9]* warnings*//')
  if [ $status = 0 ]; then status=PASS; else status=FAIL; [ -n "$detail" ] || detail=$(tail -1 "$log" | cut -c1-120); fi
  echo "HW|$KIND|$name|$status|$detail"
}

# The kernels this kind of GPU must run (exit 1 naming any that do not).
FAST_PATHS='
import sys, torch_vulkan as tv
kind, n = sys.argv[1], tv._native
k, caps = n.kernel_subgroup_size, n.capabilities()
want = {
    "intel": {
        "gemm4s (16 lanes)": k("gemm4s_f32") == 16,
        "conv4s (16 lanes)": k("conv4s_f32_d1_r8n1") == 16,
        "float16 arithmetic": bool(caps & 4),
        "gemmsg": k("gemmsg_f32") > 0,
        "float64": bool(caps & 1),
    },
    "nvidia": {
        "gemmcm float16": bool(caps & 16) and k("gemmcm_f16") == 32,
        "gemmcm bfloat16 (bf16parts)": bool(caps & 32) and k("gemmcm_bf16") == 32,
        "split k (compute units known)": n.compute_units() > 0,
        "gemmsg": k("gemmsg_f32") > 0,
        "float64": bool(caps & 1),
    },
    "mali": {
        "Mali": "Mali" in tv.device_name(),
        "gemm4s (16 lanes)": k("gemm4s_f32") == 16,
        "conv4s (16 lanes)": k("conv4s_f32_d1_r8n1") == 16,
        "float16 arithmetic": bool(caps & 4),
    },
}[kind]
missing = [p for p, ok in want.items() if not ok]
print(tv.device_name())
print("FAST PATHS " + ("missing: " + ", ".join(missing) if missing else "all " + str(len(want))))
sys.exit(1 if missing else 0)
'

run_checks() {
  KIND=$1
  if [ -f tools/cluster/cluster.env ]; then . tools/cluster/cluster.env; fi  # (HW_VALIDATION_LIBS)
  unset TORCH_VK_CPU_FALLBACK TORCH_VK_STRICT  # (the defaults: the CPU fallback off, as for users)
  mkdir -p .hw-check
  local py=.venv/bin/python
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
  case "$KIND" in
    intel | nvidia)
      VK_DRIVER_FILES=$(icd "$KIND")
      [ -n "$VK_DRIVER_FILES" ] || { echo "HW|$KIND|fast paths|FAIL|no $KIND Vulkan driver manifest"; return; }
      export VK_DRIVER_FILES
      ;;
    mali) py=python; export PYTHONPATH=$PWD/python ;;  # (Termux: Android's loader, the build in place)
    *) echo "unknown kind $KIND (intel, nvidia or mali)" >&2; exit 2 ;;
  esac
  # (a log line naming the device, for the table)
  echo "HW|$KIND|device|INFO|$($py -c 'import torch_vulkan as tv; print(tv.device_name())' 2>&1 | tail -1)"
  check "fast paths" "$py" -c "$FAST_PATHS" "$KIND"
  # The validation layer where the SDK is in ~/opt and loads, and only that
  # one (VK_LAYER_PATH: a distribution's copy as well would be a duplicate,
  # which the loader reports as a warning, failing the tests). It needs a newer
  # libstdc++ than some distributions have: HW_VALIDATION_LIBS, a directory
  # with one, goes on LD_LIBRARY_PATH for it); otherwise the tests run
  # without it, and the table says so.
  local layer validate=0
  local libs=${HW_VALIDATION_LIBS:-}${HW_VALIDATION_LIBS:+${LD_LIBRARY_PATH:+:}}${LD_LIBRARY_PATH:-}
  layer=$(ls -d "$HOME"/opt/*/x86_64/share/vulkan/explicit_layer.d 2>/dev/null | tail -1)
  if [ "${HW_VALIDATE:-1}" != 0 ] && [ -n "$layer" ] && LD_LIBRARY_PATH=$libs VK_LAYER_PATH=$layer \
    TORCH_VK_VALIDATE=1 "$py" -c 'import torch_vulkan as tv; tv.device_name()' > .hw-check/layer.log 2>&1; then
    validate=1
  fi
  if [ $validate = 1 ]; then
    LD_LIBRARY_PATH=$libs VK_LAYER_PATH=$layer TORCH_VK_VALIDATE=1 \
      check tests "$py" -m pytest -q -rs -p no:cacheprovider tests/python | sed 's/$/, validated/'
  else
    check tests "$py" -m pytest -q -rs -p no:cacheprovider tests/python | sed 's/$/, no validation layer/'
  fi
  if [ "$KIND" = intel ]; then
    TORCH_VK_LONG_TESTS=1 check long "$py" -m pytest -q -rs -p no:cacheprovider tests/python/test_long_submissions.py
  fi
  check smoke "$py" tools/release_smoke.py
}

if [ -n "${HW_RUN:-}" ]; then  # (the PBS job)
  cd "${HW_TREE:?}" || exit 1
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
  run_checks "$HW_RUN"
  exit 0
fi
if [ "${1:-}" = --run ]; then
  run_checks "${2:?usage: hw-check.sh --run intel|nvidia|mali}"
  exit 0
fi

# ---- on the local machine: copy, build, run, tabulate ----

CFG="$(dirname "$0")/cluster.env"
if [ -f "$CFG" ]; then . "$CFG"; fi
HW_DIR=${HW_DIR:-torch-vulkan-hwcheck}
INTEL=${HW_INTEL_HOST:-${LAB_HOST:-}}
Q=${HW_NVIDIA_QUEUE:-${GPU_QUEUES%% *}}
QSUB=${HW_NVIDIA_QSUB:--l select=1:ncpus=8:ngpus=1:mem=32gb -l walltime=01:30:00}
MALI=${HW_MALI_HOST:-}
LOGS=${HW_LOGS:-$(mktemp -d -t tv-hw-check.XXXXXX)}
mkdir -p "$LOGS"
KINDS=("$@")
[ ${#KINDS[@]} -gt 0 ] || KINDS=(intel nvidia ${MALI:+mali})
want() { [[ " ${KINDS[*]} " == *" $1 "* ]]; }
AT=${CLUSTER_USER:+$CLUSTER_USER@}
cluster() { ssh -o ServerAliveInterval=30 $CLUSTER_SSH "$AT$1" "${@:2}"; }
intel() { ssh -o ServerAliveInterval=30 ${HW_INTEL_SSH:-} "$INTEL" "$@"; }
mali() { ssh -o ServerAliveInterval=30 ${HW_MALI_SSH:-} "$MALI" "$@"; }

# The build of ~/$HW_DIR on a host (setup.sh the first time, which builds
# too), ending with "build exit N".
BUILD="cd ~/$HW_DIR && export CARGO_BUILD_JOBS=${HW_BUILD_JOBS:-4} && if [ -x .venv/bin/python ]; then TORCH_VULKAN=\$HOME/$HW_DIR bash tools/cluster/build.sh; \
  else bash tools/cluster/setup.sh > \$HOME/$HW_DIR-setup.log 2>&1; fi; echo \"build exit \$?\""

for k in "${KINDS[@]}"; do
  case "$k" in
    intel) [ -n "$INTEL" ] || { echo "set HW_INTEL_HOST: see tools/cluster/cluster.env.example" >&2; exit 2; } ;;
    nvidia) [ -n "${DEV_HOST:-}" ] && [ -n "${PBS_HEAD:-}" ] && [ -n "$Q" ] ||
      { echo "set DEV_HOST, PBS_HEAD and HW_NVIDIA_QUEUE: see tools/cluster/cluster.env.example" >&2; exit 2; } ;;
    mali) [ -n "$MALI" ] || { echo "set HW_MALI_HOST: see tools/cluster/cluster.env.example" >&2; exit 2; }
      want intel || { echo "mali needs intel in the same run (its SPIR-V)" >&2; exit 2; } ;;
    *) echo "unknown machine $k (intel, nvidia or mali)" >&2; exit 2 ;;
  esac
done

echo "logs: $LOGS"
# The copies (sync.sh: the files git tracks or would track) and the builds,
# side by side; a failed build fails that machine's row.
pids=()
if want intel; then
  { TORCH_VULKAN_DIR=$HW_DIR "$(dirname "$0")/sync.sh" "$INTEL" -o ServerAliveInterval=30 ${HW_INTEL_SSH:-} &&
    intel "$BUILD"; } > "$LOGS/intel.build.log" 2>&1 &
  pids+=($!)
fi
if want nvidia; then
  { TORCH_VULKAN_DIR=$HW_DIR "$(dirname "$0")/sync.sh" "$DEV_HOST" && cluster "$DEV_HOST" "$BUILD"; } \
    > "$LOGS/nvidia.build.log" 2>&1 &
  pids+=($!)
fi
if want mali; then
  TORCH_VULKAN_DIR=$HW_DIR "$(dirname "$0")/sync.sh" "$MALI" -o ServerAliveInterval=30 ${HW_MALI_SSH:-} \
    > "$LOGS/mali.build.log" 2>&1
fi
wait "${pids[@]}"
built() { grep -q "^build exit 0" "$LOGS/$1.build.log"; }

# The checks, side by side.
pids=()
if want intel; then
  if built intel; then
    intel "cd ~/$HW_DIR && bash tools/cluster/hw-check.sh --run intel" > "$LOGS/intel.log" 2>&1 &
    pids+=($!)
  else
    echo "HW|intel|build|FAIL|$(tail -1 "$LOGS/intel.build.log")" > "$LOGS/intel.log"
  fi
fi
if want nvidia; then
  if built nvidia; then
    # (qsub -W block=true returns when the job ends; PBS then copies its output back)
    out="~/$HW_DIR/.hw-check-nvidia.out"
    cluster "$PBS_HEAD" "rm -f $out; qsub -q $Q $QSUB -N tv-hw-check -W block=true -j oe -o $out \
      -v HW_RUN=nvidia,HW_TREE=\$HOME/$HW_DIR ~/$HW_DIR/tools/cluster/hw-check.sh > /dev/null
      for i in \$(seq 60); do [ -f $out ] && break; sleep 1; done; cat $out" > "$LOGS/nvidia.log" 2>&1 &
    pids+=($!)
  else
    echo "HW|nvidia|build|FAIL|$(tail -1 "$LOGS/nvidia.build.log")" > "$LOGS/nvidia.log"
  fi
fi
if want mali; then
  {
    if built intel; then
      # (Termux has no slangc: the intel build's kernels, as the release's SPIR-V bundle)
      # (wherever Cargo put the build script's output: the newest kernels.rs, as release.yml finds it)
      intel "cd \$(dirname \$(find /tmp/\$USER-$HW_DIR-target -path '*vkcore*' -name kernels.rs -printf '%T@ %p\n' |
        sort -n | tail -1 | cut -d' ' -f2)) && tar cf - *.spv" |
        mali "rm -rf ~/$HW_DIR/.spirv && mkdir -p ~/$HW_DIR/.spirv && tar xf - -C ~/$HW_DIR/.spirv" &&
        mali "cd ~/$HW_DIR && JOBS=\${JOBS:-3} tools/build-termux.sh .spirv > .hw-check-build.log 2>&1" &&
        mali "cd ~/$HW_DIR && bash tools/cluster/hw-check.sh --run mali" ||
        echo "HW|mali|build|FAIL|SPIR-V copy or tools/build-termux.sh failed (~/$HW_DIR/.hw-check-build.log there)"
    else
      echo "HW|mali|build|FAIL|no intel build to take the SPIR-V from"
    fi
  } > "$LOGS/mali.log" 2>&1 &
  pids+=($!)
fi
wait "${pids[@]}"

# The table: one row per machine, one column per check.
COLS=("fast paths" tests long smoke)
declare -A cell device
fail=0
for k in "${KINDS[@]}"; do
  seen=0
  while IFS='|' read -r _ kind name status detail; do
    seen=1
    if [ "$name" = device ]; then device[$k]=$detail; continue; fi
    [ "$status" = PASS ] || fail=1
    if [ "$name" = build ]; then cell[$k,tests]="FAIL (build: $detail)"; continue; fi
    cell[$k,$name]="$status${detail:+ ($detail)}"
  done < <(grep -a '^HW|' "$LOGS/$k.log")
  [ $seen = 1 ] || { cell[$k,tests]="FAIL (no results: see $LOGS/$k.log)"; fail=1; }
done
echo
echo "| machine | device | fast paths | tests/python | long submissions | release smoke |"
echo "|---|---|---|---|---|---|"
for k in "${KINDS[@]}"; do
  row="| $k | ${device[$k]:-?} |"
  for c in "${COLS[@]}"; do
    v=${cell[$k,$c]:-}
    if [ -z "$v" ] && { [ "$c" != long ] || [ "$k" = intel ]; }; then v="not run"; fail=1; fi
    row+=" ${v:--} |"
  done
  echo "$row"
done
echo
[ $fail = 0 ] && echo "hardware check passed" || echo "hardware check FAILED (logs: $LOGS)"
exit $fail
