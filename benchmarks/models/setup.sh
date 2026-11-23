#!/bin/bash
# The suites' sources and dependencies, for run.py and sweep.py:
#   benchmarks/models/setup.sh DIR PYTHON [PYTHON...]
# DIR (large: keep it out of the repo) gets pytorch-bench/, PyTorch's
# benchmarks/dynamo/ and CI pins at the commit of PYTHON's torch, and
# torchbench/, pytorch/benchmark at PyTorch's pin. Each PYTHON's environment
# (the vk one, whose torch is the CPU build with torch_vulkan, and one with
# the same torch release built for CUDA) gets the suites' pins: transformers
# (huggingface-requirements.txt), timm (timm.txt), torchvision, and
# TorchBench's requirements and models (TORCHBENCH_MODELS, default every
# model in all_torchbench_models_list.txt but the detectron2 ones, which need
# detectron2 built). Uses uv (UV_CACHE_DIR: keep it on the large volume too).
# Then: export PYTORCH_BENCH_DIR=DIR/pytorch-bench TORCHBENCH_DIR=DIR/torchbench
set -euo pipefail
DIR=$(mkdir -p "$1" && cd "$1" && pwd); shift
PY0=$1
COMMIT=$("$PY0" -c "import torch; print(torch.version.git_version)")
if [ "$(git -C "$DIR/pytorch-bench" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]; then
  rm -rf "$DIR/pytorch-bench"
  git init -q "$DIR/pytorch-bench"
  git -C "$DIR/pytorch-bench" remote add origin https://github.com/pytorch/pytorch.git
  git -C "$DIR/pytorch-bench" config core.sparseCheckout true
  printf 'benchmarks/dynamo/\n.ci/docker/ci_commit_pins/\n.github/ci_commit_pins/\n' > "$DIR/pytorch-bench/.git/info/sparse-checkout"
  git -C "$DIR/pytorch-bench" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
  git -C "$DIR/pytorch-bench" checkout -q FETCH_HEAD
fi
PINS=$DIR/pytorch-bench/.ci/docker/ci_commit_pins
TB=$(cat "$PINS/torchbench.txt")
if [ "$(git -C "$DIR/torchbench" rev-parse HEAD 2>/dev/null)" != "$TB" ]; then
  rm -rf "$DIR/torchbench"
  git init -q "$DIR/torchbench"
  git -C "$DIR/torchbench" remote add origin https://github.com/pytorch/benchmark.git
  git -C "$DIR/torchbench" fetch -q --depth 1 origin "$TB"
  git -C "$DIR/torchbench" checkout -q FETCH_HEAD
fi
MODELS=${TORCHBENCH_MODELS:-$(grep -v detectron2 "$DIR/pytorch-bench/benchmarks/dynamo/all_torchbench_models_list.txt" | tr '\n' ' ')}
for PY in "$@"; do
  VENV=$(cd "$(dirname "$PY")/.." && pwd)
  TORCH=$("$PY" -c "import torch; print(torch.__version__)")
  case "$TORCH" in
    *+cpu) INDEX=https://download.pytorch.org/whl/cpu ;;
    *+cu*) INDEX=https://download.pytorch.org/whl/${TORCH##*+} ;;
    *) INDEX=https://pypi.org/simple ;;
  esac
  echo "== $PY: torch $TORCH"
  VIRTUAL_ENV=$VENV uv pip install -q pip "torch==${TORCH%%+*}" torchvision --index-url "$INDEX" \
    --extra-index-url https://pypi.org/simple
  VIRTUAL_ENV=$VENV uv pip install -q -r "$PINS/huggingface-requirements.txt" numpy pandas psutil pyyaml scipy \
    tqdm tabulate huggingface_hub safetensors sentencepiece protobuf
  VIRTUAL_ENV=$VENV uv pip install -q --no-deps \
    "timm @ git+https://github.com/huggingface/pytorch-image-models@$(cat "$PINS/timm.txt")"
  # TorchBench: its requirements but PyTorch's transformers and timm, under
  # constraints that keep torch, torchvision and numpy (which its models'
  # install.py scripts pass to pip, from build/constraints.txt)
  mkdir -p "$DIR/torchbench/build"
  "$PY" -c "
import importlib.metadata as m
for p in ('torch', 'torchvision', 'numpy', 'transformers', 'timm'):
    print(f'{p}=={m.version(p).split(\"+\")[0]}')" > "$DIR/torchbench/build/constraints.txt"
  grep -v -E '^(timm|transformers)' "$DIR/torchbench/requirements.txt" > "$DIR/torchbench/build/requirements.txt"
  (cd "$DIR/torchbench" && "$PY" -m pip install -q -r build/requirements.txt -c build/constraints.txt)
  # Each model's install.py, as torchbenchmark.setup() runs them (which
  # first insists on torchaudio, whose last release is for PyTorch 2.11)
  for m in $MODELS; do
    d=$DIR/torchbench/torchbenchmark/models/$m
    if [ ! -d "$d" ]; then d=$DIR/torchbench/torchbenchmark/canary_models/$m; fi
    if [ ! -d "$d" ]; then echo "$m: not in TorchBench at this commit"; continue; fi
    if [ ! -f "$d/install.py" ]; then echo "$m: no install.py"; continue; fi
    if (cd "$d" && PYTHONPATH=$DIR/torchbench "$PY" install.py > "$DIR/torchbench/build/install-$m.log" 2>&1); then
      echo "$m: installed"
    else
      echo "$m: install failed (torchbench/build/install-$m.log)"
    fi
  done
done
