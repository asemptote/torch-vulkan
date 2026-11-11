#!/bin/bash
# One-time setup on a Linux machine: Rust, the Vulkan SDK (slangc), PyTorch's
# tests at the installed torch's commit, a venv with torch, and the build:
#   bash tools/cluster/setup.sh        (from the repo at ~/torch-vulkan)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd ~
echo "[$(date +%T)] rust"
if [ ! -x ~/.cargo/bin/cargo ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

echo "[$(date +%T)] vulkan sdk (slangc)"
SDK_VERSION=1.4.363.0
mkdir -p ~/opt
if [ ! -x ~/opt/$SDK_VERSION/x86_64/bin/slangc ]; then
  cd ~/opt
  curl -sSfL -o sdk.tar.xz "https://sdk.lunarg.com/sdk/download/$SDK_VERSION/linux/vulkansdk-linux-x86_64-$SDK_VERSION.tar.xz"
  tar xf sdk.tar.xz
  rm sdk.tar.xz
  cd ~
fi
export VULKAN_SDK=~/opt/$SDK_VERSION/x86_64
"$VULKAN_SDK/bin/slangc" -version

echo "[$(date +%T)] pytorch tests"
SRC="$REPO/pytorch-src"  # (git-ignored)
if [ ! -f "$SRC/test/test_ops.py" ] && [ -f ~/pytorch-src/test/test_ops.py ]; then
  mv ~/pytorch-src "$SRC"  # (where earlier setups put it)
fi
if [ ! -f "$SRC/test/test_ops.py" ]; then
  rm -rf "$SRC"
  git init -q "$SRC"
  cd "$SRC"
  git remote add origin https://github.com/pytorch/pytorch.git
  git config core.sparseCheckout true
  echo "test/" > .git/info/sparse-checkout
  git fetch -q --depth 1 --filter=blob:none origin 5c4886908584029761b579af026dcfb627c84070
  git checkout -q FETCH_HEAD
  cd ~
fi

echo "[$(date +%T)] python env"
cd ~/torch-vulkan
if [ ! -x .venv/bin/python ]; then
  uv venv -q --python 3.12 .venv
fi
export VIRTUAL_ENV=$PWD/.venv
uv pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q numpy pytest pytest-timeout maturin expecttest hypothesis

echo "[$(date +%T)] build"
# Cargo's many small files on local disk (NFS is slow for them).
export CARGO_TARGET_DIR=/tmp/$USER-torch-vulkan-target
.venv/bin/maturin develop --release --uv
echo "[$(date +%T)] smoke"
.venv/bin/python -c "import torch, torch_vulkan; x = torch.arange(5.0, device='vk'); print((x * 2).sum().item(), torch_vulkan.device_name())"
echo "[$(date +%T)] done"
