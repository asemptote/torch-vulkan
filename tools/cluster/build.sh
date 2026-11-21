#!/bin/bash
# Rebuilds the extension into $TORCH_VULKAN/.venv (default ~/torch-vulkan;
# after setup.sh), with Cargo's files on the machine's local /tmp, one target
# directory per checkout. Logs to $LOG (default ~/<checkout>-build.log) and
# ends it with "build exit N", so it can run detached:
#   setsid nohup bash tools/cluster/build.sh < /dev/null > /dev/null 2>&1 &
# Do not rebuild while shards are running: the library is replaced under them.
cd "${TORCH_VULKAN:-$HOME/torch-vulkan}" || exit 1
name=$(basename "$PWD")
LOG=${LOG:-$HOME/$name-build.log}
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH VULKAN_SDK=${VULKAN_SDK:-$(ls -d "$HOME"/opt/*/x86_64 | tail -1)}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/tmp/$USER-$name-target} VIRTUAL_ENV=$PWD/.venv
.venv/bin/maturin develop --release --uv > "$LOG" 2>&1
code=$?
echo "build exit $code" >> "$LOG"
exit $code
