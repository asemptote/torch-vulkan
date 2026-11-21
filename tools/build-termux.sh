#!/data/data/com.termux/files/usr/bin/bash
# Builds torch-vulkan on Android in Termux (with `pkg install python-torch
# rust clang vulkan-loader-android`: Termux's libtorch needs a libvulkan.so,
# and Android's reaches the GPU; see below), from the repo root:
#   tools/build-termux.sh SPIRV_DIR
# SPIRV_DIR holds the kernels' .spv files compiled on another machine (there
# is no slangc for Android; SPIR-V is portable): a release's
# torch-vulkan-spirv-*.tar.gz extracted, or the `<kernel>.spv` files a
# desktop build lists in target/release/build/vkcore-*/out/kernels.rs.
# Three compiles at a time by default (JOBS): a C++ file with libtorch's
# headers peaks near 350 MB, and phones have little RAM; after the first
# build only the files an edit affects recompile (crates/pyvk/build.rs caches
# objects).
#
# The module lands in python/torch_vulkan, with _build_info.py recording the
# torch it was compiled against (importing torch_vulkan with another torch,
# after a `pkg upgrade`, raises an ImportError), and the whole package in a
# wheel, dist/torch_vulkan-*-android_*.whl (tools/termux_wheel.py), the same
# kind the releases attach: `pip install dist/torch_vulkan-*.whl`, or run in
# place with PYTHONPATH=$PWD/python.
#
# The phone's GPU is reached through Android's own Vulkan loader (Termux's
# libvulkan finds only Mesa's drivers, such as llvmpipe on the CPU):
# `pkg install vulkan-loader-android` makes Termux's libvulkan.so Android's,
# or, to keep vulkan-loader-generic for other programs, point LD_LIBRARY_PATH
# at a directory with libvulkan.so -> /system/lib64/libvulkan.so
# (/system/lib for a 32-bit userspace) for torch-vulkan's processes only.
set -eo pipefail
cd "$(dirname "$0")/.."
export VKCORE_PREBUILT_SPIRV=$(realpath "${1:?usage: tools/build-termux.sh SPIRV_DIR}")
export PYO3_PYTHON=python
cargo build --release -p pyvk -j "${JOBS:-3}"
cp target/release/lib_native.so python/torch_vulkan/_native.abi3.so
torch_version=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
cat > python/torch_vulkan/_build_info.py <<EOF
# Written by tools/build-termux.sh: the PyTorch this build was compiled
# against (checked when torch_vulkan is imported).
TORCH_VERSION = "$torch_version"
EOF
echo "built python/torch_vulkan/_native.abi3.so against torch $torch_version"
python tools/termux_wheel.py dist
