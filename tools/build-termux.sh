#!/data/data/com.termux/files/usr/bin/bash
# Builds torch-vulkan on Android in Termux (with `pkg install python-torch
# rust clang`), from the repo root:
#   tools/build-termux.sh SPIRV_DIR
# SPIRV_DIR holds the kernels' .spv files compiled on another machine (there
# is no slangc for Android; SPIR-V is portable): the `<kernel>.spv` files a
# desktop build lists in target/release/build/vkcore-*/out/kernels.rs. Two
# compiles at a time (JOBS): a C++ file with libtorch's headers takes up to
# about 1 GB, and phones have little RAM; after the first build only the
# files an edit affects recompile (crates/pyvk/build.rs caches objects). The module lands in python/torch_vulkan; run with
# PYTHONPATH=$PWD/python, under Android's own Vulkan loader (Termux's
# libvulkan finds only Mesa's llvmpipe), e.g. LD_LIBRARY_PATH pointing at a
# directory with libvulkan.so -> /system/lib/libvulkan.so (/system/lib64 for
# a 64-bit userspace).
set -eo pipefail
cd "$(dirname "$0")/.."
export VKCORE_PREBUILT_SPIRV=$(realpath "${1:?usage: tools/build-termux.sh SPIRV_DIR}")
export PYO3_PYTHON=python
cargo build --release -p pyvk -j "${JOBS:-2}"
cp target/release/lib_native.so python/torch_vulkan/_native.abi3.so
echo "built python/torch_vulkan/_native.abi3.so"
