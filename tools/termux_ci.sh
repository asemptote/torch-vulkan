#!/data/data/com.termux/files/usr/bin/bash
# The release workflow's Termux steps, run inside a termux/termux-docker
# container (aarch64 or arm) with the repository mounted read-only at /src:
#   tools/termux_ci.sh build SPIRV_DIR OUT_DIR
#     installs Termux's python-torch, rust and clang, builds with
#     tools/build-termux.sh and puts the wheel in OUT_DIR;
#   tools/termux_ci.sh test WHEEL_DIR
#     in a fresh container: installs python-torch, Mesa's lavapipe and the
#     wheel, checks that a different torch is refused at import, and runs the
#     release smoke test and tests/python on lavapipe, outside the source
#     tree. On 32-bit ARM, lavapipe's own shaders fault (SIGBUS, see
#     docs/log/M17), so there only the import is checked; the arm wheel is
#     tested on a phone's GPU (RELEASING.md).
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive

install() {
    apt-get update -q
    apt-get -y -q -o Dpkg::Options::=--force-confnew upgrade >/dev/null
    apt-get -y -q -o Dpkg::Options::=--force-confnew install "$@" >/dev/null
}

# The release is compiled and tested against tools/torch-version; Termux's
# python-torch must be that release.
check_torch() {
    local want have
    want=$(cat /src/tools/torch-version)
    have=$(python -c 'import torch; print(torch.__version__.split("+")[0])')
    echo "Termux python-torch: torch $have (tools/torch-version: $want), Python $(python -c 'import platform; print(platform.python_version())'), $(uname -m)"
    [ "$have" = "$want" ] || { echo "Termux's python-torch is $have, not tools/torch-version's $want"; exit 1; }
}

case "$1" in
build)
    spirv=${2:?SPIRV_DIR}; out=${3:?OUT_DIR}
    # (Termux's libtorch links libvulkan.so without depending on a loader)
    install python-torch rust clang vulkan-loader-generic
    check_torch
    rm -rf ~/tv && cp -r /src ~/tv && rm -rf ~/tv/target ~/tv/dist
    cd ~/tv
    JOBS=$(nproc) tools/build-termux.sh "$spirv"
    cp dist/*.whl "$out"/
    ;;
test)
    wheels=${2:?WHEEL_DIR}
    install python-torch mesa-vulkan-icd-swrast
    check_torch
    pip install -q --disable-pip-version-check "$wheels"/torch_vulkan-*.whl pytest
    mkdir -p ~/t && cp -r /src/tests/python ~/t/ && cp /src/tools/release_smoke.py ~/t/
    cd ~/t
    python -c 'import torch_vulkan as tv; print("imported", tv.__version__, "device", tv.device_name())'
    # Another torch (a later python-torch) is refused with an ImportError.
    python - <<'EOF'
import torch
torch.__version__ = "2.99.0"
try:
    import torch_vulkan
except ImportError as e:
    print("refused as expected:", e)
else:
    raise SystemExit("torch_vulkan imported under another torch")
EOF
    case $(uname -m) in
    aarch64|x86_64)
        TORCH_VK_STRICT=1 python release_smoke.py
        # Not run here (docs/log/M40-termux-packages.md):
        # - lavapipe on aarch64 flushes subnormals to 0 (float32 and float64);
        # - this CPU, the tests' reference, differs from x86's for NaN and inf
        #   quantised to integers, and gives 0 for torch.dot of two
        #   conjugated complex vectors;
        # - PyTorch's aarch64 CPU packs _dyn_quant_matmul_4bit's weights in
        #   KleidiAI's layout, which vk does not read.
        python -m pytest -q -p no:cacheprovider python \
            --deselect python/test_math64.py::test_float64_math_accuracy[exp] \
            --deselect python/test_math64.py::test_float64_math_accuracy[exp2] \
            --deselect python/test_math64.py::test_float64_math_accuracy[tanh] \
            --deselect python/test_cpu_rounding.py::test_sqrt_float64_correctly_rounded \
            --deselect python/test_fallback_ops.py::test_fake_quantize_cachemask_ops \
            --deselect python/test_fallback_ops.py::test_quantized_per_tensor \
            --deselect python/test_cpu_rounding.py::test_conjugates_inside_a_dispatch_mode \
            --deselect python/test_quant.py::test_dyn_quant_matmul_4bit_exact
        ;;
    *) echo "32-bit: lavapipe's shaders fault here; the GPU run is on a phone" ;;
    esac
    ;;
*) echo "usage: $0 build SPIRV_DIR OUT_DIR | test WHEEL_DIR"; exit 2 ;;
esac
