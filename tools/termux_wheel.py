"""Packs a Termux build of torch-vulkan (tools/build-termux.sh) into a wheel
that pip installs into Termux's Python:

    python tools/termux_wheel.py [OUT_DIR]      (default: dist)

It takes python/torch_vulkan as built (with `_native.abi3.so` and the
`_build_info.py` that build-termux.sh writes) and names the wheel for this
Python's Android platform, e.g.
`torch_vulkan-0.3.0-cp311-abi3-android_24_arm64_v8a.whl` (or
`armeabi_v7a` for 32-bit ARM). The wheel requires exactly the torch it was
compiled against (`torch==2.15.0`): Termux's `python-torch` satisfies that,
and pip finds no other torch for Android, so with any other torch installed
pip refuses the wheel. After a later `python-torch` upgrade, importing
torch_vulkan raises an ImportError instead (`_build_info.py`, checked in
`__init__.py`) rather than loading against another libtorch.

Standard library only (Termux may have no wheel or maturin installed).
"""

import base64
import hashlib
import pathlib
import re
import sys
import sysconfig
import tomllib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "python" / "torch_vulkan"
NATIVE = "_native.abi3.so"


def platform_tag():
    # sysconfig gives "android-24-arm64_v8a" (API level 24: Termux's minimum).
    plat = sysconfig.get_platform()
    assert plat.startswith("android-"), f"not Termux's Python: platform {plat!r}"
    return re.sub(r"[-.]", "_", plat)


def record_hash(data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def main():
    out_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "dist")
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = meta["version"]
    native = PACKAGE / NATIVE
    info = PACKAGE / "_build_info.py"
    assert native.exists() and info.exists(), "run tools/build-termux.sh first"
    built = {}
    exec(info.read_text(), built)
    torch_version = built["TORCH_VERSION"]

    # pyo3's abi3 floor (crates/pyvk/Cargo.toml: abi3-py311).
    cargo = (ROOT / "crates" / "pyvk" / "Cargo.toml").read_text()
    floor = re.search(r'"abi3-py3(\d+)"', cargo).group(1)
    tag = f"cp3{floor}-abi3-{platform_tag()}"
    dist_info = f"torch_vulkan-{version}.dist-info"

    files = {}
    for path in sorted(PACKAGE.iterdir()):
        if path.is_file() and (path.suffix == ".py" or path.name == NATIVE):
            files[f"torch_vulkan/{path.name}"] = path.read_bytes()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    files[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.1\n"
        "Name: torch-vulkan\n"
        f"Version: {version}\n"
        f"Summary: {meta['description']}\n"
        f"License: {meta['license']['text']}\n"
        f"Requires-Python: {meta['requires-python']}\n"
        f"Requires-Dist: torch=={torch_version}\n"
        f"Project-URL: Repository, {meta['urls']['Repository']}\n"
        "Description-Content-Type: text/markdown\n\n" + readme
    ).encode()
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: torch-vulkan tools/termux_wheel.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    ).encode()
    for name in ("LICENSE", "LICENSE-APACHE", "LICENSE-MIT", "NOTICE"):
        files[f"{dist_info}/licenses/{name}"] = (ROOT / name).read_bytes()
    record = "".join(f"{n},{record_hash(d)},{len(d)}\n" for n, d in files.items())
    files[f"{dist_info}/RECORD"] = (record + f"{dist_info}/RECORD,,\n").encode()

    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / f"torch_vulkan-{version}-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            entry.external_attr = (0o755 if name.endswith(".so") else 0o644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data)
    print(f"{wheel} (requires torch=={torch_version})")


if __name__ == "__main__":
    main()
