//! Compiles the C++ PyTorch backend (csrc/*.cpp) against the libtorch that
//! ships with the target Python's `torch` package, and links it in.
//!
//! The Python interpreter is taken from PYO3_PYTHON (set by maturin), then
//! VIRTUAL_ENV, then `python` on PATH; TORCH_DIR overrides the torch location.

use std::path::{Path, PathBuf};
use std::process::Command;

fn python() -> PathBuf {
    if let Some(p) = std::env::var_os("PYO3_PYTHON") {
        return p.into();
    }
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
        let exe = Path::new(&venv).join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });
        if exe.exists() {
            return exe;
        }
    }
    "python".into()
}

fn torch_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("TORCH_DIR") {
        return dir.into();
    }
    let out = Command::new(python())
        .args(["-c", "import os, torch; print(os.path.dirname(torch.__file__))"])
        .output()
        .expect("could not run Python to locate torch; set TORCH_DIR");
    assert!(out.status.success(), "importing torch failed: {}", String::from_utf8_lossy(&out.stderr));
    PathBuf::from(String::from_utf8(out.stdout).unwrap().trim())
}

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-env-changed=TORCH_DIR");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let torch = torch_dir();
    let include = torch.join("include");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .files(["csrc/backend.cpp", "csrc/ops.cpp", "csrc/reduce.cpp", "csrc/linalg.cpp", "csrc/fft.cpp"])
        .include(&include)
        .include(include.join("torch/csrc/api/include"))
        .define("NOMINMAX", None)
        .define("_CRT_SECURE_NO_WARNINGS", None)
        .warnings(false);
    if build.get_compiler().is_like_msvc() {
        build.flags(["/std:c++20", "/EHsc", "/Zc:__cplusplus", "/permissive-", "/bigobj", "/utf-8"]);
    } else {
        build.flags(["-std=c++20", "-fvisibility=hidden"]);
    }
    build.compile("vk_backend");

    println!("cargo:rustc-link-search=native={}", torch.join("lib").display());
    for lib in ["c10", "torch_cpu", "torch"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
