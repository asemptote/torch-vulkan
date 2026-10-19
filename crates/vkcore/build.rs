//! Compiles every `shaders/*.slang` file to SPIR-V in OUT_DIR with slangc.
//!
//! slangc is found via the SLANGC env var, then the Vulkan SDK (which bundles
//! Slang), then PATH.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn slangc() -> PathBuf {
    if let Ok(p) = env::var("SLANGC") {
        return p.into();
    }
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let p = Path::new(&sdk).join("Bin").join("slangc.exe");
        if p.exists() {
            return p;
        }
    }
    "slangc".into()
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let compiler = slangc();
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-changed=shaders");

    for entry in std::fs::read_dir("shaders").unwrap() {
        let src = entry.unwrap().path();
        if src.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        let stem = src.file_stem().unwrap().to_str().unwrap();
        let spv = out.join(format!("{stem}.spv"));
        let status = Command::new(&compiler)
            .arg(&src)
            .args(["-target", "spirv", "-profile", "spirv_1_5", "-entry", "main", "-stage", "compute", "-O2"])
            .arg("-o")
            .arg(&spv)
            .status()
            .unwrap_or_else(|e| panic!("could not run slangc ({}): {e}. Set SLANGC to its path.", compiler.display()));
        assert!(status.success(), "slangc failed on {}", src.display());
    }
}
