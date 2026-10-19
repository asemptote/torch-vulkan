//! Compiles `shaders/*.slang` to SPIR-V in OUT_DIR with slangc and generates
//! `kernels.rs`, a table of (kernel name, SPIR-V bytes).
//!
//! Files starting with `_` are include files and are not compiled on their own.
//! A shader can declare template axes in its leading comment lines:
//!
//!     // axis T: f32=float f64=double i32=int i64=int64_t
//!
//! and is then compiled once per combination of axis values, as kernel
//! `<stem>_<label>[_<label>...]` with `-D T=<type> -D T_<label>`.
//!
//! slangc is found via the SLANGC env var, then the Vulkan SDK (which bundles
//! Slang), then PATH.

use std::env;
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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

/// (axis name, [(label, type)]) for each `// axis` line at the top of `source`.
fn axes(source: &str) -> Vec<(String, Vec<(String, String)>)> {
    source
        .lines()
        .take_while(|l| l.starts_with("//"))
        .filter_map(|l| l.trim_start_matches('/').trim().strip_prefix("axis "))
        .map(|rest| {
            let (name, values) = rest.split_once(':').expect("axis line needs `NAME: label=type ...`");
            let values = values
                .split_whitespace()
                .map(|v| {
                    let (label, ty) = v.split_once('=').expect("axis value needs label=type");
                    (label.to_owned(), ty.to_owned())
                })
                .collect();
            (name.trim().to_owned(), values)
        })
        .collect()
}

/// Every combination of axis values: (name suffix, defines).
fn variants(axes: &[(String, Vec<(String, String)>)]) -> Vec<(String, Vec<String>)> {
    let mut out = vec![(String::new(), Vec::new())];
    for (name, values) in axes {
        out = out
            .into_iter()
            .flat_map(|(suffix, defines)| {
                values.iter().map(move |(label, ty)| {
                    let mut d = defines.clone();
                    d.push(format!("{name}={ty}"));
                    d.push(format!("{name}_{label}"));
                    (format!("{suffix}_{label}"), d)
                })
            })
            .collect();
    }
    out
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let compiler = slangc();
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=shaders");

    let mut sources: Vec<PathBuf> = std::fs::read_dir("shaders")
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("slang"))
        .collect();
    sources.sort();

    // Launch every compilation at once, then collect.
    let mut jobs = Vec::new();
    let mut names = std::collections::HashSet::new();
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        let stem = src.file_stem().unwrap().to_str().unwrap();
        if stem.starts_with('_') {
            continue;
        }
        let text = std::fs::read_to_string(src).unwrap();
        for (suffix, defines) in variants(&axes(&text)) {
            let name = format!("{stem}{suffix}");
            assert!(names.insert(name.clone()), "two shaders produce kernel {name}; rename one");
            let spv = out.join(format!("{name}.spv"));
            let mut cmd = Command::new(&compiler);
            cmd.arg(src)
                .args(["-target", "spirv", "-profile", "spirv_1_5", "-entry", "main", "-stage", "compute", "-O2"])
                // IEEE denormals, like the CPU (GPUs may otherwise flush them to zero).
                .args(["-denorm-mode-fp32", "preserve", "-denorm-mode-fp64", "preserve"])
                .arg("-I")
                .arg("shaders");
            for d in &defines {
                cmd.arg(format!("-D{d}"));
            }
            cmd.arg("-o").arg(&spv).stdout(Stdio::piped()).stderr(Stdio::piped());
            let child = cmd.spawn().unwrap_or_else(|e| panic!("could not run slangc ({}): {e}. Set SLANGC to its path.", compiler.display()));
            jobs.push((name, spv, defines, child));
        }
    }

    let mut table = String::from("pub static SPIRV: &[(&str, &[u8])] = &[\n");
    let mut failures = Vec::new();
    for (name, spv, defines, child) in jobs {
        let result = child.wait_with_output().unwrap();
        if !result.status.success() {
            failures.push(format!(
                "slangc failed for {name} ({}):\n{}{}",
                defines.join(" "),
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            ));
            continue;
        }
        writeln!(table, "    ({name:?}, include_bytes!({:?})),", spv.display().to_string()).unwrap();
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    table.push_str("];\n");
    std::fs::write(out.join("kernels.rs"), table).unwrap();
}
