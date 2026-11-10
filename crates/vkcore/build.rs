//! Compiles `shaders/*.slang` to SPIR-V in OUT_DIR with slangc and generates
//! `kernels.rs`, a table of (kernel name, SPIR-V bytes).
//!
//! - Files starting with `_` are include files, not kernels.
//! - `shaders/opcodes.txt` becomes `opcodes.slang` (included by the shaders);
//!   crates/pyvk/build.rs turns the same file into a C++ header.
//! - A shader can declare template axes in its leading comment lines:
//!
//!       // axis T: f32=float f64=double i32=int i64=int64_t
//!
//!   and is then compiled once per combination of axis values, as kernel
//!   `<stem>_<label>[_<label>...]` with `-D T=<type> -D T_<label>`.
//! - `shaders/test/*.slang` (kernels used only by vkcore's own tests and
//!   examples) are compiled only with the `test-kernels` feature.
//! - Each variant is recompiled only when its inputs change: a hash of the
//!   source, every include file, the defines and the compiler is stored next
//!   to the SPIR-V.
//!
//! slangc is found via the SLANGC env var, then the Vulkan SDK (which bundles
//! Slang), then PATH. Where there is no slangc (Android/Termux), set
//! VKCORE_PREBUILT_SPIRV to a directory of `<kernel>.spv` files compiled
//! elsewhere (SPIR-V is portable: e.g. another build's OUT_DIR): the table is
//! made from those instead.

use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fmt::Write;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};

#[path = "src/spirv_fix.rs"]
mod spirv_fix;

/// Bumped when the post-processing of compiled modules changes (part of every
/// variant's cache key).
const POSTPROCESS: &str = "drop-forward-pointers-1";

/// Flags for every kernel.
const SLANGC_FLAGS: &[&str] = &[
    // SPIR-V 1.3, the newest Vulkan 1.1 accepts: slang then declares
    // SPV_KHR_physical_storage_buffer and friends as extensions (core in 1.5).
    "-target", "spirv", "-profile", "spirv_1_3", "-entry", "main", "-stage", "compute", "-O2",
    // Plain C-like layout for push constants and buffers (scalarBlockLayout,
    // required by vkcore), so host structs match field for field.
    "-fvk-use-scalar-layout",
    // IEEE denormals like the CPU; vkcore strips these execution modes at
    // load time on devices that cannot preserve denormals.
    "-denorm-mode-fp32", "preserve", "-denorm-mode-fp64", "preserve",
];

/// Every kernel but these is compiled with `-fp-mode precise`, which marks
/// each float operation NoContraction: drivers may then neither fuse nor
/// reassociate them (Mesa's NIR otherwise rewrites e.g. Cody-Waite's
/// (x - k c1) - k c2 into x - k (c1 + c2), and error-free transformations
/// into nothing). Matrix products keep the default: their inner loops want
/// fused multiply-adds, which only make them more accurate.
const DEFAULT_FP_MODE: &[&str] = &["gemm"];

fn slangc() -> PathBuf {
    if let Ok(p) = env::var("SLANGC") {
        return p.into();
    }
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        // Windows SDK layout, then Linux/macOS.
        for p in [Path::new(&sdk).join("Bin").join("slangc.exe"), Path::new(&sdk).join("bin").join("slangc")] {
            if p.exists() {
                return p;
            }
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

/// `opcodes.txt` lines as (FAMILY_NAME, code).
fn opcodes(text: &str) -> Vec<(String, u32)> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let parts: Vec<&str> = l.split_whitespace().collect();
            assert!(parts.len() == 3, "opcodes.txt: expected `FAMILY NAME CODE`, got `{l}`");
            (format!("{}_{}", parts[0], parts[1]), parts[2].parse().expect("opcodes.txt: code must be a number"))
        })
        .collect()
}

/// A running slangc job: (kernel name, defines, process, hash stamp path, hash).
type Job<P> = (String, Vec<String>, P, PathBuf, String);

fn wait(job: Job<Child>) -> Job<Output> {
    let (name, defines, child, stamp, key) = job;
    (name, defines, child.wait_with_output().unwrap(), stamp, key)
}

fn slang_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("slang"))
        .collect();
    files.sort();
    files
}

/// kernels.rs from the `.spv` files of a directory compiled elsewhere.
fn prebuilt(dir: &Path, out: &Path) {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("VKCORE_PREBUILT_SPIRV={}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("spv"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "VKCORE_PREBUILT_SPIRV={}: no .spv files", dir.display());
    let mut table = String::from("pub static SPIRV: &[(&str, &[u8])] = &[\n");
    for f in files {
        let name = f.file_stem().unwrap().to_str().unwrap();
        let abs = std::fs::canonicalize(&f).unwrap();
        writeln!(table, "    ({name:?}, include_bytes!({:?})),", abs.display().to_string()).unwrap();
    }
    table.push_str("];\n");
    std::fs::write(out.join("kernels.rs"), table).unwrap();
}

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-env-changed=VKCORE_PREBUILT_SPIRV");
    if let Some(dir) = env::var_os("VKCORE_PREBUILT_SPIRV") {
        // (the directory: a kernel added or removed re-runs this)
        println!("cargo:rerun-if-changed={}", Path::new(&dir).display());
        prebuilt(Path::new(&dir), &out);
        return;
    }
    let compiler = slangc();
    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=VKCORE_SLANGC_JOBS");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=shaders");

    // Shared op codes.
    let codes_text = std::fs::read_to_string("shaders/opcodes.txt").unwrap();
    let mut slang_codes = String::from("// Generated from shaders/opcodes.txt by crates/vkcore/build.rs.\n");
    let mut seen = std::collections::HashSet::new();
    for (name, code) in opcodes(&codes_text) {
        assert!(seen.insert(name.clone()), "opcodes.txt: {name} defined twice");
        writeln!(slang_codes, "static const int {name} = {code};").unwrap();
    }
    std::fs::write(out.join("opcodes.slang"), &slang_codes).unwrap();

    let mut sources = slang_files(Path::new("shaders"));
    if env::var_os("CARGO_FEATURE_TEST_KERNELS").is_some() {
        sources.extend(slang_files(Path::new("shaders/test")));
    }

    // Everything a kernel can include: hashed into every variant's key.
    let mut includes = DefaultHasher::new();
    for f in slang_files(Path::new("shaders")).iter().filter(|p| p.file_name().unwrap().to_str().unwrap().starts_with('_')) {
        std::fs::read(f).unwrap().hash(&mut includes);
    }
    slang_codes.hash(&mut includes);
    compiler.hash(&mut includes);
    SLANGC_FLAGS.hash(&mut includes);
    POSTPROCESS.hash(&mut includes);
    let includes = includes.finish();

    // Launch every compilation that is out of date at once, then collect.
    let mut jobs = Vec::new();
    let mut table = String::from("pub static SPIRV: &[(&str, &[u8])] = &[\n");
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
            writeln!(table, "    ({name:?}, include_bytes!({:?})),", spv.display().to_string()).unwrap();

            let precise = !DEFAULT_FP_MODE.contains(&stem);
            let mut key = DefaultHasher::new();
            (includes, &text, &defines, precise).hash(&mut key);
            let key = format!("{:016x}", key.finish());
            let stamp = out.join(format!("{name}.hash"));
            if spv.exists() && std::fs::read_to_string(&stamp).ok().as_deref() == Some(key.as_str()) {
                continue;  // up to date
            }
            let mut cmd = Command::new(&compiler);
            cmd.arg(src).args(SLANGC_FLAGS).arg("-I").arg("shaders").arg("-I").arg(&out);
            if precise {
                cmd.args(["-fp-mode", "precise"]);
            }
            for d in &defines {
                cmd.arg(format!("-D{d}"));
            }
            cmd.arg("-o").arg(&spv).stdout(Stdio::piped()).stderr(Stdio::piped());
            jobs.push((name, defines, cmd, stamp, key));
        }
    }

    // Run slangc on every stale variant, a bounded number at a time: each
    // instance peaks near 400 MB, and hundreds at once exhaust memory. Beyond
    // one per logical CPU the build is no faster (measured: 4 jobs 30 s, 8
    // jobs 24 s, 12-16 jobs 23-25 s for 233 variants on a 4-core laptop).
    // VKCORE_SLANGC_JOBS overrides the count.
    let width = std::env::var("VKCORE_SLANGC_JOBS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let mut running = std::collections::VecDeque::new();
    let mut done = Vec::new();
    for (name, defines, mut cmd, stamp, key) in jobs {
        if running.len() >= width {
            done.push(wait(running.pop_front().unwrap()));
        }
        let child = cmd.spawn().unwrap_or_else(|e| panic!("could not run slangc ({}): {e}. Set SLANGC to its path.", compiler.display()));
        running.push_back((name, defines, child, stamp, key));
    }
    done.extend(running.into_iter().map(wait));

    let mut failures = Vec::new();
    for (name, defines, result, stamp, key) in done {
        if result.status.success() {
            // (spirv_fix.rs: no forward pointers, which Intel's driver rejects)
            let spv = out.join(format!("{name}.spv"));
            let bytes = std::fs::read(&spv).unwrap();
            let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
            if let Some(fixed) = spirv_fix::drop_forward_pointers(&words) {
                std::fs::write(&spv, fixed.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
            }
            std::fs::write(stamp, key).unwrap();
        } else {
            failures.push(format!(
                "slangc failed for {name} ({}):\n{}{}",
                defines.join(" "),
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    table.push_str("];\n");
    std::fs::write(out.join("kernels.rs"), table).unwrap();
}
