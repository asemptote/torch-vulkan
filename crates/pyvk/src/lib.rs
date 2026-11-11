//! `torch_vulkan._native`: the extension module behind device "vk".
//!
//! It links the Rust runtime (memory pool, kernels, stream) with the C++
//! PyTorch backend in `csrc/` (allocator, device guard, hooks, aten kernels)
//! through the C ABI in `ffi.rs`. Importing it registers the backend.

mod ffi;

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use vkcore::Runtime;

pub(crate) static RUNTIME: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

/// The process-wide runtime, created on first use.
pub(crate) fn runtime() -> Result<Arc<Runtime>, String> {
    let mut slot = RUNTIME.lock().unwrap();
    if slot.is_none() {
        *slot = Some(Arc::new(Runtime::new().map_err(|e| e.to_string())?));
    }
    Ok(slot.as_ref().unwrap().clone())
}

fn py_runtime() -> PyResult<Arc<Runtime>> {
    runtime().map_err(PyRuntimeError::new_err)
}

extern "C" {
    fn vkt_register_backend();
    fn vkt_clear_fft_caches();
    fn vkt_release_device_error_word();
    fn vkt_take_device_error() -> i32;
    fn vkt_record_fallback(op: *const std::ffi::c_char) -> i32;
    fn vkt_manual_seed(seed: u64);
    fn vkt_initial_seed() -> u64;
    fn vkt_seed() -> u64;
    fn vkt_get_rng_state(seed: *mut u64, offset: *mut u64);
    fn vkt_set_rng_state(seed: u64, offset: u64);
    fn vkt_fallback_counts(buf: *mut std::ffi::c_char, len: usize) -> usize;
}

/// Waits until all queued GPU work has finished; raises IndexError if a
/// kernel met an out-of-range index since the last check.
#[pyfunction]
fn synchronize() -> PyResult<()> {
    py_runtime()?.synchronize().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    if unsafe { vkt_take_device_error() } != 0 {
        return Err(PyIndexError::new_err("torch_vulkan: index out of range in an earlier indexing kernel"));
    }
    Ok(())
}

/// Counts (and warns about, once) an op that ran on the CPU; false in strict
/// mode (TORCH_VK_STRICT), where the caller must raise instead.
#[pyfunction]
fn record_fallback(op: &str) -> bool {
    let name = std::ffi::CString::new(op.replace('\0', " ")).unwrap_or_default();
    unsafe { vkt_record_fallback(name.as_ptr()) == 0 }
}

/// Optional device features as bits (1 float64, 2 int16, 4 float16, 8 int8).
#[pyfunction]
fn capabilities() -> u32 {
    ffi::vkt_capabilities()
}

/// The subgroup size kernel `name` runs with (e.g. "gemmsg_f32"): 0 for
/// kernels not written for subgroups and where the device cannot run it.
#[pyfunction]
fn kernel_subgroup_size(name: &str) -> u32 {
    runtime().map(|rt| vkcore::kernel_subgroup_size(&rt.ctx.caps, name)).unwrap_or(0)
}

/// Seeds the default random generator (offset back to 0).
#[pyfunction]
fn manual_seed(seed: u64) {
    unsafe { vkt_manual_seed(seed) }
}

/// The default generator's seed.
#[pyfunction]
fn initial_seed() -> u64 {
    unsafe { vkt_initial_seed() }
}

/// Seeds the default generator nondeterministically; returns the seed.
#[pyfunction]
fn seed() -> u64 {
    unsafe { vkt_seed() }
}

/// (seed, offset) of the default generator.
#[pyfunction]
fn get_rng_state() -> (u64, u64) {
    let (mut s, mut o) = (0u64, 0u64);
    unsafe { vkt_get_rng_state(&mut s, &mut o) };
    (s, o)
}

#[pyfunction]
fn set_rng_state(seed: u64, offset: u64) {
    unsafe { vkt_set_rng_state(seed, offset) }
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    Ok(py_runtime()?.ctx.device_name.clone())
}

/// Kernels dispatched so far (for profiling: launches per op).
#[pyfunction]
fn dispatch_count() -> PyResult<u64> {
    Ok(py_runtime()?.stream.lock().unwrap().dispatches)
}

/// Names of all compiled kernels.
#[pyfunction]
fn kernels() -> Vec<&'static str> {
    vkcore::KernelCache::names().collect()
}

/// (allocated bytes, reserved bytes, blocks) of the memory pool.
#[pyfunction]
fn memory_stats() -> PyResult<(u64, u64, usize)> {
    let s = py_runtime()?.pool.lock().unwrap().stats();
    Ok((s.allocated, s.reserved, s.blocks))
}

/// Releases cached, unused pool blocks back to the driver.
#[pyfunction]
fn empty_cache() -> PyResult<()> {
    let rt = py_runtime()?;
    rt.synchronize().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    rt.pool.lock().unwrap().trim();
    Ok(())
}

/// Validation-layer warnings/errors so far (needs TORCH_VK_VALIDATE=1).
#[pyfunction]
fn validation_issues() -> usize {
    vkcore::validation_issues()
}

/// The most recent validation-layer message ("" if none yet).
#[pyfunction]
fn last_validation_message() -> String {
    vkcore::last_validation_message()
}

/// {op: count} of aten ops that ran on the CPU instead of the GPU.
#[pyfunction]
fn fallback_counts() -> Vec<(String, u64)> {
    let len = unsafe { vkt_fallback_counts(std::ptr::null_mut(), 0) };
    let mut buf = vec![0u8; len + 64];
    unsafe { vkt_fallback_counts(buf.as_mut_ptr() as *mut std::ffi::c_char, buf.len()) };
    let text = String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(0)]).into_owned();
    text.lines()
        .filter_map(|l| l.rsplit_once('='))
        .filter_map(|(op, n)| n.parse().ok().map(|n| (op.to_owned(), n)))
        .collect()
}

/// Finishes queued work and destroys the runtime (saving the pipeline cache).
#[pyfunction]
fn shutdown() {
    // Cached device tables must be freed while the memory pool still exists.
    unsafe {
        vkt_clear_fft_caches();
        vkt_release_device_error_word();
    }
    let rt = RUNTIME.lock().unwrap().take();
    if let Some(rt) = &rt {
        let _ = rt.synchronize();
    }
    drop(rt);
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    unsafe { vkt_register_backend() };
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch_count, m)?)?;
    m.add_function(wrap_pyfunction!(record_fallback, m)?)?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_subgroup_size, m)?)?;
    m.add_function(wrap_pyfunction!(manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(initial_seed, m)?)?;
    m.add_function(wrap_pyfunction!(seed, m)?)?;
    m.add_function(wrap_pyfunction!(get_rng_state, m)?)?;
    m.add_function(wrap_pyfunction!(set_rng_state, m)?)?;
    m.add_function(wrap_pyfunction!(kernels, m)?)?;
    m.add_function(wrap_pyfunction!(memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(empty_cache, m)?)?;
    m.add_function(wrap_pyfunction!(validation_issues, m)?)?;
    m.add_function(wrap_pyfunction!(last_validation_message, m)?)?;
    m.add_function(wrap_pyfunction!(fallback_counts, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    Ok(())
}
