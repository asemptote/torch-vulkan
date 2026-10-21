//! `torch_vulkan._native`: the extension module behind device "vk".
//!
//! It links the Rust runtime (memory pool, kernels, stream) with the C++
//! PyTorch backend in `csrc/` (allocator, device guard, hooks, aten kernels)
//! through the C ABI in `ffi.rs`. Importing it registers the backend.

mod ffi;

use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
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
    fn vkt_fallback_counts(buf: *mut std::ffi::c_char, len: usize) -> usize;
}

/// Waits until all queued GPU work has finished.
#[pyfunction]
fn synchronize() -> PyResult<()> {
    py_runtime()?.synchronize().map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    Ok(py_runtime()?.ctx.device_name.clone())
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
    unsafe { vkt_clear_fft_caches() };
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
    m.add_function(wrap_pyfunction!(kernels, m)?)?;
    m.add_function(wrap_pyfunction!(memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(empty_cache, m)?)?;
    m.add_function(wrap_pyfunction!(validation_issues, m)?)?;
    m.add_function(wrap_pyfunction!(fallback_counts, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    Ok(())
}
