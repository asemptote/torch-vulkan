//! `torch_vulkan._native`: a thin Python view of the vkcore runtime.
//!
//! Tensor logic (shapes, strides, dtypes, dispatch) lives in Python; this
//! module only allocates memory, moves bytes between host and device, and
//! queues kernels with a packed push-constant block.

use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use vkcore::{Allocation, Runtime, PUSH_CONSTANT_BYTES};

static RUNTIME: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

fn to_py(e: Box<dyn std::error::Error>) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// The process-wide runtime, created on first use.
fn runtime() -> PyResult<Arc<Runtime>> {
    let mut slot = RUNTIME.lock().unwrap();
    if slot.is_none() {
        *slot = Some(Arc::new(Runtime::new().map_err(to_py)?));
    }
    Ok(slot.as_ref().unwrap().clone())
}

/// A pool allocation. Freed (stream-ordered) when the Python object dies.
#[pyclass(frozen, module = "torch_vulkan._native")]
struct Buffer {
    alloc: Allocation,
}

#[pymethods]
impl Buffer {
    /// GPU address of the first byte.
    #[getter]
    fn address(&self) -> u64 {
        self.alloc.address
    }

    /// Usable size in bytes (rounded up to the pool alignment).
    #[getter]
    fn nbytes(&self) -> u64 {
        self.alloc.size
    }

    fn __repr__(&self) -> String {
        format!("Buffer(address=0x{:x}, nbytes={})", self.alloc.address, self.alloc.size)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // After shutdown() the pool is gone and its memory already released.
        if let Some(rt) = RUNTIME.lock().unwrap().as_ref() {
            rt.free(self.alloc);
        }
    }
}

fn check_range(buf: &Buffer, offset: u64, nbytes: u64) -> PyResult<()> {
    if offset.checked_add(nbytes).is_none_or(|end| end > buf.alloc.size) {
        return Err(PyValueError::new_err(format!("range {offset}+{nbytes} outside buffer of {} bytes", buf.alloc.size)));
    }
    Ok(())
}

/// Allocates `nbytes` of device memory.
#[pyfunction]
fn alloc(nbytes: u64) -> PyResult<Buffer> {
    Ok(Buffer { alloc: runtime()?.alloc(nbytes).map_err(to_py)? })
}

/// Copies `nbytes` from host address `src` into `buf` at `offset`.
/// Waits for queued GPU work first.
#[pyfunction]
fn upload(buf: &Buffer, offset: u64, src: usize, nbytes: u64) -> PyResult<()> {
    check_range(buf, offset, nbytes)?;
    runtime()?.synchronize().map_err(to_py)?;
    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, buf.alloc.host_ptr().add(offset as usize), nbytes as usize) };
    Ok(())
}

/// Copies `nbytes` from `buf` at `offset` to host address `dst`.
/// Waits for queued GPU work first.
#[pyfunction]
fn download(buf: &Buffer, offset: u64, dst: usize, nbytes: u64) -> PyResult<()> {
    check_range(buf, offset, nbytes)?;
    runtime()?.synchronize().map_err(to_py)?;
    unsafe { std::ptr::copy_nonoverlapping(buf.alloc.host_ptr().add(offset as usize), dst as *mut u8, nbytes as usize) };
    Ok(())
}

/// Queues kernel `name`, specialized with `spec`, over `groups` workgroups,
/// with `push` (at most 128 bytes) as its push constants.
#[pyfunction]
fn dispatch(name: &str, spec: Vec<u32>, push: &[u8], groups: u32) -> PyResult<()> {
    if push.len() > PUSH_CONSTANT_BYTES as usize {
        return Err(PyValueError::new_err(format!("push constants are {} bytes; max {PUSH_CONSTANT_BYTES}", push.len())));
    }
    let rt = runtime()?;
    let kernel = rt.kernels.lock().unwrap().get(name, &spec).map_err(to_py)?;
    let queued = rt.stream.lock().unwrap().dispatch(&kernel, push, [groups, 1, 1]);
    queued.map_err(to_py)
}

/// (workgroups, grid stride) for a 1-D grid-stride kernel over `n` elements.
#[pyfunction]
fn grid_1d(n: u64) -> PyResult<(u32, u32)> {
    Ok(runtime()?.grid_1d(n, 64))
}

/// Waits until all queued GPU work has finished.
#[pyfunction]
fn synchronize() -> PyResult<()> {
    runtime()?.synchronize().map_err(to_py)
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    Ok(runtime()?.ctx.device_name.clone())
}

/// Names of all compiled kernels.
#[pyfunction]
fn kernels() -> Vec<&'static str> {
    vkcore::KernelCache::names().collect()
}

/// (allocated bytes, reserved bytes, blocks) of the memory pool.
#[pyfunction]
fn memory_stats() -> PyResult<(u64, u64, usize)> {
    let s = runtime()?.pool.lock().unwrap().stats();
    Ok((s.allocated, s.reserved, s.blocks))
}

/// Releases cached, unused pool blocks back to the driver.
#[pyfunction]
fn empty_cache() -> PyResult<()> {
    let rt = runtime()?;
    rt.synchronize().map_err(to_py)?;
    rt.pool.lock().unwrap().trim();
    Ok(())
}

/// Validation-layer warnings/errors so far (needs TORCH_VK_VALIDATE=1).
#[pyfunction]
fn validation_issues() -> usize {
    vkcore::validation_issues()
}

/// Finishes queued work and destroys the runtime (saving the pipeline cache).
#[pyfunction]
fn shutdown() {
    let rt = RUNTIME.lock().unwrap().take();
    drop(rt);
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_function(wrap_pyfunction!(alloc, m)?)?;
    m.add_function(wrap_pyfunction!(upload, m)?)?;
    m.add_function(wrap_pyfunction!(download, m)?)?;
    m.add_function(wrap_pyfunction!(dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(grid_1d, m)?)?;
    m.add_function(wrap_pyfunction!(synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(kernels, m)?)?;
    m.add_function(wrap_pyfunction!(memory_stats, m)?)?;
    m.add_function(wrap_pyfunction!(empty_cache, m)?)?;
    m.add_function(wrap_pyfunction!(validation_issues, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    Ok(())
}
