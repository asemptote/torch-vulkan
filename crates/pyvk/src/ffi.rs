//! C ABI used by the C++ PyTorch backend (csrc/) to reach the Rust runtime.
//!
//! Every function returns 0 on success; on failure it returns -1 and the
//! message is available from `vkt_last_error` on the same thread.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};

use vkcore::Allocation;

use crate::{runtime, RUNTIME};

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(message: impl std::fmt::Display) -> i32 {
    let text = CString::new(message.to_string().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = text);
    -1
}

fn status<T, E: std::fmt::Display>(r: Result<T, E>) -> i32 {
    match r {
        Ok(_) => 0,
        Err(e) => fail(e),
    }
}

/// An allocation as seen from C++.
#[repr(C)]
pub struct VktAlloc {
    /// Opaque handle to pass back to `vkt_free`.
    pub handle: *mut c_void,
    /// Host pointer to the first byte (memory is persistently mapped).
    pub host: *mut u8,
    /// GPU address of the first byte.
    pub address: u64,
    pub size: u64,
}

#[no_mangle]
pub extern "C" fn vkt_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn vkt_alloc(nbytes: u64, out: *mut VktAlloc) -> i32 {
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(e) => return fail(e),
    };
    match rt.alloc(nbytes) {
        Ok(a) => {
            *out = VktAlloc { handle: Box::into_raw(Box::new(a)) as *mut c_void, host: a.host_ptr(), address: a.address, size: a.size };
            0
        }
        Err(e) => fail(e),
    }
}

/// Returns an allocation to the pool (stream-ordered). Safe after shutdown.
#[no_mangle]
pub unsafe extern "C" fn vkt_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let alloc = *Box::from_raw(handle as *mut Allocation);
    if let Some(rt) = RUNTIME.lock().unwrap().as_ref() {
        rt.free(alloc);
    }
}

#[no_mangle]
pub extern "C" fn vkt_synchronize() -> i32 {
    match RUNTIME.lock().unwrap().clone() {
        Some(rt) => status(rt.synchronize()),
        None => 0,
    }
}

/// Queues kernel `name` specialized with `spec[..nspec]`, with `push[..npush]`
/// as push constants, over a gx x gy x gz grid of workgroups.
#[no_mangle]
pub unsafe extern "C" fn vkt_dispatch(
    name: *const c_char,
    spec: *const u32,
    nspec: usize,
    push: *const u8,
    npush: usize,
    gx: u32,
    gy: u32,
    gz: u32,
) -> i32 {
    let rt = match runtime() {
        Ok(rt) => rt,
        Err(e) => return fail(e),
    };
    let Ok(name) = CStr::from_ptr(name).to_str() else { return fail("kernel name is not UTF-8") };
    let spec = if nspec == 0 { &[][..] } else { std::slice::from_raw_parts(spec, nspec) };
    let push = std::slice::from_raw_parts(push, npush);
    let kernel = match rt.kernels.lock().unwrap().get(name, spec) {
        Ok(k) => k,
        Err(e) => return fail(e),
    };
    let queued = rt.stream.lock().unwrap().dispatch(&kernel, push, [gx, gy, gz]);
    if queued.is_ok() && trace() {
        // Each kernel runs to completion before the next is queued, so a
        // kernel that crashes the driver is the last one printed.
        eprintln!("[torch_vulkan] {name} {spec:?} grid {gx}x{gy}x{gz}");
        return status(rt.synchronize());
    }
    status(queued)
}

/// `TORCH_VK_TRACE=1`: print every kernel launch and wait for it to finish.
fn trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("TORCH_VK_TRACE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// Workgroups and grid stride for a grid-stride kernel over `n` elements.
#[no_mangle]
pub unsafe extern "C" fn vkt_grid_1d(n: u64, groups: *mut u32, stride: *mut u32) -> i32 {
    match runtime() {
        Ok(rt) => {
            let (g, s) = rt.grid_1d(n, 64);
            *groups = g;
            *stride = s;
            0
        }
        Err(e) => fail(e),
    }
}

/// 1 if the device supports float64 shader arithmetic.
#[no_mangle]
pub extern "C" fn vkt_has_float64() -> i32 {
    runtime().map(|rt| rt.ctx.caps.shader_float64 as i32).unwrap_or(0)
}

/// Optional device features as bits (csrc/vk.h `Feature`): 1 float64,
/// 2 int16 (with 16-bit storage), 4 float16 (with 16-bit storage), 8 int8
/// (with 8-bit storage).
#[no_mangle]
pub extern "C" fn vkt_capabilities() -> u32 {
    let Ok(rt) = runtime() else { return 0 };
    let c = rt.ctx.caps;
    (c.shader_float64 as u32)
        | (((c.shader_int16 && c.storage_buffer_16bit) as u32) << 1)
        | (((c.shader_float16 && c.storage_buffer_16bit) as u32) << 2)
        | (((c.shader_int8 && c.storage_buffer_8bit) as u32) << 3)
}
