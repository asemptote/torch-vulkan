//! Vulkan compute core for the torch-vulkan backend.
//!
//! - [`Context`]: one device and compute queue (optionally validated).
//! - [`Pool`]: caching sub-allocator over persistently mapped, device-addressable memory.
//! - [`KernelCache`]: compiled pipelines keyed by (kernel, specialization constants).
//! - [`Stream`]: batched, in-order command submission tracked by a timeline semaphore.
//! - [`Runtime`]: all of the above, shared behind mutexes.

pub mod context;
pub mod kernels;
pub mod memory;
pub mod stream;

use std::sync::{Arc, Mutex};

pub use context::{last_validation_message, validation_issues, validation_requested, version_string, Capabilities, Context};
pub use kernels::{Kernel, KernelCache, PUSH_CONSTANT_BYTES};
pub use memory::{Allocation, Pool, PoolStats};
pub use stream::Stream;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The bytes of a `#[repr(C)]` push-constant struct.
pub fn as_bytes<P: Copy>(p: &P) -> &[u8] {
    unsafe { std::slice::from_raw_parts(p as *const P as *const u8, std::mem::size_of::<P>()) }
}

/// Device, memory pool, kernel cache and stream for one GPU.
///
/// Fields drop in declaration order: the stream finishes its work before
/// pipelines and memory are released.
pub struct Runtime {
    pub stream: Mutex<Stream>,
    pub kernels: Mutex<KernelCache>,
    pub pool: Mutex<Pool>,
    pub ctx: Arc<Context>,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        let ctx = Arc::new(Context::new()?);
        let kernels = KernelCache::new(ctx.clone())?;
        let stream = Stream::new(ctx.clone(), kernels.layout())?;
        Ok(Self { stream: Mutex::new(stream), kernels: Mutex::new(kernels), pool: Mutex::new(Pool::new(ctx.clone())), ctx })
    }

    pub fn alloc(&self, bytes: u64) -> Result<Allocation> {
        // (Lock order pool -> stream: the stream never takes the pool's lock.)
        self.pool.lock().unwrap().alloc(bytes, &mut || self.synchronize())
    }

    /// Returns `a` to the pool. Queued GPU work may still use it: reuse is
    /// stream-ordered (see [`memory`]).
    pub fn free(&self, a: Allocation) {
        self.pool.lock().unwrap().free(a)
    }

    /// Queues kernel `name` (specialized with `spec`) with push constants `params`.
    pub fn dispatch<P: Copy>(&self, name: &str, spec: &[u32], params: &P, groups: [u32; 3]) -> Result<()> {
        let kernel = self.kernels.lock().unwrap().get(name, spec)?;
        self.stream.lock().unwrap().dispatch(&kernel, as_bytes(params), groups)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.lock().unwrap().synchronize()
    }

    /// Workgroup count and grid stride for a 1-D grid-stride kernel over `n`
    /// elements with `workgroup` threads per group.
    pub fn grid_1d(&self, n: u64, workgroup: u32) -> (u32, u32) {
        let max = self.ctx.caps.max_workgroup_count_x.min(65_535) as u64;
        let groups = n.div_ceil(workgroup as u64).clamp(1, max) as u32;
        (groups, groups * workgroup)
    }
}
