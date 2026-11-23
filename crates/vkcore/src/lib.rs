//! Vulkan compute core for the torch-vulkan backend.
//!
//! - [`Context`]: one device and compute queue (optionally validated).
//! - [`Pool`]: caching sub-allocator over device-addressable memory, mapped where the driver allows.
//! - [`KernelCache`]: compiled pipelines keyed by (kernel, specialization constants).
//! - [`Stream`]: batched, in-order command submission tracked by a timeline semaphore.
//! - [`Runtime`]: all of the above, shared behind mutexes.

pub mod context;
pub mod declare;
pub mod kernels;
pub mod memory;
pub mod spirv_fix;
pub mod stream;
pub mod wide_address;

use std::sync::{Arc, Mutex};

pub use context::{last_validation_message, validation_issues, validation_requested, version_string, Capabilities, Context};
pub use kernels::{kernel_subgroup_size, Kernel, KernelCache, PUSH_CONSTANT_BYTES};
pub use declare::Declared;
pub use memory::{Allocation, Placement, Pool, PoolStats};
pub use stream::{Access, BarrierMode, BarrierStats, Stream};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The most host staging memory a read or write of unmapped memory uses at
/// once (larger ones go in pieces).
pub const STAGING_BYTES: u64 = 64 << 20;

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
    /// `TORCH_VK_WIDE_ADDRESSES=1`: every dispatch computes its addresses
    /// in 64 bits (for tests of those kernels; docs/log/M56).
    pub wide_always: bool,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        let ctx = Arc::new(Context::new()?);
        let kernels = KernelCache::new(ctx.clone())?;
        let stream = Stream::new(ctx.clone(), kernels.layout())?;
        let wide_always = std::env::var("TORCH_VK_WIDE_ADDRESSES").is_ok_and(|v| !v.is_empty() && v != "0");
        Ok(Self {
            stream: Mutex::new(stream),
            kernels: Mutex::new(kernels),
            pool: Mutex::new(Pool::new(ctx.clone())),
            ctx,
            wide_always,
        })
    }

    /// Whether a dispatch with push constants `push` must compute its
    /// addresses in 64 bits (wide_address.rs; docs/log/M56): an address in
    /// them has more than 4 GiB of its allocation after it, which 32-bit
    /// offsets from it do not reach correctly on every driver. Kernels take
    /// every address they use from their push constants (as for barriers:
    /// declare.rs), so this sees them all.
    pub fn wide_addresses(&self, push: &[u8]) -> bool {
        const REACH: u64 = 1 << 32;
        if self.wide_always {
            return true;
        }
        let pool = self.pool.lock().unwrap();
        let (lo, hi) = pool.span();
        if hi - lo <= REACH {
            return false;
        }
        push.as_chunks::<8>().0.iter().any(|word| {
            let w = u64::from_le_bytes(*word);
            w >= lo && w < hi && matches!(pool.locate(w), memory::Located::Allocation(_, end) if end - w > REACH)
        })
    }

    pub fn alloc(&self, bytes: u64) -> Result<Allocation> {
        // (Lock order pool -> stream: the stream never takes the pool's lock.)
        self.pool.lock().unwrap().alloc(bytes, &mut || self.synchronize())
    }

    /// [`Self::alloc`] in memory of one placement (mapped memory for words
    /// the host reads often, say).
    pub fn alloc_in(&self, bytes: u64, placement: Placement) -> Result<Allocation> {
        self.pool.lock().unwrap().alloc_in(bytes, placement, &mut || self.synchronize())
    }

    /// Returns `a` to the pool. Queued GPU work may still use it: reuse is
    /// stream-ordered (see [`memory`]).
    pub fn free(&self, a: Allocation) {
        self.pool.lock().unwrap().free(a)
    }

    /// The pool memory `[address, address + bytes)`, which must lie in one block.
    fn view(&self, address: u64, bytes: u64) -> Result<Allocation> {
        self.pool.lock().unwrap().view(address, bytes).ok_or_else(|| format!("{address:#x} + {bytes} is not in the pool's memory").into())
    }

    /// Host staging for unmapped memory: a [`Placement::Host`] allocation of
    /// up to [`STAGING_BYTES`], and the bytes it holds.
    fn staging(&self, bytes: u64) -> Result<(Allocation, u64)> {
        let len = bytes.min(STAGING_BYTES);
        Ok((self.alloc_in(len, Placement::Host)?, len))
    }

    /// Copies the pool memory at GPU address `address` to `dst`, once the
    /// queued work is done: from the mapping, or for unmapped memory through
    /// host staging memory, in pieces of at most [`STAGING_BYTES`].
    pub fn read(&self, address: u64, dst: &mut [u8]) -> Result<()> {
        let n = dst.len() as u64;
        self.synchronize()?;
        if n == 0 {
            return Ok(());
        }
        let src = self.view(address, n)?;
        if src.mapped {
            unsafe { std::ptr::copy_nonoverlapping(src.host_ptr(), dst.as_mut_ptr(), dst.len()) };
            return Ok(());
        }
        let (staging, len) = self.staging(n)?;
        let result = (|| -> Result<()> {
            let mut done = 0;
            while done < n {
                let k = len.min(n - done);
                self.stream.lock().unwrap().copy(&src, done, &staging, 0, k)?;
                self.synchronize()?;
                unsafe { std::ptr::copy_nonoverlapping(staging.host_ptr(), dst.as_mut_ptr().add(done as usize), k as usize) };
                done += k;
            }
            Ok(())
        })();
        self.free(staging);
        result
    }

    /// Copies `src` to the pool memory at GPU address `address`, after the
    /// queued work (which may still use that memory): into the mapping, or
    /// for unmapped memory through host staging memory and a copy command
    /// (queued: later commands see it).
    pub fn write(&self, address: u64, src: &[u8]) -> Result<()> {
        let n = src.len() as u64;
        self.synchronize()?;
        if n == 0 {
            return Ok(());
        }
        let dst = self.view(address, n)?;
        if dst.mapped {
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst.host_ptr(), src.len()) };
            return Ok(());
        }
        let (staging, len) = self.staging(n)?;
        let result = (|| -> Result<()> {
            let mut done = 0;
            while done < n {
                let k = len.min(n - done);
                if done > 0 {
                    self.synchronize()?; // (the staging memory's last piece copied out)
                }
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().add(done as usize), staging.host_ptr(), k as usize) };
                self.stream.lock().unwrap().copy(&staging, 0, &dst, done, k)?;
                done += k;
            }
            Ok(())
        })();
        // (freed while its copy may be queued: reuse is stream-ordered, and
        // a host write to it synchronizes first, as this one did)
        self.free(staging);
        result
    }

    /// Queues a copy of `bytes` bytes of pool memory from GPU address `src`
    /// to `dst` (each within one block).
    pub fn copy(&self, dst: u64, src: u64, bytes: u64) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let (s, d) = (self.view(src, bytes)?, self.view(dst, bytes)?);
        self.stream.lock().unwrap().copy(&s, 0, &d, 0, bytes)
    }

    /// Queues kernel `name` (specialized with `spec`) with push constants `params`.
    pub fn dispatch<P: Copy>(&self, name: &str, spec: &[u32], params: &P, groups: [u32; 3]) -> Result<()> {
        let wide = self.wide_addresses(as_bytes(params));
        let kernel = self.kernels.lock().unwrap().get(name, spec, wide)?;
        self.stream.lock().unwrap().dispatch(&kernel, as_bytes(params), groups)
    }

    /// Queues kernel `name` (specialized with `spec`) with push constants
    /// `push`, declared to touch the memory its push constants address
    /// ([`declare::push_accesses`]), with the caller's narrower `declared`
    /// ranges for some of those addresses. The stream then puts a barrier
    /// before it only where that memory conflicts with commands since the
    /// last barrier.
    pub fn dispatch_declared(&self, name: &str, spec: &[u32], push: &[u8], groups: [u32; 3], declared: &[Declared]) -> Result<()> {
        let wide = self.wide_addresses(push);
        let kernel = self.kernels.lock().unwrap().get(name, spec, wide)?;
        let mode = self.stream.lock().unwrap().mode();
        if !mode.tracked() {
            return self.stream.lock().unwrap().dispatch(&kernel, push, groups);
        }
        let acc = declare::push_accesses(&self.pool.lock().unwrap(), push, declared);
        if mode == BarrierMode::Check {
            if let Some(acc) = &acc {
                return self.checked_dispatch(&kernel, push, groups, acc);
            }
        }
        self.stream.lock().unwrap().dispatch_declared(&kernel, push, groups, acc.as_deref())
    }

    /// TORCH_VK_BARRIERS=check: runs the dispatch alone (synchronizing
    /// before and after) and fails if it changed any byte of live memory
    /// outside the ranges `acc` declares written. (A write of the value
    /// already there goes unseen, and so do reads: the check covers the
    /// declarations' writes.)
    fn checked_dispatch(&self, kernel: &Kernel, push: &[u8], groups: [u32; 3], acc: &[Access]) -> Result<()> {
        self.synchronize()?;
        let live = self.pool.lock().unwrap().live_allocations();
        let writes = declare::merged(acc.iter().filter(|a| a.write).map(|a| (a.lo, a.hi)).collect());
        let hashes = |live: &[(u64, u64, *const u8)]| -> Vec<u64> {
            live.iter().map(|&(lo, size, host)| unsafe { declare::hash_outside(lo, size, host, &writes) }).collect()
        };
        let before = hashes(&live);
        self.stream.lock().unwrap().dispatch_declared(kernel, push, groups, Some(acc))?;
        self.synchronize()?;
        let after = hashes(&live);
        for (i, &(lo, size, _)) in live.iter().enumerate() {
            if before[i] != after[i] {
                return Err(format!(
                    "TORCH_VK_BARRIERS=check: kernel {} changed memory outside the ranges it declared written: allocation {lo:#x} + {size}; declared {acc:x?}",
                    kernel.name
                )
                .into());
            }
        }
        // An address at an odd 4-byte offset would escape push_accesses.
        let pool = self.pool.lock().unwrap();
        for word in push.get(4..).unwrap_or(&[]).as_chunks::<8>().0 {
            let w = u64::from_le_bytes(*word);
            if let memory::Located::Allocation(..) = pool.locate(w) {
                eprintln!("[torch_vulkan] TORCH_VK_BARRIERS=check: kernel {} has a word {w:#x} at a 4-byte offset inside a live allocation", kernel.name);
            }
        }
        Ok(())
    }

    /// Barriers and commands recorded so far.
    pub fn barrier_stats(&self) -> BarrierStats {
        self.stream.lock().unwrap().stats
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.lock().unwrap().synchronize()
    }

    /// Workgroup count and grid stride for a 1-D grid-stride kernel over `n`
    /// elements with `workgroup` threads per group.
    pub fn grid_1d(&self, n: u64, workgroup: u32) -> (u32, u32) {
        let max = self.ctx.caps.max_workgroup_count[0].min(65_535) as u64;
        let groups = n.div_ceil(workgroup as u64).clamp(1, max) as u32;
        (groups, groups * workgroup)
    }
}
