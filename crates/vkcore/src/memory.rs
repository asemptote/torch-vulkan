//! Caching memory pool: tensors are sub-allocated from a few large buffers
//! instead of one Vulkan allocation each: persistently mapped where the
//! driver allows, else device memory the host reaches through copies
//! ([`Placement`]).
//!
//! Reuse is *stream-ordered*, as in PyTorch's CUDA allocator: memory freed on
//! the host can be handed out again immediately, because every later GPU
//! command on the (single) stream runs after the commands that used it. The
//! host itself must synchronize the stream before reading or writing memory
//! that queued GPU work touches.

use std::collections::BTreeMap;
use std::sync::Arc;

use ash::vk;

use crate::{Context, Result};

/// Alignment of every allocation (offset and size).
pub const ALIGNMENT: u64 = 256;
/// Size of a regular pool block.
pub const BLOCK_SIZE: u64 = 64 << 20;
/// Granularity for blocks made for requests larger than [`BLOCK_SIZE`].
const LARGE_GRANULE: u64 = 2 << 20;
/// Unused bytes before and after each block's allocations. Drivers may load
/// from addresses a lane computes but does not use (lavapipe runs a loop's
/// loads one iteration past its end, e.g. `piv[-1]`); near the ends of a
/// block those land outside the mapping, and the process crashes. Blocks
/// within 2 * GUARD of the device's per-buffer limit get what room is left.
const GUARD: u64 = 1 << 20;

fn round_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

/// Free-range bookkeeping for one block: an offset-ordered free list with
/// best-fit allocation and coalescing on free.
#[derive(Debug)]
pub struct RangeAllocator {
    size: u64,
    free: BTreeMap<u64, u64>, // offset -> length
}

impl RangeAllocator {
    pub fn new(size: u64) -> Self {
        Self { size, free: BTreeMap::from([(0, size)]) }
    }

    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        let (&offset, &len) = self.free.iter().filter(|(_, &len)| len >= size).min_by_key(|(_, &len)| len)?;
        self.free.remove(&offset);
        if len > size {
            self.free.insert(offset + size, len - size);
        }
        Some(offset)
    }

    pub fn free(&mut self, mut offset: u64, mut size: u64) {
        if let Some((&prev, &prev_len)) = self.free.range(..offset).next_back() {
            debug_assert!(prev + prev_len <= offset, "double free or overlap at {offset}");
            if prev + prev_len == offset {
                self.free.remove(&prev);
                offset = prev;
                size += prev_len;
            }
        }
        if let Some(next_len) = self.free.remove(&(offset + size)) {
            size += next_len;
        }
        self.free.insert(offset, size);
    }

    /// The length of the free range [`Self::alloc`] would take for `size`
    /// bytes (the smallest that holds them), if any.
    pub fn best(&self, size: u64) -> Option<u64> {
        self.free.values().copied().filter(|&len| len >= size).min()
    }

    pub fn free_bytes(&self) -> u64 {
        self.free.values().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.free.get(&0) == Some(&self.size)
    }
}

/// Where a block's memory is, and whether the host maps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// Host-visible memory, mapped (the host reads and writes it directly):
    /// device-local where the device has such a memory type. On discrete
    /// GPUs that is the memory the host reaches through the PCIe BAR, which
    /// drivers limit: 246 MiB without resizable BAR, 34 GiB of the H100
    /// NVL's 94 (docs/log/M53).
    Mapped,
    /// Device-local memory the host does not map, for tensors past what
    /// mapped memory holds: the host reads and writes it through copies to
    /// and from [`Placement::Host`] memory. Its allocations' host pointers
    /// are addresses no process memory has (see [`UNMAPPED_BASE`]), only
    /// keys for the GPU address.
    Unmapped,
    /// Host memory (cached where possible), mapped: staging for unmapped
    /// memory.
    Host,
}

/// The host addresses unmapped blocks' allocations get start here: past
/// every 64-bit platform's user address space (2^56: non-canonical on
/// x86-64, beyond Arm64's 48- or 52-bit addresses), so they never alias
/// process memory, and a stray host access faults. (32-bit hosts have no
/// unmapped memory.)
pub const UNMAPPED_BASE: u64 = 1 << 56;

#[cfg(target_pointer_width = "64")]
static NEXT_UNMAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(UNMAPPED_BASE);

/// `TORCH_VK_UNMAPPED=1`: every tensor in unmapped memory, as tensors past
/// the mapped memory's limit are: for testing the host's copies to and from
/// it on any device.
pub fn all_unmapped() -> bool {
    static ALL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALL.get_or_init(|| std::env::var("TORCH_VK_UNMAPPED").is_ok_and(|v| !v.is_empty() && v != "0"))
}

struct Block {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// The mapping (for unmapped memory: the block's key addresses).
    host: *mut u8,
    address: vk::DeviceAddress,
    size: u64,
    guard: u64,
    ranges: RangeAllocator,
    placement: Placement,
}

impl Block {
    /// Whether GPU address `address` is in this block's buffer (guards included).
    fn contains(&self, address: u64) -> bool {
        address >= self.address && address - self.address < self.size + 2 * self.guard
    }
}

/// A sub-allocation. Plain data: give it back with [`Pool::free`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub block: u32,
    /// Offset of this allocation inside `buffer`.
    pub offset: u64,
    pub size: u64,
    /// GPU address of the first byte, usable as a shader pointer.
    pub address: vk::DeviceAddress,
    /// The block's buffer, for transfer commands.
    pub buffer: vk::Buffer,
    /// Whether the host maps the memory (not for [`Placement::Unmapped`]).
    pub mapped: bool,
    host: usize,
}

impl Allocation {
    /// Host pointer to the first byte: the pool's mapped memory is
    /// persistently mapped; for unmapped memory, a key that must not be
    /// dereferenced.
    pub fn host_ptr(&self) -> *mut u8 {
        self.host as *mut u8
    }

    /// Host view of the allocation (mapped memory only).
    ///
    /// # Safety
    /// No queued or running GPU work may access this memory while the slice
    /// is alive (synchronize the stream first), and the allocation must not
    /// have been freed.
    // The memory is owned by the pool's mapping, not borrowed from `self`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_slice_mut<T: Copy>(&self) -> &mut [T] {
        assert!(self.mapped, "host view of unmapped memory");
        std::slice::from_raw_parts_mut(self.host as *mut T, self.size as usize / std::mem::size_of::<T>())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Bytes handed out (after rounding to [`ALIGNMENT`]).
    pub allocated: u64,
    /// Bytes held from Vulkan.
    pub reserved: u64,
    pub blocks: usize,
    /// The most bytes handed out at once since the pool started or
    /// [`Pool::reset_peak`].
    pub peak: u64,
    /// Of `reserved`, the bytes in unmapped blocks.
    pub unmapped: u64,
}

pub struct Pool {
    ctx: Arc<Context>,
    blocks: Vec<Option<Block>>,
    allocated: u64,
    peak: u64,
    /// Live allocations by GPU address (address -> size), for [`Pool::locate`].
    live: BTreeMap<u64, u64>,
    /// A new mapped block failed (the driver's limit on mapped memory, or its
    /// heap full): tensors' new blocks are unmapped until a mapped block is
    /// released, rather than each first failing a mapped allocation.
    mapped_full: bool,
}

/// What a GPU address points into (see [`Pool::locate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Located {
    /// The live allocation `[lo, hi)` (GPU addresses) containing it.
    Allocation(u64, u64),
    /// A pool block's memory, but no live allocation: free or guard bytes.
    Unallocated,
    /// None of the pool's memory.
    Outside,
}

// Block host pointers are only dereferenced through `Allocation`s, whose
// synchronization contract is documented above.
unsafe impl Send for Pool {}

impl Pool {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self { ctx, blocks: Vec::new(), allocated: 0, peak: 0, live: BTreeMap::new(), mapped_full: false }
    }

    /// Sub-allocates `size` bytes for a tensor: in mapped memory, or once
    /// the driver maps no more, in unmapped device memory (all of it
    /// unmapped with [`all_unmapped`]). Grows the pool if no cached block
    /// has room. Releasing cached blocks first calls `wait_idle`, which must
    /// wait for queued GPU work: a freed allocation may still be in use by it.
    pub fn alloc(&mut self, size: u64, wait_idle: &mut dyn FnMut() -> Result<()>) -> Result<Allocation> {
        let unmapped_ok = cfg!(target_pointer_width = "64");
        let order: &[Placement] = if all_unmapped() && unmapped_ok {
            &[Placement::Unmapped]
        } else if unmapped_ok {
            &[Placement::Mapped, Placement::Unmapped]
        } else {
            &[Placement::Mapped]
        };
        self.alloc_placed(size, order, wait_idle)
    }

    /// Sub-allocates `size` bytes in memory of one placement.
    pub fn alloc_in(&mut self, size: u64, placement: Placement, wait_idle: &mut dyn FnMut() -> Result<()>) -> Result<Allocation> {
        self.alloc_placed(size, &[placement], wait_idle)
    }

    fn alloc_placed(&mut self, size: u64, order: &[Placement], wait_idle: &mut dyn FnMut() -> Result<()>) -> Result<Allocation> {
        let size = round_up(size.max(1), ALIGNMENT);
        // One block is one buffer, which may not be larger than
        // maxBufferSize. (maxMemoryAllocationSize is the size past which an
        // allocation may fail: larger blocks are tried, and the error names
        // it when they fail.)
        let limit = self.ctx.caps.max_buffer_size;
        if size > limit {
            return Err(format!(
                "an allocation of {} is larger than the device's largest buffer (maxBufferSize, {})",
                bytes(size),
                bytes(limit)
            )
            .into());
        }
        for &p in order {
            if let Some(a) = self.alloc_from_existing(size, p) {
                return Ok(a);
            }
        }
        let block_size = round_up(size.max(BLOCK_SIZE), LARGE_GRANULE).min(limit);
        // Cached empty blocks too small for this request go back to Vulkan
        // first when together they would cover it: otherwise every size a
        // sequence of growing requests passes through stays reserved (a
        // blocked LU's trailing matrices held 3 GB of idle blocks).
        let idle: u64 = self.blocks.iter().flatten().filter(|b| b.ranges.is_empty()).map(|b| b.size).sum();
        if idle >= block_size {
            wait_idle()?;
            self.trim();
        }
        let guard = GUARD.min((limit - block_size) / 2 / ALIGNMENT * ALIGNMENT);
        let mut failures = Vec::new();
        for &p in order {
            if p == Placement::Mapped && self.mapped_full && order.len() > 1 {
                failures.push((p, "refused earlier, no mapped block released since".to_string()));
                continue;
            }
            let block = match self.create_block(block_size, guard, p) {
                Err(_) if self.blocks.iter().flatten().any(|b| b.ranges.is_empty()) => {
                    // Out of memory: release cached empty blocks and retry once.
                    wait_idle()?;
                    self.trim();
                    self.create_block(block_size, guard, p)
                }
                other => other,
            };
            match block {
                Ok(block) => {
                    let i = match self.blocks.iter().position(Option::is_none) {
                        Some(i) => {
                            self.blocks[i] = Some(block);
                            i
                        }
                        None => {
                            self.blocks.push(Some(block));
                            self.blocks.len() - 1
                        }
                    };
                    return Ok(self.alloc_in_block(i, size).expect("fresh block fits the request"));
                }
                Err(e) => {
                    if p == Placement::Mapped {
                        self.mapped_full = true;
                    }
                    failures.push((p, e.to_string()));
                }
            }
        }
        Err(self.out_of_memory(size, block_size, &failures).into())
    }

    /// The error for a request no block could take: what was asked, what
    /// the pool holds, and why each kind of memory refused.
    fn out_of_memory(&self, size: u64, block_size: u64, failures: &[(Placement, String)]) -> String {
        let s = self.stats();
        let heaps = self.ctx.memory_heaps();
        let mut msg = format!(
            "out of device memory allocating {} (a new block of {}); the pool holds {} in {} blocks ({} of it unmapped), {} of it in use by tensors",
            bytes(size),
            bytes(block_size),
            bytes(s.reserved),
            s.blocks,
            bytes(s.unmapped),
            bytes(s.allocated)
        );
        for (p, e) in failures {
            let kind = match p {
                Placement::Mapped => "mapped memory",
                Placement::Unmapped => "unmapped device memory",
                Placement::Host => "host memory",
            };
            msg += &format!("; {kind}: {e}");
            if let Some(t) = self.memory_type_for(*p, u32::MAX) {
                let heap = self.ctx.memory_types()[t as usize].heap_index as usize;
                msg += &format!(" (memory type {t}, heap {heap} of {})", bytes(heaps[heap].size));
            }
        }
        let max_alloc = self.ctx.caps.max_allocation_size;
        if block_size > max_alloc {
            msg += &format!(
                "; the request is larger than the device's maxMemoryAllocationSize ({}), past which allocations may fail",
                bytes(max_alloc)
            );
        }
        msg + ". Free tensors no longer needed (torch_vulkan.empty_cache() returns cached blocks to the driver), or use smaller batches."
    }

    /// The smallest free range, over all blocks of placement `p`, that
    /// holds `size` bytes: best fit across blocks, so that a large empty
    /// block is not split by small requests while a tighter range elsewhere
    /// would do.
    fn alloc_from_existing(&mut self, size: u64, p: Placement) -> Option<Allocation> {
        let (_, i) = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().filter(|b| b.placement == p).and_then(|b| b.ranges.best(size)).map(|len| (len, i)))
            .min()?;
        self.alloc_in_block(i, size)
    }

    fn alloc_in_block(&mut self, i: usize, size: u64) -> Option<Allocation> {
        let block = self.blocks[i].as_mut()?;
        let offset = block.guard + block.ranges.alloc(size)?; // (within the buffer)
        self.allocated += size;
        self.peak = self.peak.max(self.allocated);
        self.live.insert(block.address + offset, size);
        Some(Allocation {
            block: i as u32,
            offset,
            size,
            address: block.address + offset,
            buffer: block.buffer,
            mapped: block.placement != Placement::Unmapped,
            host: block.host as usize + offset as usize,
        })
    }

    pub fn free(&mut self, a: Allocation) {
        let block = self.blocks[a.block as usize].as_mut().expect("allocation from a released block");
        block.ranges.free(a.offset - block.guard, a.size);
        self.allocated -= a.size;
        self.live.remove(&a.address);
    }

    /// The live allocation containing GPU address `address`, if any, else
    /// whether it lies in the pool's memory at all.
    pub fn locate(&self, address: u64) -> Located {
        if let Some((&lo, &size)) = self.live.range(..=address).next_back() {
            if address - lo < size {
                return Located::Allocation(lo, lo + size);
            }
        }
        if self.blocks.iter().flatten().any(|b| b.contains(address)) {
            Located::Unallocated
        } else {
            Located::Outside
        }
    }

    /// `[address, address + len)` in one block, as an [`Allocation`] (not a
    /// live one: for transfer commands and host access), or None if it is
    /// not within one of the pool's blocks.
    pub fn view(&self, address: u64, len: u64) -> Option<Allocation> {
        let (i, b) = self.blocks.iter().enumerate().find_map(|(i, s)| s.as_ref().filter(|b| b.contains(address)).map(|b| (i, b)))?;
        let offset = address - b.address;
        if offset + len > b.size + 2 * b.guard {
            return None;
        }
        Some(Allocation {
            block: i as u32,
            offset,
            size: len,
            address,
            buffer: b.buffer,
            mapped: b.placement != Placement::Unmapped,
            host: b.host as usize + offset as usize,
        })
    }

    /// The GPU addresses `[lo, hi)` spanning all of the pool's blocks (empty
    /// without blocks): no address outside is the pool's.
    pub fn span(&self) -> (u64, u64) {
        let lo = self.blocks.iter().flatten().map(|b| b.address).min().unwrap_or(0);
        let hi = self.blocks.iter().flatten().map(|b| b.address + b.size + 2 * b.guard).max().unwrap_or(0);
        (lo, hi)
    }

    /// Every live allocation in mapped memory: its GPU address, size and
    /// host pointer.
    pub fn live_allocations(&self) -> Vec<(u64, u64, *const u8)> {
        let mut out = Vec::with_capacity(self.live.len());
        for (&address, &size) in &self.live {
            if let Some(b) = self.blocks.iter().flatten().find(|b| b.contains(address) && b.placement != Placement::Unmapped) {
                out.push((address, size, b.host.wrapping_add((address - b.address) as usize) as *const u8));
            }
        }
        out
    }

    /// Returns cached blocks that hold no allocations to Vulkan. Queued GPU
    /// work must not use them (synchronize the stream first).
    pub fn trim(&mut self) {
        let empty: Vec<Block> =
            self.blocks.iter_mut().filter(|slot| slot.as_ref().is_some_and(|b| b.ranges.is_empty())).filter_map(Option::take).collect();
        for b in empty {
            if b.placement == Placement::Mapped {
                self.mapped_full = false; // (room again, perhaps)
            }
            self.destroy_block(b);
        }
    }

    pub fn stats(&self) -> PoolStats {
        let live = self.blocks.iter().flatten();
        PoolStats {
            allocated: self.allocated,
            reserved: live.clone().map(|b| b.size).sum(),
            blocks: live.clone().count(),
            peak: self.peak,
            unmapped: live.filter(|b| b.placement == Placement::Unmapped).map(|b| b.size).sum(),
        }
    }

    /// Restarts the peak of [`PoolStats`] from the bytes allocated now.
    pub fn reset_peak(&mut self) {
        self.peak = self.allocated;
    }

    /// The memory type blocks of placement `p` take (among `bits`).
    /// Mapped: host-visible and coherent, device-local where it can be
    /// (cached too where it can be). Unmapped: device-local, and not
    /// host-visible where it can be (drivers limit the host-visible kind).
    /// Host: host-visible and coherent, cached and not device-local where
    /// it can be.
    fn memory_type_for(&self, p: Placement, bits: u32) -> Option<u32> {
        let types = self.ctx.memory_types();
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let cached = vk::MemoryPropertyFlags::HOST_CACHED;
        let none = vk::MemoryPropertyFlags::empty();
        let find = |want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags| {
            (0..types.len() as u32).find(|&i| {
                let f = types[i as usize].property_flags;
                bits & (1 << i) != 0 && f.contains(want) && !f.intersects(avoid)
            })
        };
        match p {
            Placement::Mapped => find(host | local | cached, none).or_else(|| find(host | local, none)).or_else(|| find(host, none)),
            Placement::Unmapped => find(local, vk::MemoryPropertyFlags::HOST_VISIBLE).or_else(|| find(local, none)),
            Placement::Host => find(host | cached, local).or_else(|| find(host, local)).or_else(|| find(host, none)),
        }
    }

    fn create_block(&self, size: u64, guard: u64, placement: Placement) -> Result<Block> {
        let d = &self.ctx.device;
        unsafe {
            let buffer = d.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size + 2 * guard)
                    .usage(
                        vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = d.get_buffer_memory_requirements(buffer);
            let Some(type_index) = self.memory_type_for(placement, req.memory_type_bits) else {
                d.destroy_buffer(buffer, None);
                return Err(format!("no memory type for {placement:?} memory").into());
            };
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let info = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(type_index).push_next(&mut flags);
            let memory = match d.allocate_memory(&info, None) {
                Ok(m) => m,
                Err(e) => {
                    d.destroy_buffer(buffer, None);
                    return Err(e.into());
                }
            };
            if let Err(e) = d.bind_buffer_memory(buffer, memory, 0) {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
                return Err(e.into());
            }
            let host = if placement == Placement::Unmapped {
                unmapped_keys(size + 2 * guard)
            } else {
                match d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
                    Ok(p) => p as *mut u8,
                    Err(e) => {
                        d.destroy_buffer(buffer, None);
                        d.free_memory(memory, None);
                        return Err(e.into());
                    }
                }
            };
            let address = self.ctx.buffer_device_address(buffer);
            Ok(Block { buffer, memory, host, address, size, guard, ranges: RangeAllocator::new(size), placement })
        }
    }

    fn destroy_block(&self, b: Block) {
        unsafe {
            let d = &self.ctx.device;
            if b.placement != Placement::Unmapped {
                d.unmap_memory(b.memory);
            }
            d.destroy_buffer(b.buffer, None);
            d.free_memory(b.memory, None);
        }
    }
}

/// Key addresses for an unmapped block of `span` bytes (never reused; a
/// page apart).
#[cfg(target_pointer_width = "64")]
fn unmapped_keys(span: u64) -> *mut u8 {
    NEXT_UNMAPPED.fetch_add(round_up(span, 4096) + 4096, std::sync::atomic::Ordering::Relaxed) as usize as *mut u8
}
#[cfg(not(target_pointer_width = "64"))]
fn unmapped_keys(_span: u64) -> *mut u8 {
    unreachable!("no unmapped memory on 32-bit hosts")
}

/// A byte count for messages: exact, and in GiB or MiB.
fn bytes(n: u64) -> String {
    if n >= 1 << 30 {
        format!("{n} bytes ({:.2} GiB)", n as f64 / (1u64 << 30) as f64)
    } else {
        format!("{n} bytes ({:.1} MiB)", n as f64 / (1u64 << 20) as f64)
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for b in std::mem::take(&mut self.blocks).into_iter().flatten() {
            self.destroy_block(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAllocator;

    #[test]
    fn best_fit_and_coalescing() {
        let mut r = RangeAllocator::new(1024);
        let a = r.alloc(256).unwrap();
        let b = r.alloc(256).unwrap();
        let c = r.alloc(256).unwrap();
        assert_eq!((a, b, c), (0, 256, 512));
        r.free(b, 256);
        r.free(c, 256); // b, c and the tail coalesce into [256, 1024)
        assert_eq!(r.free_bytes(), 768);
        assert_eq!(r.alloc(768), Some(256));
        r.free(256, 768);
        r.free(a, 256);
        assert!(r.is_empty());
    }

    #[test]
    fn picks_smallest_hole() {
        let mut r = RangeAllocator::new(4096);
        let offs: Vec<u64> = (0..4).map(|_| r.alloc(1024).unwrap()).collect();
        r.free(offs[0], 1024); // hole [0, 1024)
        r.free(offs[2], 1024);
        r.free(offs[3], 1024); // coalesced hole [2048, 4096)
        assert_eq!(r.alloc(512), Some(0)); // smallest hole that fits
        assert_eq!(r.alloc(2048), Some(2048));
        assert_eq!(r.alloc(512), Some(512)); // rest of the first hole
        assert_eq!(r.alloc(1), None);
    }

    #[test]
    fn exhaustion() {
        let mut r = RangeAllocator::new(512);
        assert!(r.alloc(512).is_some());
        assert!(r.alloc(1).is_none());
    }
}
