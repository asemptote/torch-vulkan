//! Caching memory pool: tensors are sub-allocated from a few large,
//! persistently mapped buffers instead of one Vulkan allocation each.
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

    pub fn free_bytes(&self) -> u64 {
        self.free.values().sum()
    }

    pub fn is_empty(&self) -> bool {
        self.free.get(&0) == Some(&self.size)
    }
}

struct Block {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    host: *mut u8,
    address: vk::DeviceAddress,
    size: u64,
    guard: u64,
    ranges: RangeAllocator,
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
    host: usize,
}

impl Allocation {
    /// Host pointer to the first byte (the pool's memory is persistently mapped).
    pub fn host_ptr(&self) -> *mut u8 {
        self.host as *mut u8
    }

    /// Host view of the allocation.
    ///
    /// # Safety
    /// No queued or running GPU work may access this memory while the slice
    /// is alive (synchronize the stream first), and the allocation must not
    /// have been freed.
    // The memory is owned by the pool's mapping, not borrowed from `self`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn as_slice_mut<T: Copy>(&self) -> &mut [T] {
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
}

pub struct Pool {
    ctx: Arc<Context>,
    blocks: Vec<Option<Block>>,
    allocated: u64,
}

// Block host pointers are only dereferenced through `Allocation`s, whose
// synchronization contract is documented above.
unsafe impl Send for Pool {}

impl Pool {
    pub fn new(ctx: Arc<Context>) -> Self {
        Self { ctx, blocks: Vec::new(), allocated: 0 }
    }

    /// Sub-allocates `size` bytes, growing the pool if no cached block has
    /// room. Releasing cached blocks first calls `wait_idle`, which must wait
    /// for queued GPU work: a freed allocation may still be in use by it.
    pub fn alloc(&mut self, size: u64, wait_idle: &mut dyn FnMut() -> Result<()>) -> Result<Allocation> {
        let size = round_up(size.max(1), ALIGNMENT);
        // One block is one buffer and one memory allocation: both limits apply.
        let limit = self.ctx.caps.max_buffer_size.min(self.ctx.caps.max_allocation_size);
        if size > limit {
            return Err(format!("allocation of {size} bytes exceeds the device's limit of {limit} bytes per buffer").into());
        }
        if let Some(a) = self.alloc_from_existing(size) {
            return Ok(a);
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
        let block = match self.create_block(block_size, guard) {
            Ok(b) => b,
            Err(_) => {
                // Out of memory: release cached empty blocks and retry once.
                wait_idle()?;
                self.trim();
                self.create_block(block_size, guard)?
            }
        };
        match self.blocks.iter().position(Option::is_none) {
            Some(i) => self.blocks[i] = Some(block),
            None => self.blocks.push(Some(block)),
        }
        Ok(self.alloc_from_existing(size).expect("fresh block fits the request"))
    }

    fn alloc_from_existing(&mut self, size: u64) -> Option<Allocation> {
        for (i, slot) in self.blocks.iter_mut().enumerate() {
            let Some(block) = slot else { continue };
            if let Some(offset) = block.ranges.alloc(size) {
                let offset = block.guard + offset;  // (within the buffer)
                self.allocated += size;
                return Some(Allocation {
                    block: i as u32,
                    offset,
                    size,
                    address: block.address + offset,
                    buffer: block.buffer,
                    host: block.host as usize + offset as usize,
                });
            }
        }
        None
    }

    pub fn free(&mut self, a: Allocation) {
        let block = self.blocks[a.block as usize].as_mut().expect("allocation from a released block");
        block.ranges.free(a.offset - block.guard, a.size);
        self.allocated -= a.size;
    }

    /// Returns cached blocks that hold no allocations to Vulkan. Queued GPU
    /// work must not use them (synchronize the stream first).
    pub fn trim(&mut self) {
        let empty: Vec<Block> =
            self.blocks.iter_mut().filter(|slot| slot.as_ref().is_some_and(|b| b.ranges.is_empty())).filter_map(Option::take).collect();
        for b in empty {
            self.destroy_block(b);
        }
    }

    pub fn stats(&self) -> PoolStats {
        let live = self.blocks.iter().flatten();
        PoolStats { allocated: self.allocated, reserved: live.clone().map(|b| b.size).sum(), blocks: live.count() }
    }

    fn create_block(&self, size: u64, guard: u64) -> Result<Block> {
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
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let Some(type_index) = self
                .ctx
                .memory_type(req.memory_type_bits, host | local | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| self.ctx.memory_type(req.memory_type_bits, host | local))
                .or_else(|| self.ctx.memory_type(req.memory_type_bits, host))
            else {
                d.destroy_buffer(buffer, None);
                return Err("no host-visible memory type".into());
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
            d.bind_buffer_memory(buffer, memory, 0)?;
            let host = d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;
            let address = self.ctx.buffer_device_address(buffer);
            Ok(Block { buffer, memory, host, address, size, guard, ranges: RangeAllocator::new(size) })
        }
    }

    fn destroy_block(&self, b: Block) {
        unsafe {
            let d = &self.ctx.device;
            d.unmap_memory(b.memory);
            d.destroy_buffer(b.buffer, None);
            d.free_memory(b.memory, None);
        }
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
