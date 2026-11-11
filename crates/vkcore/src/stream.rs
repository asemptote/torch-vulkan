//! An in-order command stream, like a CUDA stream.
//!
//! Commands are recorded into a command buffer and submitted in batches; a
//! timeline semaphore counts finished batches. Consecutive commands are
//! separated by a full compute/transfer memory barrier, so every command sees
//! the results of the ones before it (simple and safe; finer-grained
//! dependency tracking can come later).

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::kernels::{Kernel, PUSH_CONSTANT_BYTES};
use crate::memory::Allocation;
use crate::{Context, Result};

/// Commands per batch before the stream submits on its own.
pub const DEFAULT_MAX_BATCH: u32 = 256;

/// Workgroups per batch before the stream submits on its own, on Intel GPUs
/// (0: no limit, elsewhere). On Linux, a submission that kept a Gen9 GPU
/// (HD 530, Mesa's ANV, i915) busy for a couple of seconds came back with
/// its later commands' results missing, and no lost device reported: a
/// 2^27-entry sort, whose bitonic passes of 65535 workgroups take about a
/// second each there, was unsorted with two passes per submission and
/// right with one. A big dispatch is then submitted alone (small ones still
/// batch up to DEFAULT_MAX_BATCH), with a margin: 2^13 workgroups.
pub const INTEL_MAX_BATCH_GROUPS: u64 = 1 << 13;

/// Microseconds [`Stream::wait`] polls before blocking: TORCH_VK_SPIN_US, by
/// default 500 (0: block at once).
pub fn spin_us() -> u64 {
    static SPIN: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SPIN.get_or_init(|| std::env::var("TORCH_VK_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(500))
}

pub struct Stream {
    ctx: Arc<Context>,
    layout: vk::PipelineLayout,
    pool: vk::CommandPool,
    timeline: vk::Semaphore,
    recording: Option<vk::CommandBuffer>,
    commands: u32,
    bound: vk::Pipeline,
    /// Timeline value signaled by the most recent submission.
    submitted: u64,
    in_flight: VecDeque<(u64, vk::CommandBuffer)>,
    spare: Vec<vk::CommandBuffer>,
    pub max_batch: u32,
    /// Workgroups per batch before it is submitted (0: no limit; see
    /// INTEL_MAX_BATCH_GROUPS; TORCH_VK_MAX_BATCH_GROUPS overrides).
    pub max_batch_groups: u64,
    batch_groups: u64,
    /// Kernels dispatched since the stream was created.
    pub dispatches: u64,
}

// The stream owns its queue: only one stream is created per Context.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(ctx: Arc<Context>, layout: vk::PipelineLayout) -> Result<Self> {
        unsafe {
            let d = &ctx.device;
            let pool = d.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(ctx.queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let mut kind = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);
            let timeline = d.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut kind), None)?;
            let max_batch_groups = std::env::var("TORCH_VK_MAX_BATCH_GROUPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if ctx.vendor_id == 0x8086 { INTEL_MAX_BATCH_GROUPS } else { 0 });
            Ok(Self {
                ctx,
                layout,
                pool,
                timeline,
                recording: None,
                commands: 0,
                bound: vk::Pipeline::null(),
                submitted: 0,
                in_flight: VecDeque::new(),
                spare: Vec::new(),
                max_batch: DEFAULT_MAX_BATCH,
                max_batch_groups,
                batch_groups: 0,
                dispatches: 0,
            })
        }
    }

    /// The command buffer being recorded, starting one if needed, with a
    /// barrier against the previous command (in this batch or an earlier one).
    fn begin_command(&mut self) -> Result<vk::CommandBuffer> {
        let ctx = self.ctx.clone();
        let d = &ctx.device;
        let cb = match self.recording {
            Some(cb) => cb,
            None => unsafe {
                self.recycle()?;
                let cb = match self.spare.pop() {
                    Some(cb) => cb,
                    None => d.allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(self.pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?[0],
                };
                d.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
                self.recording = Some(cb);
                self.bound = vk::Pipeline::null();
                cb
            },
        };
        // Also before a batch's first command: submission order alone orders
        // nothing, and a barrier's first scope reaches back into earlier
        // submissions on the queue. (Without it, lavapipe could start a batch's
        // first dispatch while the previous batch's last one still ran.)
        {
            let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                );
            unsafe { d.cmd_pipeline_barrier(cb, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
        }
        Ok(cb)
    }

    fn end_command(&mut self) -> Result<()> {
        self.commands += 1;
        if self.commands >= self.max_batch || (self.max_batch_groups > 0 && self.batch_groups >= self.max_batch_groups) {
            self.flush()?;
        }
        Ok(())
    }

    /// Queues `kernel` over `groups` workgroups with `push` (at most
    /// [`PUSH_CONSTANT_BYTES`] bytes) as its push constants.
    pub fn dispatch(&mut self, kernel: &Kernel, push: &[u8], groups: [u32; 3]) -> Result<()> {
        assert!(push.len() <= PUSH_CONSTANT_BYTES as usize, "push constants too large for {}", kernel.name);
        self.dispatches += 1;
        // Always set the whole range the layout declares, so no byte a kernel
        // might read is left undefined.
        let mut padded = [0u8; PUSH_CONSTANT_BYTES as usize];
        padded[..push.len()].copy_from_slice(push);

        let cb = self.begin_command()?;
        self.batch_groups += groups[0] as u64 * groups[1] as u64 * groups[2] as u64;
        let d = &self.ctx.device;
        unsafe {
            if self.bound != kernel.pipeline {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
                self.bound = kernel.pipeline;
            }
            d.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, &padded);
            d.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
        }
        self.end_command()
    }

    /// Queues a copy of `bytes` bytes from `src + src_offset` to `dst + dst_offset`.
    pub fn copy(&mut self, src: &Allocation, src_offset: u64, dst: &Allocation, dst_offset: u64, bytes: u64) -> Result<()> {
        assert!(src_offset + bytes <= src.size && dst_offset + bytes <= dst.size, "copy out of bounds");
        if bytes == 0 {
            return Ok(());
        }
        let cb = self.begin_command()?;
        let region = vk::BufferCopy::default().src_offset(src.offset + src_offset).dst_offset(dst.offset + dst_offset).size(bytes);
        unsafe { self.ctx.device.cmd_copy_buffer(cb, src.buffer, dst.buffer, &[region]) };
        self.end_command()
    }

    /// Queues filling `bytes` bytes at `dst + offset` with the 32-bit pattern
    /// `value` (offset and size must be multiples of 4).
    pub fn fill_u32(&mut self, dst: &Allocation, offset: u64, bytes: u64, value: u32) -> Result<()> {
        assert!(offset.is_multiple_of(4) && bytes.is_multiple_of(4) && offset + bytes <= dst.size, "bad fill range");
        if bytes == 0 {
            return Ok(());
        }
        let cb = self.begin_command()?;
        unsafe { self.ctx.device.cmd_fill_buffer(cb, dst.buffer, dst.offset + offset, bytes, value) };
        self.end_command()
    }

    /// Submits everything recorded so far. Returns the timeline value that
    /// will be signaled when it completes.
    pub fn flush(&mut self) -> Result<u64> {
        let Some(cb) = self.recording.take() else { return Ok(self.submitted) };
        let d = &self.ctx.device;
        unsafe {
            // Make the batch's writes visible to the host once it completes.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ | vk::AccessFlags::HOST_WRITE);
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            d.end_command_buffer(cb)?;

            let value = self.submitted + 1;
            let values = [value];
            let semaphores = [self.timeline];
            let buffers = [cb];
            let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
            let submit = vk::SubmitInfo::default().command_buffers(&buffers).signal_semaphores(&semaphores).push_next(&mut timeline);
            d.queue_submit(self.ctx.queue, &[submit], vk::Fence::null())?;
            self.submitted = value;
            self.in_flight.push_back((value, cb));
        }
        self.commands = 0;
        self.batch_groups = 0;
        Ok(self.submitted)
    }

    /// Timeline value of the newest finished batch.
    pub fn completed(&self) -> Result<u64> {
        Ok(unsafe { self.ctx.semaphore_counter_value(self.timeline)? })
    }

    /// Blocks until the batch with timeline value `value` has finished: by
    /// polling the timeline for up to [`spin_us`] microseconds, then in the
    /// driver's wait. (A driver's blocking wait wakes the thread late: on the
    /// H100 NVL, a tiny kernel followed by a synchronize took 43 us, most of
    /// it the wake-up; CUDA's synchronize spins likewise by default.)
    pub fn wait(&self, value: u64) -> Result<()> {
        let spin = spin_us();
        if spin > 0 {
            let start = std::time::Instant::now();
            while start.elapsed().as_micros() < spin as u128 {
                if self.completed()? >= value {
                    return Ok(());
                }
                std::hint::spin_loop();
            }
        }
        let semaphores = [self.timeline];
        let values = [value];
        unsafe { self.ctx.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values), u64::MAX)? };
        Ok(())
    }

    /// Submits pending work and blocks until all of it has finished. After
    /// this, the host may read and write any memory the stream used.
    pub fn synchronize(&mut self) -> Result<()> {
        let value = self.flush()?;
        self.wait(value)?;
        self.recycle()
    }

    /// Number of commands recorded but not yet submitted.
    pub fn pending(&self) -> u32 {
        self.commands
    }

    fn recycle(&mut self) -> Result<()> {
        let done = self.completed()?;
        while let Some(&(value, cb)) = self.in_flight.front() {
            if value > done {
                break;
            }
            unsafe { self.ctx.device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())? };
            self.spare.push(cb);
            self.in_flight.pop_front();
        }
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.synchronize();
        unsafe {
            let d = &self.ctx.device;
            d.destroy_command_pool(self.pool, None);
            d.destroy_semaphore(self.timeline, None);
        }
    }
}
