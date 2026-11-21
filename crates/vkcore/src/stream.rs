//! An in-order command stream, like a CUDA stream.
//!
//! Commands are recorded into a command buffer and submitted in batches; a
//! timeline semaphore counts finished batches. Each command declares the GPU
//! memory it reads and writes ([`Access`]), and the stream puts a memory
//! barrier before a command only when it conflicts with a command since the
//! last barrier (read after write, write after read, write after write), as
//! llama.cpp's Vulkan backend does. A command without a declaration is
//! ordered against everything, before and after it. See docs/log/M37.

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::kernels::{Kernel, PUSH_CONSTANT_BYTES};
use crate::memory::Allocation;
use crate::{Context, Result};

/// Commands per batch before the stream submits on its own.
pub const DEFAULT_MAX_BATCH: u32 = 256;

/// Workgroups per batch before the stream submits on its own, on Intel GPUs
/// (0: no limit, elsewhere; TORCH_VK_MAX_BATCH_GROUPS overrides). A big
/// dispatch is then submitted alone, while small ones still batch up to
/// DEFAULT_MAX_BATCH, so that one submission is a few bitonic passes of a
/// huge sort rather than the whole sort.
///
/// Under Linux, i915 cancels GPU work still unfinished 20 s
/// (`i915.request_timeout_ms`) after it was submitted, and Mesa's ANV reports
/// VK_SUCCESS all the same (mesa#16166): the commands cut off simply leave
/// their outputs unwritten. The 20 s count from submission, not from when the
/// work starts, so they cover everything queued ahead of it as well. On the
/// HD 530 a 2^27-entry float64 sort (~26 s of GPU work, 172 dispatches) in
/// one submission came back unsorted. In one submission per dispatch it was
/// right, but only because i915 also blocks a submission while the context's
/// ring holds a few dozen unfinished ones (see [`default_max_in_flight`]),
/// which kept ~10 s of its ~0.15 s passes queued at a time; with passes of
/// ~0.4 s that is over 20 s again. Small batches therefore go with a bound on
/// the submissions in flight.
pub const INTEL_MAX_BATCH_GROUPS: u64 = 1 << 13;

/// An Intel GPU under Linux, where the kernel driver (i915) silently cancels
/// work still unfinished 20 s after it was submitted (see
/// [`INTEL_MAX_BATCH_GROUPS`]). The stream then bounds the submissions in
/// flight and checks each one's fence status as it retires.
pub fn intel_linux(vendor_id: u32) -> bool {
    cfg!(target_os = "linux") && vendor_id == 0x8086
}

/// Submissions the stream lets the GPU have queued or running at once
/// (0: no limit): TORCH_VK_MAX_IN_FLIGHT, by default 2 on Intel GPUs under
/// Linux ([`intel_linux`]) and 0 elsewhere. Before submitting a batch, the
/// stream waits until fewer than this many earlier ones are unfinished. With
/// 2, a batch waits behind at most the one before it, so each finishes within
/// two batches' time of its submission, while the GPU still has the next
/// batch queued whenever one finishes. (i915 itself blocks a submission only
/// once the context's ring is nearly full, a few dozen submissions: on the
/// HD 530 that let over 20 s of work queue up.)
pub fn default_max_in_flight(vendor_id: u32) -> u32 {
    std::env::var("TORCH_VK_MAX_IN_FLIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if intel_linux(vendor_id) { 2 } else { 0 })
}

/// Microseconds [`Stream::wait`] polls before blocking: TORCH_VK_SPIN_US, by
/// default 500 on discrete GPUs and 0 (block at once) elsewhere. Integrated
/// GPUs share the CPU's power budget (and a phone's battery), lavapipe its
/// cores; on the HD 530 polling saved nothing measurable and cost ~5% more CPU
/// time, while on NVIDIA it took ~19 us off every synchronize.
pub fn spin_us(discrete: bool) -> u64 {
    static SPIN: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    SPIN.get_or_init(|| std::env::var("TORCH_VK_SPIN_US").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(if discrete { 500 } else { 0 })
}

/// Where the stream puts barriers (TORCH_VK_BARRIERS).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierMode {
    /// (default) Only before a command that conflicts with one since the
    /// last barrier, or after an undeclared one.
    Tracked,
    /// `check`: as Tracked, and the runtime checks each declared dispatch's
    /// writes against its declaration (see `Runtime::dispatch_declared`).
    Check,
    /// `every`: before every command, from the previous command's kind to
    /// the next's (compute -> compute between dispatches): 0.2.0's barriers.
    Every,
    /// `full`: a full compute + transfer barrier before every command.
    Full,
}

impl BarrierMode {
    pub fn tracked(self) -> bool {
        matches!(self, BarrierMode::Tracked | BarrierMode::Check)
    }
}

/// TORCH_VK_BARRIERS: `full`, `every`, `check`, or anything else (unset) for
/// [`BarrierMode::Tracked`].
pub fn barrier_mode() -> BarrierMode {
    static MODE: std::sync::OnceLock<BarrierMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("TORCH_VK_BARRIERS").as_deref() {
        Ok("full") => BarrierMode::Full,
        Ok("every") => BarrierMode::Every,
        Ok("check") => BarrierMode::Check,
        _ => BarrierMode::Tracked,
    })
}

/// TORCH_VK_BARRIERS=full (a debugging switch: see [`BarrierMode::Full`]).
pub fn full_barriers() -> bool {
    barrier_mode() == BarrierMode::Full
}

/// GPU memory `[lo, hi)` (device addresses) that a command reads, or reads
/// and writes (`write`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Access {
    pub lo: u64,
    pub hi: u64,
    pub write: bool,
}

impl Access {
    /// Whether the two accesses touch common bytes and one of them writes.
    pub fn conflicts(&self, other: &Access) -> bool {
        (self.write || other.write) && self.lo < other.hi && other.lo < self.hi
    }
}

/// Accesses kept since the last barrier before the stream puts in a barrier
/// anyway, so that checking a command against them stays cheap.
pub const MAX_UNSYNCED: usize = 128;

/// Barrier counts since the stream was created.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BarrierStats {
    /// Commands recorded (dispatches, copies and fills).
    pub commands: u64,
    /// Barriers between commands (not counting each batch's final one).
    pub barriers: u64,
    /// Commands recorded without a declaration of the memory they touch.
    pub undeclared: u64,
}

/// How long a submission must have been in flight (submitted to seen
/// finished) before its fence's status is read, on [`intel_linux`]
/// platforms: i915 cancels a submission only once it is 20 s old
/// (`i915.request_timeout_ms`; settable, but rarely below that). Reading the
/// status costs about 12 us (exporting a sync_file), resetting the fence
/// instead about 1.
const CHECK_AFTER: std::time::Duration = std::time::Duration::from_secs(1);

/// A submitted batch not yet retired.
struct Submission {
    /// The timeline value it signals.
    value: u64,
    cb: vk::CommandBuffer,
    /// The exportable fence it signals (null unless the context has `fence_fd`).
    fence: vk::Fence,
    at: std::time::Instant,
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
    /// Submitted batches not yet retired.
    in_flight: VecDeque<Submission>,
    spare: Vec<vk::CommandBuffer>,
    spare_fences: Vec<vk::Fence>,
    /// Set once a retired submission's fence carried an error (the kernel
    /// cancelled it): from then on every call fails, as on a lost device.
    cancelled: Option<i32>,
    pub max_batch: u32,
    /// Submissions in flight at most (0: no limit; see [`default_max_in_flight`]).
    pub max_in_flight: u32,
    /// Workgroups per batch before it is submitted (0: no limit; see
    /// INTEL_MAX_BATCH_GROUPS; TORCH_VK_MAX_BATCH_GROUPS overrides).
    pub max_batch_groups: u64,
    batch_groups: u64,
    /// Kernels dispatched since the stream was created.
    pub dispatches: u64,
    /// Whether the last command recorded (in this batch or an earlier one) was
    /// a transfer (copy, fill) rather than a dispatch; None before the first.
    last_transfer: Option<bool>,
    /// Set with [`Self::set_mode`].
    mode: BarrierMode,
    /// The accesses of the commands recorded since the last barrier (in this
    /// batch or earlier ones: a barrier orders everything before it in
    /// submission order, other submissions too).
    unsynced: Vec<Access>,
    /// A command was recorded since the last barrier.
    pending: bool,
    /// An undeclared command was recorded since the last barrier.
    opaque: bool,
    /// A transfer command was ever recorded: barriers then include the
    /// transfer stage, so that every write is made visible to both kinds.
    transfers: bool,
    pub stats: BarrierStats,
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
            let max_in_flight = default_max_in_flight(ctx.vendor_id);
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
                spare_fences: Vec::new(),
                cancelled: None,
                max_batch: DEFAULT_MAX_BATCH,
                max_in_flight,
                max_batch_groups,
                batch_groups: 0,
                dispatches: 0,
                last_transfer: None,
                mode: barrier_mode(),
                unsynced: Vec::new(),
                pending: false,
                opaque: false,
                transfers: false,
                stats: BarrierStats::default(),
            })
        }
    }

    pub fn mode(&self) -> BarrierMode {
        self.mode
    }

    /// Switches barrier modes (for tests and measurements; the default comes
    /// from TORCH_VK_BARRIERS). The next command is ordered after all earlier
    /// ones.
    pub fn set_mode(&mut self, mode: BarrierMode) {
        self.mode = mode;
        self.unsynced.clear();
        self.pending = self.last_transfer.is_some();
        self.opaque = self.pending;
    }

    /// The command buffer being recorded, starting one if needed, with a
    /// barrier against earlier commands (in this batch or an earlier one)
    /// where the new command needs one. `transfer`: the new command is a copy
    /// or fill rather than a dispatch; `acc`: the memory it touches, or None
    /// (undeclared) for all of it.
    fn begin_command(&mut self, transfer: bool, acc: Option<&[Access]>) -> Result<vk::CommandBuffer> {
        self.stats.commands += 1;
        if acc.is_none() {
            self.stats.undeclared += 1;
        }
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
        if self.mode.tracked() {
            self.track(cb, transfer, acc);
            return Ok(cb);
        }
        // TORCH_VK_BARRIERS=every and full: a barrier before every command.
        // Also before a batch's first command: submission order alone orders
        // nothing, and a barrier's first scope reaches back into earlier
        // submissions on the queue. (Without it, lavapipe could start a batch's
        // first dispatch while the previous batch's last one still ran.)
        // Still a full memory dependency on the previous command, but with only
        // the stages and accesses of the two commands' kinds (compute ->
        // compute for consecutive dispatches): earlier commands are ordered
        // through the chain of barriers, and writes made available by an
        // earlier barrier become visible to later accesses with this one's
        // visibility operation. (The full compute + transfer barrier cost about
        // 5.5 us of GPU time per dispatch on the H100.)
        if self.mode == BarrierMode::Full {
            // (TORCH_VK_BARRIERS=full: the full compute + transfer barrier
            // before every command, as before cccade8, to rule out the
            // narrowed barriers when a race is suspected)
            let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                );
            unsafe { d.cmd_pipeline_barrier(cb, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
            self.stats.barriers += 1;
        } else if let Some(prev) = self.last_transfer {
            let (src_stage, src_access) = if prev {
                (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_WRITE)
            } else {
                (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE)
            };
            let (dst_stage, dst_access) = if transfer {
                (vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)
            } else {
                (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            };
            let barrier = vk::MemoryBarrier::default().src_access_mask(src_access).dst_access_mask(dst_access);
            unsafe { d.cmd_pipeline_barrier(cb, src_stage, dst_stage, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
            self.stats.barriers += 1;
        }
        self.last_transfer = Some(transfer);
        Ok(cb)
    }

    /// Tracked barriers: puts a barrier into `cb` before the new command if
    /// it needs one, then records the command's accesses.
    ///
    /// A barrier is needed when a command was recorded since the last one
    /// and: the new command is undeclared, or one since the last barrier was;
    /// the new command reads or writes memory written since the last barrier,
    /// or writes memory read since then; it is the first transfer command;
    /// or the accesses kept reach [`MAX_UNSYNCED`].
    ///
    /// Each barrier is a global memory barrier from every earlier command to
    /// every later one (the first scope of a pipeline barrier is all commands
    /// before it in submission order, earlier submissions included), with the
    /// compute stage, and the transfer stage once transfers were used, on
    /// both sides: once it is recorded, every earlier write is available and
    /// visible to every later command, so the kept accesses are dropped.
    /// Between barriers, commands may run in any order or concurrently, which
    /// is right for any two of them that touch disjoint memory or only read
    /// common memory.
    fn track(&mut self, cb: vk::CommandBuffer, transfer: bool, acc: Option<&[Access]>) {
        // (the first transfer: earlier barriers made writes visible to the
        // compute stage only, so one is needed even with nothing pending)
        let new_kind = transfer && !self.transfers && self.last_transfer.is_some();
        let conflict = match acc {
            None => true,
            Some(a) => {
                self.unsynced.len() + a.len() > MAX_UNSYNCED || a.iter().any(|x| self.unsynced.iter().any(|u| x.conflicts(u)))
            }
        };
        let need = new_kind || self.pending && (self.opaque || conflict);
        self.transfers |= transfer;
        if need {
            let (mut stages, mut src, mut dst) =
                (vk::PipelineStageFlags::COMPUTE_SHADER, vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            if self.transfers {
                stages |= vk::PipelineStageFlags::TRANSFER;
                src |= vk::AccessFlags::TRANSFER_WRITE;
                dst |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
            }
            let barrier = vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst);
            unsafe { self.ctx.device.cmd_pipeline_barrier(cb, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
            self.stats.barriers += 1;
            self.unsynced.clear();
            self.opaque = false;
        }
        match acc {
            None => self.opaque = true,
            Some(a) => {
                for x in a {
                    if x.lo >= x.hi {
                        continue;
                    }
                    // (the same range again, e.g. an op's output in each of
                    // its passes, is kept once)
                    if let Some(u) = self.unsynced.iter_mut().find(|u| u.lo == x.lo && u.hi == x.hi) {
                        u.write |= x.write;
                    } else {
                        self.unsynced.push(*x);
                    }
                }
            }
        }
        self.pending = true;
        self.last_transfer = Some(transfer);
    }

    fn end_command(&mut self) -> Result<()> {
        self.commands += 1;
        if self.commands >= self.max_batch || (self.max_batch_groups > 0 && self.batch_groups >= self.max_batch_groups) {
            self.flush()?;
        }
        Ok(())
    }

    /// Queues `kernel` over `groups` workgroups with `push` (at most
    /// [`PUSH_CONSTANT_BYTES`] bytes) as its push constants, ordered against
    /// every command before and after it.
    pub fn dispatch(&mut self, kernel: &Kernel, push: &[u8], groups: [u32; 3]) -> Result<()> {
        self.dispatch_declared(kernel, push, groups, None)
    }

    /// [`Self::dispatch`] for a kernel that touches only the memory `acc`
    /// declares (None: any memory). A kernel that reads or writes memory
    /// outside its declaration races with the commands around it.
    pub fn dispatch_declared(&mut self, kernel: &Kernel, push: &[u8], groups: [u32; 3], acc: Option<&[Access]>) -> Result<()> {
        assert!(push.len() <= PUSH_CONSTANT_BYTES as usize, "push constants too large for {}", kernel.name);
        self.dispatches += 1;
        // Always set the whole range the layout declares, so no byte a kernel
        // might read is left undefined.
        let mut padded = [0u8; PUSH_CONSTANT_BYTES as usize];
        padded[..push.len()].copy_from_slice(push);

        let cb = self.begin_command(false, acc)?;
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
        let acc = [
            Access { lo: src.address + src_offset, hi: src.address + src_offset + bytes, write: false },
            Access { lo: dst.address + dst_offset, hi: dst.address + dst_offset + bytes, write: true },
        ];
        let cb = self.begin_command(true, Some(&acc))?;
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
        let acc = [Access { lo: dst.address + offset, hi: dst.address + offset + bytes, write: true }];
        let cb = self.begin_command(true, Some(&acc))?;
        unsafe { self.ctx.device.cmd_fill_buffer(cb, dst.buffer, dst.offset + offset, bytes, value) };
        self.end_command()
    }

    /// Submits everything recorded so far. Returns the timeline value that
    /// will be signaled when it completes.
    pub fn flush(&mut self) -> Result<u64> {
        if self.recording.is_none() {
            return Ok(self.submitted);
        }
        self.limit_in_flight()?;
        let fence = self.fence()?;
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
            let at = std::time::Instant::now();
            d.queue_submit(self.ctx.queue, &[submit], fence)?;
            self.submitted = value;
            self.in_flight.push_back(Submission { value, cb, fence, at });
        }
        self.commands = 0;
        self.batch_groups = 0;
        Ok(self.submitted)
    }

    /// With [`Self::max_in_flight`] set, waits until fewer than that many
    /// submissions are unfinished, so that the next one can go.
    fn limit_in_flight(&mut self) -> Result<()> {
        let max = self.max_in_flight as usize;
        if max == 0 || self.in_flight.len() < max {
            return Ok(());
        }
        self.recycle()?;
        if self.in_flight.len() >= max {
            let value = self.in_flight[self.in_flight.len() - max].value;
            self.wait(value)?;
            self.recycle()?;
        }
        Ok(())
    }

    /// A fence for the next submission: an unsignalled exportable one where
    /// the context has `fence_fd`, else null.
    fn fence(&mut self) -> Result<vk::Fence> {
        if self.ctx.fence_fd.is_none() {
            return Ok(vk::Fence::null());
        }
        if let Some(f) = self.spare_fences.pop() {
            return Ok(f);
        }
        let mut export = vk::ExportFenceCreateInfo::default().handle_types(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
        Ok(unsafe { self.ctx.device.create_fence(&vk::FenceCreateInfo::default().push_next(&mut export), None)? })
    }

    /// The error, if any, that the kernel recorded on a finished submission's
    /// fence (ANV does not report it: mesa#16166), read if the submission
    /// was in flight for [`CHECK_AFTER`] or more; the fence is reset for reuse.
    fn fence_error(&self, s: &Submission) -> Result<Option<i32>> {
        if s.at.elapsed() >= CHECK_AFTER {
            if let Some(error) = exported_fence_error(&self.ctx, s.fence) {
                return error;
            }
        }
        // (the timeline semaphore may be seen signalled before the fence)
        unsafe {
            self.ctx.device.wait_for_fences(&[s.fence], true, u64::MAX)?;
            self.ctx.device.reset_fences(&[s.fence])?;
        }
        Ok(None)
    }

    /// Once a submission was cancelled, every call fails with this.
    fn check_cancelled(&self) -> Result<()> {
        match self.cancelled {
            None => Ok(()),
            Some(status) => Err(cancelled_message(status).into()),
        }
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
        self.check_cancelled()?;
        let spin = spin_us(self.ctx.discrete);
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

    /// Number of submissions not yet finished.
    pub fn unfinished(&self) -> Result<u64> {
        Ok(self.submitted.saturating_sub(self.completed()?))
    }

    /// Number of commands recorded but not yet submitted.
    pub fn pending(&self) -> u32 {
        self.commands
    }

    /// Retires the finished submissions, reusing their command buffers and
    /// fences, and fails if the kernel cancelled any of them.
    fn recycle(&mut self) -> Result<()> {
        self.check_cancelled()?;
        let done = self.completed()?;
        while self.in_flight.front().is_some_and(|s| s.value <= done) {
            let s = self.in_flight.pop_front().unwrap();
            unsafe { self.ctx.device.reset_command_buffer(s.cb, vk::CommandBufferResetFlags::empty())? };
            self.spare.push(s.cb);
            if s.fence != vk::Fence::null() {
                match self.fence_error(&s) {
                    Ok(error) => {
                        self.spare_fences.push(s.fence);
                        self.cancelled = self.cancelled.or(error);
                    }
                    Err(e) => {
                        unsafe { self.ctx.device.destroy_fence(s.fence, None) };
                        return Err(e);
                    }
                }
            }
        }
        self.check_cancelled()
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.synchronize();
        unsafe {
            // (synchronize fails at once after a cancellation: still wait for
            // everything submitted before destroying its command buffers)
            let semaphores = [self.timeline];
            let values = [self.submitted];
            let _ = self.ctx.wait_semaphores(&vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values), u64::MAX);
            let d = &self.ctx.device;
            for s in self.in_flight.drain(..) {
                self.spare_fences.push(s.fence);
            }
            for f in self.spare_fences.drain(..).filter(|&f| f != vk::Fence::null()) {
                d.destroy_fence(f, None);
            }
            d.destroy_command_pool(self.pool, None);
            d.destroy_semaphore(self.timeline, None);
        }
    }
}

/// The error, if any, on a finished submission's fence, from its status as
/// an exported sync_file (exporting also resets the fence); None where the
/// context cannot export fences.
#[cfg(target_os = "linux")]
fn exported_fence_error(ctx: &Context, fence: vk::Fence) -> Option<Result<Option<i32>>> {
    ctx.fence_fd.as_ref().map(|fence_fd| sync_file_error(fence_fd, fence))
}

#[cfg(not(target_os = "linux"))]
fn exported_fence_error(_ctx: &Context, _fence: vk::Fence) -> Option<Result<Option<i32>>> {
    None
}

#[cfg(target_os = "linux")]
fn sync_file_error(fence_fd: &ash::khr::external_fence_fd::Device, fence: vk::Fence) -> Result<Option<i32>> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    let info = vk::FenceGetFdInfoKHR::default().fence(fence).handle_type(vk::ExternalFenceHandleTypeFlags::SYNC_FD);
    let fd = unsafe { fence_fd.get_fence_fd(&info)? };
    if fd < 0 {
        return Ok(None); // (allowed for a signalled fence: no status to read)
    }
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    let mut status = sync_file_status(fd.as_raw_fd())?;
    if status == 0 {
        // Still active (the timeline semaphore can signal first): wait.
        let mut p = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        unsafe { libc::poll(&mut p, 1, -1) };
        status = sync_file_status(fd.as_raw_fd())?;
    }
    Ok((status < 0).then_some(status))
}

/// The status `SYNC_IOC_FILE_INFO` reports for a sync_file: 1 signalled, 0
/// still active, negative: signalled with that error (-errno).
#[cfg(target_os = "linux")]
fn sync_file_status(fd: std::os::fd::RawFd) -> Result<i32> {
    #[repr(C)]
    struct SyncFileInfo {
        name: [u8; 32],
        status: i32,
        flags: u32,
        num_fences: u32,
        pad: u32,
        sync_fence_info: u64,
    }
    // _IOWR('>', 4, struct sync_file_info), from linux/sync_file.h
    const SYNC_IOC_FILE_INFO: u32 = (3 << 30) | ((std::mem::size_of::<SyncFileInfo>() as u32) << 16) | ((b'>' as u32) << 8) | 4;
    let mut info = SyncFileInfo { name: [0; 32], status: 0, flags: 0, num_fences: 0, pad: 0, sync_fence_info: 0 };
    if unsafe { libc::ioctl(fd, SYNC_IOC_FILE_INFO as _, &mut info) } < 0 {
        return Err(format!("SYNC_IOC_FILE_INFO: {}", std::io::Error::last_os_error()).into());
    }
    Ok(info.status)
}

/// The error once the kernel cancelled a submission (fence status `status`).
fn cancelled_message(status: i32) -> String {
    let why = if status == -4 { "-4, EINTR: i915's request watchdog".to_string() } else { status.to_string() };
    format!(
        "GPU work was cancelled by the kernel driver while the Vulkan driver reported success (fence status {why}), \
         so results computed on the device since then are invalid; restart the process. On Linux, i915 cancels work \
         still unfinished 20 s (i915.request_timeout_ms) after it was submitted: keep less work queued at once \
         (TORCH_VK_MAX_IN_FLIGHT, TORCH_VK_MAX_BATCH_GROUPS) or smaller single operations, or raise that timeout"
    )
}
