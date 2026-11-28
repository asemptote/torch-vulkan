//! Back-pressure: how much GPU work the stream lets the host queue ahead of
//! the GPU (docs/log/M59).
//!
//! Left alone, an eager loop that never reads a result records and submits
//! as fast as the host can, however far behind the GPU is: minutes of work
//! can queue up, and the next synchronize waits for all of it. The stream
//! therefore waits before a submission while the work already submitted and
//! unfinished is too much, by three measures:
//!
//! - **estimated GPU time** ([`Pacing::max_queued_ns`]): each kernel's time
//!   is estimated from earlier runs of the same pipeline over a similar grid,
//!   timed with timestamp queries ([`Costs`]);
//! - **submissions** (`Stream::max_in_flight`): bounds the command buffers,
//!   and the host and driver memory they hold, whatever the estimates say;
//! - on Intel GPUs under Linux, also **measured before queued**
//!   ([`Pacing::measure_first`]): a kernel whose time is not yet known is
//!   submitted at the end of its batch, and only once earlier such batches
//!   have finished, so that i915's 20 s watchdog (docs/log/M34) never sees
//!   more than one batch of unknown cost queued.
//!
//! Batches are also submitted once their estimated time reaches
//! [`Pacing::max_batch_ns`], so that the bound acts on pieces much smaller
//! than itself.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use ash::vk;

/// Estimated GPU time queued at most, by default (TORCH_VK_MAX_QUEUED_MS).
pub const DEFAULT_MAX_QUEUED_MS: u64 = 500;

/// Submissions in flight at most by default, except on Intel GPUs under
/// Linux (2: [`crate::stream::default_max_in_flight`]).
pub const DEFAULT_MAX_IN_FLIGHT: u32 = 16;

/// Kernels timed per batch at most (two timestamp queries each).
pub const TIMED_PER_BATCH: u32 = 16;

/// TORCH_VK_MAX_QUEUED_MS, else [`DEFAULT_MAX_QUEUED_MS`]: the estimated GPU
/// time (milliseconds) of the work submitted and unfinished, plus the batch
/// about to go, above which the stream waits before submitting (0: no bound,
/// and no kernels timed).
pub fn default_max_queued_ns() -> u64 {
    std::env::var("TORCH_VK_MAX_QUEUED_MS").ok().and_then(|v| v.trim().parse::<f64>().ok()).map_or(DEFAULT_MAX_QUEUED_MS * 1_000_000, |ms| {
        (ms.max(0.0) * 1e6) as u64
    })
}

/// TORCH_VK_MEASURE_FIRST (on/off), by default on for Intel GPUs under Linux
/// only ([`crate::stream::intel_linux`]): see [`Pacing::measure_first`].
pub fn default_measure_first(vendor_id: u32) -> bool {
    match std::env::var("TORCH_VK_MEASURE_FIRST") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => crate::stream::intel_linux(vendor_id),
    }
}

/// A kernel's cost class: its pipeline, and the power of two its workgroup
/// count rounds up to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CostKey {
    pipeline: u64,
    class: u8,
}

impl CostKey {
    pub fn new(pipeline: vk::Pipeline, groups: u64) -> Self {
        use ash::vk::Handle;
        Self { pipeline: pipeline.as_raw(), class: (64 - groups.saturating_sub(1).leading_zeros()) as u8 }
    }
}

/// What is known of one cost class.
#[derive(Debug, Default, Clone, Copy)]
struct Cost {
    /// Estimated nanoseconds per workgroup (valid once `measured`).
    ns_per_group: f64,
    measured: bool,
    /// Dispatches seen.
    seen: u64,
}

/// A multiply-shift hasher for [`CostKey`]s (looked up once per dispatch).
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(26) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    fn write_u8(&mut self, v: u8) {
        self.write_u64(v as u64);
    }
}

/// Kernel times learned from timestamp queries, per [`CostKey`].
///
/// An estimate is nanoseconds per workgroup, so that it scales within a
/// class. A measurement above the estimate replaces it at once; one below
/// moves it halfway down, so that a slow outlier (a cold cache, a clock
/// still rising, another process on the GPU) errs on the side of queueing
/// less. Each class is timed at its 1st, 2nd, 4th, ... 256th dispatch and
/// then every 256th, so a training loop's kernels are timed a few times
/// each and then rarely: a timed kernel waits for the commands before it to
/// finish (its first timestamp), so it does not overlap with them.
#[derive(Default)]
pub struct Costs {
    map: HashMap<CostKey, Cost, BuildHasherDefault<KeyHasher>>,
}

impl Costs {
    /// Counts one more dispatch of `key` over `groups` workgroups: its
    /// estimated time (ns) if known, and whether to time it.
    pub fn note(&mut self, key: CostKey, groups: u64) -> (Option<f64>, bool) {
        let c = self.map.entry(key).or_default();
        c.seen += 1;
        let time = (c.seen.is_power_of_two() && c.seen <= 256) || c.seen.is_multiple_of(256);
        (c.measured.then(|| c.ns_per_group * groups.max(1) as f64), time)
    }

    /// The estimated time (ns) of a dispatch of `key` over `groups`
    /// workgroups, if any of its class has been timed.
    pub fn estimate(&self, key: CostKey, groups: u64) -> Option<f64> {
        self.map.get(&key).filter(|c| c.measured).map(|c| c.ns_per_group * groups.max(1) as f64)
    }

    /// Learns from a dispatch of `key` over `groups` workgroups that took `ns`.
    pub fn learn(&mut self, key: CostKey, groups: u64, ns: f64) {
        let rate = ns / groups.max(1) as f64;
        let c = self.map.entry(key).or_default();
        c.ns_per_group = if !c.measured || rate > c.ns_per_group { rate } else { 0.5 * (c.ns_per_group + rate) };
        c.measured = true;
    }

    /// Cost classes seen.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Counts of what the stream's back-pressure did, since the stream was created.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct QueueStats {
    /// Batches submitted.
    pub submissions: u64,
    /// Waits before a submission (any bound).
    pub waits: u64,
    /// Time spent in those waits (ns).
    pub wait_ns: u64,
    /// Kernels timed with timestamp queries.
    pub timed: u64,
    /// The most estimated GPU time queued after a submission (ns).
    pub max_queued_ns: f64,
    /// The most submissions in flight after a submission.
    pub max_in_flight: u64,
}

/// The stream's time bounds and kernel-cost estimates (see the module's
/// documentation).
pub struct Pacing {
    /// Estimated GPU time (ns) that the work submitted and unfinished, plus
    /// the batch about to go, may reach before the stream waits (0: no bound).
    /// The oldest submission counts in full, though it may be half done. A
    /// batch goes at once when nothing is in flight, however long it is.
    pub max_queued_ns: u64,
    /// A batch is submitted once its estimated time reaches this (0: no
    /// limit): a quarter of `max_queued_ns` by default.
    pub max_batch_ns: u64,
    /// Before submitting a batch with a kernel whose time is not yet known,
    /// wait until no earlier such batch is unfinished, and submit each such
    /// kernel at the end of its batch. Needs timestamps.
    pub measure_first: bool,
    /// Timestamp queries are available (and kernels are timed).
    pub timing: bool,
    /// Nanoseconds per timestamp tick, and the mask of valid bits.
    pub period_ns: f64,
    pub mask: u64,
    pub costs: Costs,
    pub stats: QueueStats,
}

impl Pacing {
    pub fn new(vendor_id: u32, timestamp_bits: u32, timestamp_period: f32) -> Self {
        let max_queued_ns = default_max_queued_ns();
        let timing = timestamp_bits > 0 && max_queued_ns > 0;
        Self {
            max_queued_ns,
            max_batch_ns: max_queued_ns / 4,
            measure_first: timing && default_measure_first(vendor_id),
            timing,
            period_ns: timestamp_period as f64,
            mask: if timestamp_bits >= 64 { u64::MAX } else { (1u64 << timestamp_bits) - 1 },
            costs: Costs::default(),
            stats: QueueStats::default(),
        }
    }

    /// Nanoseconds between two timestamps (ticks, `end` after `start`).
    pub fn elapsed_ns(&self, start: u64, end: u64) -> f64 {
        (end.wrapping_sub(start) & self.mask) as f64 * self.period_ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_and_estimates() {
        use ash::vk::Handle;
        let p = vk::Pipeline::from_raw(0x1234);
        assert_eq!(CostKey::new(p, 1).class, 0);
        assert_eq!(CostKey::new(p, 2).class, 1);
        assert_eq!(CostKey::new(p, 3).class, 2);
        assert_eq!(CostKey::new(p, 4).class, 2);
        assert_eq!(CostKey::new(p, 1 << 20).class, 20);
        let mut c = Costs::default();
        let k = CostKey::new(p, 1000);
        let timed: Vec<bool> = (1..=1024).map(|_| c.note(k, 1000).1).collect();
        let at: Vec<usize> = timed.iter().enumerate().filter(|(_, &t)| t).map(|(i, _)| i + 1).collect();
        assert_eq!(at, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]);
        assert_eq!(c.estimate(k, 1000), None);
        c.learn(k, 1000, 2000.0);
        assert_eq!(c.estimate(k, 1000), Some(2000.0));
        assert_eq!(c.estimate(k, 600), Some(1200.0));
        c.learn(k, 1000, 4000.0); // up at once
        assert_eq!(c.estimate(k, 1000), Some(4000.0));
        c.learn(k, 1000, 1000.0); // down halfway
        assert_eq!(c.estimate(k, 1000), Some(2500.0));
    }

    #[test]
    fn wrapping_timestamps() {
        let p = Pacing { mask: (1 << 36) - 1, period_ns: 2.0, ..Pacing::new(0, 36, 2.0) };
        assert_eq!(p.elapsed_ns(10, 15), 10.0);
        assert_eq!(p.elapsed_ns((1 << 36) - 2, 3), 10.0);
    }
}
