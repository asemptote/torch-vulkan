//! GPU tests for the M1 runtime. They need a Vulkan device with float64.
//! Run under validation with `TORCH_VK_VALIDATE=1 cargo test`; every test
//! then also fails on any validation warning or error.

use vkcore::{as_bytes, memory::BLOCK_SIZE, validation_issues, Placement, Result, Runtime, STAGING_BYTES};

#[repr(C)]
#[derive(Clone, Copy)]
struct Fill {
    y: u64,
    value: f64,
    n: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Axpy {
    x: u64,
    y: u64,
    a: f64,
    n: u32,
    stride: u32,
}

fn fill(rt: &Runtime, y: u64, value: f64, n: u32) -> Result<()> {
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    rt.dispatch("simple_fill_f64", &[], &Fill { y, value, n, stride }, [groups, 1, 1])
}

fn axpy(rt: &Runtime, accumulate: bool, a: f64, x: u64, y: u64, n: u32) -> Result<()> {
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    rt.dispatch("axpy_f64", &[accumulate as u32], &Axpy { x, y, a, n, stride }, [groups, 1, 1])
}

fn assert_clean() {
    assert_eq!(validation_issues(), 0, "validation layer reported problems (see stderr)");
}

#[test]
fn dependent_kernels_in_one_batch() -> Result<()> {
    let rt = Runtime::new()?;
    let n = (1 << 20) + 3; // not a multiple of the workgroup size
    let x = rt.alloc(n as u64 * 8)?;
    let y = rt.alloc(n as u64 * 8)?;
    fill(&rt, x.address, 2.0, n)?;
    fill(&rt, y.address, 1.0, n)?;
    for _ in 0..100 {
        axpy(&rt, true, 3.0, x.address, y.address, n)?; // each reads the previous result
    }
    rt.synchronize()?;
    let ys = unsafe { y.as_slice_mut::<f64>() };
    assert!(ys[..n as usize].iter().all(|&v| v == 601.0), "y = 1 + 100 * 3 * 2 everywhere");
    rt.free(x);
    rt.free(y);
    assert_clean();
    Ok(())
}

#[test]
fn many_batches_without_sync() -> Result<()> {
    let rt = Runtime::new()?;
    rt.stream.lock().unwrap().max_batch = 16; // force many submissions
    let n = 4096;
    let y = rt.alloc(n as u64 * 8)?;
    for i in 0..1000 {
        fill(&rt, y.address, i as f64, n)?;
    }
    rt.synchronize()?;
    assert!(unsafe { y.as_slice_mut::<f64>() }.iter().all(|&v| v == 999.0));
    rt.free(y);
    assert_clean();
    Ok(())
}

#[test]
fn bounded_submissions_in_flight() -> Result<()> {
    let rt = Runtime::new()?;
    {
        let mut s = rt.stream.lock().unwrap();
        s.max_batch = 1; // one submission per command
        s.max_in_flight = 2;
    }
    let n = 1 << 16;
    let y = rt.alloc(n as u64 * 8)?;
    for i in 0..200 {
        fill(&rt, y.address, i as f64, n)?;
        let unfinished = rt.stream.lock().unwrap().unfinished()?;
        assert!(unfinished <= 2, "{unfinished} submissions in flight");
    }
    rt.synchronize()?;
    assert!(unsafe { y.as_slice_mut::<f64>() }.iter().all(|&v| v == 199.0));
    rt.free(y);
    assert_clean();
    Ok(())
}

/// The stream's estimate of a kernel's time, against the wall time of one
/// run of it alone, and the bound on the estimated GPU time queued
/// (docs/log/M59): after every dispatch, the work in flight is within the
/// bound, or a single submission.
#[test]
fn bounded_queued_time() -> Result<()> {
    let rt = Runtime::new()?;
    if !rt.stream.lock().unwrap().pacing.timing {
        eprintln!("no timestamps on this queue: skipped");
        return Ok(());
    }
    let n = 1 << 22;
    let y = rt.alloc(n as u64 * 8)?;
    // Timed at its 1st, 2nd and 4th runs: the estimate is then known.
    let mut wall = f64::MAX;
    for i in 0..4 {
        let t = std::time::Instant::now();
        fill(&rt, y.address, i as f64, n)?;
        rt.synchronize()?;
        wall = wall.min(t.elapsed().as_nanos() as f64);
    }
    {
        let mut s = rt.stream.lock().unwrap();
        assert_eq!(s.queued_ns(), 0.0, "nothing queued after a synchronize");
        assert!(s.queue_stats().timed >= 3, "{:?}", s.queue_stats());
        s.max_batch = 1; // one submission per kernel
        s.max_in_flight = 0;
        s.pacing.max_batch_ns = 0;
    }
    fill(&rt, y.address, 0.0, n)?;
    let one = rt.stream.lock().unwrap().queued_ns();
    assert!(one > 0.0 && one <= wall * 1.5 + 1e6, "estimate {one} ns, alone {wall} ns");
    rt.synchronize()?;
    // (about 4 kernels' estimated time)
    let limit = 4.0 * one;
    rt.stream.lock().unwrap().pacing.max_queued_ns = limit as u64;
    let limit = limit.floor();
    for i in 0..200 {
        fill(&rt, y.address, i as f64, n)?;
        let s = rt.stream.lock().unwrap();
        let (in_flight, _) = s.in_flight();
        assert!(in_flight <= 1 || s.queued_ns() <= limit * 1.0001, "{in_flight} submissions, {} ns queued > {limit}", s.queued_ns());
    }
    rt.synchronize()?;
    let stats = rt.stream.lock().unwrap().queue_stats();
    assert!(stats.max_queued_ns <= limit * 2.0, "{stats:?}");
    assert!(unsafe { y.as_slice_mut::<f64>() }.iter().all(|&v| v == 199.0));
    rt.free(y);
    assert_clean();
    Ok(())
}

/// Measure-first (Intel GPUs under Linux, docs/log/M59): a kernel whose time
/// is unknown goes at the end of its batch, and at most one submission with
/// such a kernel is in flight.
#[test]
fn measure_first() -> Result<()> {
    let rt = Runtime::new()?;
    {
        let mut s = rt.stream.lock().unwrap();
        if !s.pacing.timing {
            eprintln!("no timestamps on this queue: skipped");
            return Ok(());
        }
        s.pacing.measure_first = true;
        s.max_in_flight = 0;
    }
    let y = rt.alloc(1 << 20)?;
    // Grids of new sizes, so new cost classes, as well as repeats.
    for i in 0..60u32 {
        let elems = 64 << (i % 12);
        fill(&rt, y.address, i as f64, elems)?;
        let s = rt.stream.lock().unwrap();
        let (_, unknown) = s.in_flight();
        assert!(unknown <= 1, "{unknown} submissions of unknown cost in flight");
        if i < 12 {
            assert_eq!(s.pending(), 0, "a new class is submitted at once");
        }
    }
    rt.synchronize()?;
    assert!(unsafe { y.as_slice_mut::<f64>() }[..64].iter().all(|&v| v == 59.0));
    rt.free(y);
    assert_clean();
    Ok(())
}

#[test]
fn specialization_constants() -> Result<()> {
    let rt = Runtime::new()?;
    let n = 1000;
    let x = rt.alloc(n as u64 * 8)?;
    let y = rt.alloc(n as u64 * 8)?;
    fill(&rt, x.address, 5.0, n)?;
    fill(&rt, y.address, 100.0, n)?;
    axpy(&rt, false, 2.0, x.address, y.address, n)?; // y = 2 x, ignoring old y
    rt.synchronize()?;
    assert!(unsafe { y.as_slice_mut::<f64>() }[..n as usize].iter().all(|&v| v == 10.0));
    rt.free(x);
    rt.free(y);
    assert_clean();
    Ok(())
}

#[test]
fn copy_and_fill_with_offsets() -> Result<()> {
    let rt = Runtime::new()?;
    let a = rt.alloc(1024 * 8)?;
    let b = rt.alloc(1024 * 8)?;
    for (i, v) in unsafe { a.as_slice_mut::<f64>() }.iter_mut().enumerate() {
        *v = i as f64;
    }
    {
        let mut s = rt.stream.lock().unwrap();
        s.fill_u32(&b, 0, 1024 * 8, 0)?;
        s.copy(&a, 100 * 8, &b, 10 * 8, 50 * 8)?; // b[10..60] = a[100..150]
        s.synchronize()?;
    }
    let bs = unsafe { b.as_slice_mut::<f64>() };
    for (i, &v) in bs.iter().enumerate() {
        let expected = if (10..60).contains(&i) { (i + 90) as f64 } else { 0.0 };
        assert_eq!(v, expected, "b[{i}]");
    }
    rt.free(a);
    rt.free(b);
    assert_clean();
    Ok(())
}

#[test]
fn stream_ordered_reuse() -> Result<()> {
    let rt = Runtime::new()?;
    let n = 1 << 16;
    let x = rt.alloc(n as u64 * 8)?;
    let tmp = rt.alloc(n as u64 * 8)?;
    fill(&rt, tmp.address, 7.0, n)?;
    axpy(&rt, false, 1.0, tmp.address, x.address, n)?; // x = tmp (queued, not finished)
    rt.free(tmp); // freed while still in use by queued work
    let reused = rt.alloc(n as u64 * 8)?;
    assert_eq!(reused.address, tmp.address, "pool hands the memory out again immediately");
    fill(&rt, reused.address, -1.0, n)?; // ordered after the read of tmp
    rt.synchronize()?;
    assert!(unsafe { x.as_slice_mut::<f64>() }.iter().all(|&v| v == 7.0));
    assert!(unsafe { reused.as_slice_mut::<f64>() }.iter().all(|&v| v == -1.0));
    rt.free(x);
    rt.free(reused);
    assert_clean();
    Ok(())
}

#[test]
fn pool_blocks_large_allocations_and_trim() -> Result<()> {
    let rt = Runtime::new()?;
    let small: Vec<_> = (0..10).map(|_| rt.alloc(1000)).collect::<Result<_>>()?;
    let stats = rt.pool.lock().unwrap().stats();
    assert_eq!(stats.blocks, 1, "small allocations share one block");
    assert_eq!(stats.allocated, 10 * 1024, "sizes round up to 256 B");

    let big = rt.alloc(BLOCK_SIZE + 1)?; // gets its own, larger block
    assert!(big.size > BLOCK_SIZE);
    assert_eq!(rt.pool.lock().unwrap().stats().blocks, 2);
    unsafe { big.as_slice_mut::<u8>() }.fill(1);

    rt.free(big);
    for a in small {
        rt.free(a);
    }
    let mut pool = rt.pool.lock().unwrap();
    assert_eq!(pool.stats().allocated, 0);
    assert_eq!(pool.stats().blocks, 2, "empty blocks stay cached");
    assert!(pool.stats().peak > BLOCK_SIZE + 10 * 1024, "the peak counts every allocation live at once");
    pool.trim();
    pool.reset_peak();
    assert_eq!(pool.stats(), Default::default());
    drop(pool);
    assert_clean();
    Ok(())
}

#[test]
fn unmapped_memory_through_staging() -> Result<()> {
    // Unmapped device memory (tensors past what the driver maps): kernels use
    // it as any other, the host reaches it through copies to and from host
    // staging memory, in pieces of STAGING_BYTES; best fit across blocks.
    let rt = Runtime::new()?;
    let n = (STAGING_BYTES / 8 + 1000) as u32; // (two staging pieces)
    let x = rt.alloc_in(n as u64 * 8, Placement::Unmapped)?;
    assert!(!x.mapped && x.host_ptr() as u64 >= vkcore::memory::UNMAPPED_BASE);
    let y = rt.alloc(n as u64 * 8)?;
    fill(&rt, y.address, 0.0, n)?;
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    rt.write(x.address, &bytes)?;
    axpy(&rt, true, 2.0, x.address, y.address, n)?; // y = 2 x
    rt.copy(x.address + 8, y.address, 16)?; // x[1..3] = y[0..2]
    let mut back = vec![0u8; bytes.len()];
    rt.read(x.address, &mut back)?;
    let got: Vec<f64> = back.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
    assert_eq!(&got[..4], &[0.0, 0.0, 2.0, 3.0]);
    assert!(got[4..].iter().enumerate().all(|(i, &v)| v == (i + 4) as f64));
    let mut ys = vec![0u8; 8];
    rt.read(y.address + (n as u64 - 1) * 8, &mut ys)?; // (mapped: straight from the mapping)
    assert_eq!(f64::from_le_bytes(ys.try_into().unwrap()), 2.0 * (n - 1) as f64);
    assert!(rt.pool.lock().unwrap().stats().unmapped >= n as u64 * 8);
    rt.free(x);
    rt.free(y);
    rt.pool.lock().unwrap().trim();
    assert_clean();
    Ok(())
}

#[test]
fn small_allocations_leave_an_empty_large_block_alone() -> Result<()> {
    // Best fit across blocks: a small request goes to the tightest range,
    // not into a large block left empty (which would then never be released).
    let rt = Runtime::new()?;
    let small = rt.alloc(1000)?;
    let big = rt.alloc(4 * BLOCK_SIZE)?;
    rt.free(big);
    let more: Vec<_> = (0..10).map(|_| rt.alloc(1000)).collect::<Result<_>>()?;
    assert!(more.iter().all(|a| a.block == small.block), "small allocations went into the large block");
    for a in more {
        rt.free(a);
    }
    rt.free(small);
    rt.pool.lock().unwrap().trim();
    assert_clean();
    Ok(())
}

#[test]
fn kernel_cache_reuses_pipelines() -> Result<()> {
    let rt = Runtime::new()?;
    let mut cache = rt.kernels.lock().unwrap();
    let a = cache.get("axpy_f64", &[1], false)?;
    let b = cache.get("axpy_f64", &[1], false)?;
    let c = cache.get("axpy_f64", &[0], false)?;
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_ne!(a.pipeline, c.pipeline);
    assert!(cache.get("no_such_kernel", &[], false).is_err());
    drop(cache);
    assert_clean();
    Ok(())
}

/// Every kernel this device can run (those whose capabilities it has) builds
/// a pipeline, with default specialization constants, and again with 64-bit
/// addresses (docs/log/M56): the first check on a new GPU or driver.
/// `cargo test -p vkcore --release every_kernel_builds -- --nocapture` lists
/// the failures.
#[test]
fn every_kernel_builds() -> Result<()> {
    let rt = Runtime::new()?;
    let mut cache = rt.kernels.lock().unwrap();
    let mut failed = Vec::new();
    let mut refused = 0;
    for name in vkcore::kernels::KernelCache::names() {
        for wide in [false, true] {
            match cache.get(name, &[], wide) {
                Ok(_) => {}
                Err(e) if e.to_string().contains("which this device does not support") => refused += 1,
                Err(e) => failed.push(format!("{name} (wide: {wide}): {e}")),
            }
        }
    }
    drop(cache);
    eprintln!("{refused} kernels need features this device lacks");
    assert!(failed.is_empty(), "kernels that did not build:
{}", failed.join("
"));
    assert_clean();
    Ok(())
}

// Declared dispatches (docs/log/M37): the runtime finds the memory each one
// touches from its push constants, and the stream puts barriers only where
// two commands conflict.

fn fill_declared(rt: &Runtime, y: u64, value: f64, n: u32) -> Result<()> {
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    rt.dispatch_declared("simple_fill_f64", &[], as_bytes(&Fill { y, value, n, stride }), [groups, 1, 1], &[])
}

fn axpy_declared(rt: &Runtime, accumulate: bool, a: f64, x: u64, y: u64, n: u32) -> Result<()> {
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    rt.dispatch_declared("axpy_f64", &[accumulate as u32], as_bytes(&Axpy { x, y, a, n, stride }), [groups, 1, 1], &[])
}

use vkcore::Declared;

fn tracked(rt: &Runtime) -> bool {
    rt.stream.lock().unwrap().mode().tracked()
}

#[test]
fn barriers_only_between_conflicting_dispatches() -> Result<()> {
    let rt = Runtime::new()?;
    let n = 1 << 16;
    let xs = (0..8).map(|_| rt.alloc(n as u64 * 8)).collect::<Result<Vec<_>>>()?;
    let barriers = || rt.barrier_stats().barriers;
    let b0 = barriers();
    for (i, x) in xs.iter().enumerate() {
        fill_declared(&rt, x.address, i as f64, n)?; // independent
    }
    let b1 = barriers();
    for x in &xs[1..] {
        axpy_declared(&rt, true, 1.0, x.address, xs[0].address, n)?; // reads x (RAW on the first), writes x0 (WAW)
    }
    let b2 = barriers();
    // Readers of x0, then a write to x0 (WAR), through an address inside it.
    // (Undeclared beyond their addresses, the readers also count as writing
    // x0, so they are ordered among themselves too; a declaration of x as
    // read only lets them overlap.)
    for x in &xs[1..] {
        axpy_declared(&rt, false, 2.0, xs[0].address, x.address, n)?;
    }
    let b3 = barriers();
    fill_declared(&rt, xs[0].address + 8 * 100, -1.0, 1)?;
    let b4 = barriers();
    // Declared read only, readers of x7 need no barrier between them.
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    for x in &xs[1..4] {
        let ro = Declared { ptr: xs[7].address, lo: xs[7].address, hi: xs[7].address + n as u64 * 8, write: 0, pad: 0 };
        let p = Axpy { x: xs[7].address, y: x.address, a: 1.0, n, stride };
        rt.dispatch_declared("axpy_f64", &[1], as_bytes(&p), [groups, 1, 1], &[ro])?;
    }
    let b4r = barriers();
    // An undeclared dispatch is ordered against everything before and after.
    fill(&rt, xs[7].address, 5.0, n)?;
    fill_declared(&rt, xs[6].address, 6.0, n)?;
    let b5 = barriers();
    rt.synchronize()?;
    let x0 = unsafe { xs[0].as_slice_mut::<f64>() };
    assert_eq!(x0[0], 28.0);
    assert_eq!(x0[100], -1.0);
    for (i, x) in xs.iter().enumerate().skip(1).take(5) {
        let expected = if i <= 3 { 112.0 } else { 56.0 }; // (x1..x3 += x7, which is 56 too)
        assert!(unsafe { x.as_slice_mut::<f64>() }.iter().all(|&v| v == expected), "x{i}");
    }
    assert!(unsafe { xs[7].as_slice_mut::<f64>() }.iter().all(|&v| v == 5.0));
    assert!(unsafe { xs[6].as_slice_mut::<f64>() }.iter().all(|&v| v == 6.0));
    if tracked(&rt) {
        assert_eq!(b1 - b0, 0, "independent fills");
        assert_eq!(b2 - b1, 7, "each accumulation into x0 after the one before");
        assert_eq!(b3 - b2, 7, "each reader of x0 (counted as writing it) after the one before");
        assert_eq!(b4 - b3, 1, "the write after the reads");
        assert_eq!(b4r - b4, 0, "readers declared read only");
        assert_eq!(b5 - b4, 2, "around the undeclared dispatch");
    }
    for x in xs {
        rt.free(x);
    }
    assert_clean();
    Ok(())
}

#[test]
fn declared_stream_ordered_reuse() -> Result<()> {
    let rt = Runtime::new()?;
    let n = 1 << 16;
    let x = rt.alloc(n as u64 * 8)?;
    for _ in 0..50 {
        let tmp = rt.alloc(n as u64 * 8)?;
        fill_declared(&rt, tmp.address, 7.0, n)?;
        axpy_declared(&rt, false, 1.0, tmp.address, x.address, n)?; // x = tmp
        rt.free(tmp);
        let reused = rt.alloc(n as u64 * 8)?;
        assert_eq!(reused.address, tmp.address);
        fill_declared(&rt, reused.address, -1.0, n)?; // must wait for the read of tmp (WAR)
        rt.free(reused);
    }
    rt.synchronize()?;
    assert!(unsafe { x.as_slice_mut::<f64>() }.iter().all(|&v| v == 7.0));
    rt.free(x);
    assert_clean();
    Ok(())
}

#[test]
fn accesses_from_push_constants() -> Result<()> {
    use vkcore::declare::push_accesses;
    use vkcore::Access;
    let rt = Runtime::new()?;
    let a = rt.alloc(1000)?;
    let b = rt.alloc(4096)?;
    let whole = |x: &vkcore::Allocation| Access { lo: x.address, hi: x.address + x.size, write: true };
    {
        let pool = rt.pool.lock().unwrap();
        // Addresses anywhere inside an allocation stand for all of it; other words are ignored.
        let push = [a.address + 16, 12345, b.address, 0];
        assert_eq!(push_accesses(&pool, as_bytes(&push), &[]), Some(vec![whole(&a), whole(&b)]));
        // A declaration that holds its address narrows it; one that does not is ignored.
        let narrow = Declared { ptr: a.address + 16, lo: a.address + 16, hi: a.address + 48, write: 0, pad: 0 };
        let wrong = Declared { ptr: b.address, lo: b.address + 8, hi: b.address + 16, write: 0, pad: 0 };
        let acc = push_accesses(&pool, as_bytes(&push), &[narrow, wrong]);
        assert_eq!(acc, Some(vec![Access { lo: a.address + 16, hi: a.address + 48, write: false }, whole(&b)]));
        // A declaration reaching outside the allocation is ignored too.
        let outside = Declared { ptr: a.address, lo: a.address, hi: a.address + a.size + 8, write: 1, pad: 0 };
        assert_eq!(push_accesses(&pool, as_bytes(&[a.address]), &[outside]), Some(vec![whole(&a)]));
    }
    let stale = b.address;
    rt.free(b);
    // An address in the pool's memory outside every live allocation: undeclared.
    assert_eq!(push_accesses(&rt.pool.lock().unwrap(), as_bytes(&[stale]), &[]), None);
    rt.free(a);
    Ok(())
}
