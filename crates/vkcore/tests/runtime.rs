//! GPU tests for the M1 runtime. They need a Vulkan device with float64.
//! Run under validation with `TORCH_VK_VALIDATE=1 cargo test`; every test
//! then also fails on any validation warning or error.

use vkcore::{memory::BLOCK_SIZE, validation_issues, Result, Runtime};

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
    pool.trim();
    assert_eq!(pool.stats(), Default::default());
    drop(pool);
    assert_clean();
    Ok(())
}

#[test]
fn kernel_cache_reuses_pipelines() -> Result<()> {
    let rt = Runtime::new()?;
    let mut cache = rt.kernels.lock().unwrap();
    let a = cache.get("axpy_f64", &[1])?;
    let b = cache.get("axpy_f64", &[1])?;
    let c = cache.get("axpy_f64", &[0])?;
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert_ne!(a.pipeline, c.pipeline);
    assert!(cache.get("no_such_kernel", &[]).is_err());
    drop(cache);
    assert_clean();
    Ok(())
}

/// Every kernel this device can run (those whose capabilities it has) builds
/// a pipeline, with default specialization constants: the first check on a
/// new GPU or driver. `cargo test -p vkcore --release every_kernel_builds -- --nocapture`
/// lists the failures.
#[test]
fn every_kernel_builds() -> Result<()> {
    let rt = Runtime::new()?;
    let mut cache = rt.kernels.lock().unwrap();
    let mut failed = Vec::new();
    let mut refused = 0;
    for name in vkcore::kernels::KernelCache::names() {
        match cache.get(name, &[]) {
            Ok(_) => {}
            Err(e) if e.to_string().contains("which this device does not support") => refused += 1,
            Err(e) => failed.push(format!("{name}: {e}")),
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
