//! Per-command overhead: synchronizing after every kernel vs letting the
//! stream batch them. This is the overhead that dominates many small tensor ops.
//!
//!     cargo run --release --example dispatch_bench

use std::time::Instant;

use vkcore::{Result, Runtime};

#[repr(C)]
#[derive(Clone, Copy)]
struct Fill {
    y: u64,
    value: f64,
    n: u32,
    stride: u32,
}

fn main() -> Result<()> {
    let rt = Runtime::new()?;
    let n = 1024u32;
    let y = rt.alloc(n as u64 * 8)?;
    let (groups, stride) = rt.grid_1d(n as u64, 64);
    let ops = 5_000;
    let run = |sync_each: bool| -> Result<f64> {
        let t = Instant::now();
        for i in 0..ops {
            let p = Fill { y: y.address, value: i as f64, n, stride };
            rt.dispatch("fill_f64", &[], &p, [groups, 1, 1])?;
            if sync_each {
                rt.synchronize()?;
            }
        }
        rt.synchronize()?;
        Ok(t.elapsed().as_secs_f64() / ops as f64 * 1e6)
    };
    run(false)?; // warm-up: pipeline compile, command buffers
    println!("{} ({ops} fills of {n} float64 each)", rt.ctx.device_name);
    println!("  sync after every op: {:7.1} us/op", run(true)?);
    let batch = rt.stream.lock().unwrap().max_batch;
    println!("  batched ({batch}/submit): {:7.1} us/op", run(false)?);
    assert_eq!(unsafe { y.as_slice_mut::<f64>() }[n as usize - 1], (ops - 1) as f64);
    rt.free(y);
    Ok(())
}
