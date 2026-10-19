//! End-to-end check of the backend's core assumptions on this GPU:
//! float64 shaders, raw device-address pointers, and zero-copy unified memory.
//!
//!     cargo run --release --example smoke

use std::time::Instant;

use vkcore::{version_string, Context, Result};

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    x: u64,
    y: u64,
    n: u32,
    stride: u32,
}

const SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/square_plus_third.spv"));
const WORKGROUP: u32 = 64;

fn main() -> Result<()> {
    let ctx = Context::new()?;
    let c = ctx.caps;
    println!("device: {} (Vulkan {})", ctx.device_name, version_string(c.api_version));
    println!(
        "  f64 {} | i64 {} | i16 {} | f16 {} | bda {} | scalar layout {} | memory model {}",
        c.shader_float64, c.shader_int64, c.shader_int16, c.shader_float16, c.buffer_device_address, c.scalar_block_layout, c.vulkan_memory_model
    );
    println!(
        "  subgroup {} | shared mem {} KiB | max groups.x {} | max alloc {:.2} GiB | max buffer {:.2} GiB",
        c.subgroup_size,
        c.max_shared_memory / 1024,
        c.max_workgroup_count_x,
        c.max_allocation_size as f64 / (1u64 << 30) as f64,
        c.max_buffer_size as f64 / (1u64 << 30) as f64
    );
    for (i, h) in ctx.memory_heaps().iter().enumerate() {
        println!("  heap {i}: {:.2} GiB {:?}", h.size as f64 / (1u64 << 30) as f64, h.flags);
    }
    println!("  unified (device-local + host-visible) memory type: {:?}", c.unified_memory_type);
    if !c.shader_float64 {
        return Err("this GPU has no float64 shader support".into());
    }

    let n: u32 = 1 << 24; // 16M doubles = 128 MiB per buffer
    let bytes = n as u64 * 8;
    let mut x = ctx.create_buffer(bytes)?;
    let mut y = ctx.create_buffer(bytes)?;

    // Fill the input straight in GPU-visible memory: no staging copy.
    for (i, v) in x.as_slice_mut::<f64>().iter_mut().enumerate() {
        *v = (i as f64 * 0.001).sin() * 3.0;
    }

    let kernel = ctx.create_kernel(SPIRV, std::mem::size_of::<Params>() as u32)?;
    let groups = (n.div_ceil(WORKGROUP)).min(c.max_workgroup_count_x).min(65_535);
    let params = Params { x: x.address, y: y.address, n, stride: groups * WORKGROUP };

    ctx.run(&kernel, &params, groups)?; // warm-up
    let reps = 10;
    let t = Instant::now();
    for _ in 0..reps {
        ctx.run(&kernel, &params, groups)?;
    }
    let gpu = t.elapsed().as_secs_f64() / reps as f64;

    let xs: Vec<f64> = x.as_slice_mut::<f64>().to_vec();
    let t = Instant::now();
    let mut reference = vec![0.0f64; n as usize];
    for _ in 0..reps {
        for (r, v) in reference.iter_mut().zip(&xs) {
            *r = v * v + 1.0 / 3.0;
        }
        std::hint::black_box(&reference);
    }
    let cpu = t.elapsed().as_secs_f64() / reps as f64;

    let max_err = y.as_slice_mut::<f64>().iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("\nkernel y = x^2 + 1/3 on {n} float64 values (pointers via buffer device address):");
    println!("  max |gpu - cpu| = {max_err:e}  ({})", if max_err == 0.0 { "bit-exact" } else if max_err < 1e-12 { "float64 OK" } else { "MISMATCH" });
    println!("  gpu {:.2} ms (incl. submit + wait) | cpu single-thread {:.2} ms", gpu * 1e3, cpu * 1e3);
    println!("  effective GPU bandwidth: {:.1} GB/s", 2.0 * bytes as f64 / gpu / 1e9);
    Ok(())
}
