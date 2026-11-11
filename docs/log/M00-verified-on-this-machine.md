# Verified on this machine (M0)

`cargo run --release --example smoke` on Intel UHD 620, driver 101.2134:

- Vulkan 1.3.215; float64, int64, int16, float16, buffer device address,
  scalar block layout, Vulkan memory model: all supported
- Subgroup size 32 (8–32), 32 KiB shared memory, max workgroups.x 65536,
  max single allocation 7.94 GiB, max buffer 4 GiB
- One 7.94 GiB heap with a device-local + host-visible memory type (unified memory)
- Not supported: int64 atomics, float32/float64 atomic add, cooperative matrix
  (tensor cores) -- reductions must use subgroup ops / shared memory instead
- Kernel `y = x² + 1/3` over 16M float64 values via device-address pointers:
  max error 1.8e-15 vs CPU (FMA contraction), 15 ms GPU vs 22 ms single-thread CPU,
  ~18 GB/s effective bandwidth
