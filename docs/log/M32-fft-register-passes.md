# Power-of-two FFTs in two or three passes, values in registers (M32)

After M31, a complex FFT from 2^15 up still took one pass per 7 bits of n
(fft_radix.slang, r <= 128): three passes at 2^20 and four at 2^24. Each
pass is a full read and write of the array. At 2^20 the array fits in L2,
and each pass costs about as much as a separate copy dispatch. The 0.2.0
release waits on every FFT case being within 2x of CUDA on the H100 NVL
and the L40S (#35).

## The pass (shaders/fft_reg.slang)

Rows from 2^14 up now take radix-R passes, with R from 128 to 1024: two
passes up to 2^20, three up to 2^30.

- **Indexing.** The Stockham indexing is fft_radix.slang's, and so are the
  two-level twiddle table and the outer twiddles.
- **Work per workgroup.** A workgroup takes g consecutive columns of the
  pass (4 for complex64 at R = 1024, 16 at R = 256). Its loads and stores
  are runs of g values: 32 bytes and up, a whole sector on NVIDIA. This is
  VkFFT's coalescedMemory, and it's why a plain 2 x 1024 split with one
  column per workgroup was slower than three passes.
- **First stage, from global memory.** Each thread holds E values of its
  column in registers (16, or 8 in the e8 variant) and takes their E-point
  DFT. The DFTs of 8 and 16 points are 2 x 4 and 4 x 4 butterflies.
- **Later stages, through shared memory.** Radix-E Stockham stages follow
  while they fit, then one of R / m. The last stage stores to global memory
  directly.
- **Shared memory.** g R = WG E values, 32 KiB (16 KiB for complex64 with
  E = 8). Rows are XOR-swizzled within blocks of E, against the first
  stage's writes E rows apart. The host uses the pass where the device has
  32 KiB (a new capability bit), and fft_radix.slang's passes elsewhere.

Each stage's radix is a compile-time generic parameter (`stage<RHO>`), with
a fixed sequence of stages per R. A first version picked the radix at run
time and indexed the register array dynamically. That spilled it, and it
ran 25-40% slower.

Stage twiddles depend only on the thread, not the data, so complex64 loads
them once, before the first block. complex128 loads them at each stage's
start, before its barrier; holding them all took too many registers.

## Measured and dropped

- **E = 16 for everything.** It is best for complex64 up to 2^20. From 2^21,
  and for complex128 on the H100, E = 8 (256 threads, 8 values each) is
  faster.
- **VkFFT's complex64 layout** (512 threads x 8 values). It was within a
  few percent of E = 16 and needs 512-thread workgroups.
- **Padding instead of the swizzle.** It would push complex128 past 32 KiB.
  The swizzle itself made no measurable difference on NVIDIA.

## Where 2^20 stands

On the H100, a synced 2^20 complex64 call is 7.4 us of host time, about
17 us of submission and synchronisation (as for a tiny element-wise op),
and two passes of about a copy dispatch each. The kernels are at the cost
of the stack's dispatches. The remaining distance to 2x CUDA there is fixed
per-call cost, which #45 (dependency tracking between dispatches) is about.

## Results

Median of 25 synced calls (each `fn(); synchronize()`), against PyTorch
2.14.1 + CUDA 13.0 (cuFFT) on the same GPU. Ratios over 2 in bold.

| op | n | H100 NVL vk / CUDA (ms) | ratio | L40S vk / CUDA (ms) | ratio |
|---|---|---|---|---|---|
| fft complex64 | 2^16 | 0.040 / 0.017 | **2.29** | 0.030 / 0.014 | **2.15** |
| fft complex64 | 2^18 | 0.040 / 0.020 | 1.95 | 0.030 / 0.017 | 1.75 |
| fft complex64 | 2^20 | 0.061 / 0.028 | **2.17** | 0.048 / 0.026 | 1.85 |
| fft complex64 | 2^22 | 0.143 / 0.079 | 1.82 | 0.119 / 0.098 | 1.21 |
| fft complex64 | 2^24 | 0.481 / 0.304 | 1.58 | 1.363 / 1.476 | 0.92 |
| fft complex128 | 2^16 | 0.037 / 0.020 | 1.88 | 0.057 / 0.045 | 1.26 |
| fft complex128 | 2^18 | 0.040 / 0.026 | 1.57 | 0.065 / 0.089 | 0.73 |
| fft complex128 | 2^20 | 0.099 / 0.047 | **2.10** | 0.207 / 0.202 | 1.03 |
| fft complex128 | 2^22 | 0.270 / 0.162 | 1.66 | 0.814 / 0.966 | 0.84 |
| fft complex128 | 2^24 | 0.998 / 0.548 | 1.82 | 3.177 / 2.929 | 1.08 |
| rfft float32 | 2^16 | 0.045 / 0.021 | **2.14** | 0.033 / 0.016 | **2.04** |
| rfft float32 | 2^18 | 0.045 / 0.022 | **2.04** | 0.032 / 0.018 | 1.84 |
| rfft float32 | 2^20 | 0.052 / 0.028 | 1.85 | 0.039 / 0.025 | 1.59 |
| rfft float32 | 2^22 | 0.147 / 0.053 | **2.76** | 0.099 / 0.050 | 1.99 |
| rfft float32 | 2^24 | 0.460 / 0.195 | **2.36** | 0.889 / 0.776 | 1.15 |
| rfft float64 | 2^16 | 0.042 / 0.022 | 1.89 | 0.060 / 0.050 | 1.20 |
| rfft float64 | 2^18 | 0.043 / 0.026 | 1.65 | 0.066 / 0.070 | 0.94 |
| rfft float64 | 2^20 | 0.073 / 0.040 | 1.82 | 0.127 / 0.160 | 0.79 |
| rfft float64 | 2^22 | 0.226 / 0.103 | **2.20** | 0.500 / 0.484 | 1.03 |
| rfft float64 | 2^24 | 0.769 / 0.359 | **2.14** | 2.172 / 1.882 | 1.15 |

The cases over 2x:
- **2^16 to 2^20.** A call has about 40 us of fixed cost: host, submission,
  synchronisation and two dispatches. cuFFT's whole call takes 17 to 28 us.
  The kernels are at the cost of the stack's dispatches (above). Dispatch
  latency is #45, for 0.3.0. The user accepted the H100's 2^20 at about
  2.1 to 2.2x for 0.2.0.
- **H100 rfft at 2^22 and 2^24.** This is throughput, not latency.
  - The half-length c2c is 2^21 or 2^23: three passes for an odd number of
    bits.
  - fft.slang's R2C pass is separate. For float32 at 2^22 it takes about
    0.05 ms where the bandwidth allows 0.02.
  - Fusing R2C into this pass's last store is the next step (#35).

