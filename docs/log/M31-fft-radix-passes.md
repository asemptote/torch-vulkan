# Large power-of-two FFTs: cached twiddles, overlapped loads (M31)

Large complex FFTs (n = 2^20 and up) run as fft_radix.slang's radix-r
Stockham passes, r <= 128: three passes at 2^20 and four at 2^24. Each pass
reads and writes the whole array. On the H100 NVL, a 2^24 complex64 FFT took
11.8 times as long as copying the array, about three copies per pass, where
the bandwidth allows one. Issue #35 asks for at most 2x CUDA.

## Twiddles from two small tables

A pass multiplies each value by w^e, w = exp(-2 pi i / n). The kernel read
w^e from a table of every w^t, t < n/2, at places scattered over the table.
That table is 64 MiB for 2^24 complex64, so the reads missed the caches.

The table is now two levels, computed on the host in float64:
w^e = lo[e mod 2^L] hi[e >> L], with L = ceil(log2(n) / 2). That is about
2 sqrt(n) entries, which stay in cache. The extra complex product costs
about an ulp. The float32 errors stay within 1.5x of the CPU's, and float64
stays at about 1e-15.

## Loads in flight while a block is transformed

Every workgroup loaded its block of g groups (8 KiB), transformed it, and
stored it. With a workgroup per block, nothing overlapped the loads.

Now at most 4096 workgroups loop over the blocks. Each one loads the next
block's values and twiddles into registers before it transforms the
current block, and applies the twiddles once the values arrive. The first
pass (ns = 1) has no twiddles at all.

The cap was chosen by measurement:
- on the H100, 4096 and 8192 were best;
- no cap was 15% slower at 2^24;
- the L40S is fastest without a cap, but 4096 costs it at most 10% at 2^24
  complex64, where it stays at 1.2x CUDA.

## Tried and dropped

- **Larger radices** (r up to 256, 512 or 1024, so fewer passes). These
  give shorter contiguous runs per row, and 2^24 got slower: complex128
  went from 1.06 to 1.46 ms on the H100. Only 2^22 complex64 gained.
- **A 32 KiB workgroup** (four times as many groups per block, so longer
  runs). It was slower everywhere, for example 0.96 against 0.79 ms at 2^24
  complex64 on the H100. The shared memory cost more occupancy than the
  longer runs saved.

## Results

Pipelined ms per call, n = 2^24:

| GPU | dtype | before | now | CUDA (vs_cuda.py) |
|---|---|---|---|---|
| H100 NVL | complex64 | 1.05 | 0.69 | 0.32 |
| H100 NVL | complex128 | 1.70 | 1.12 | 0.62 |
| L40S | complex64 | 2.16 | 1.80 | 1.49 |
| L40S | complex128 | 4.84 | 4.17 | 2.94 |

At 2^22 the gain is 15-25%. At 2^20 a call is three passes of about a copy
each, plus the submission. Two passes at most need a different kernel: a
whole 1024-point DFT per pass, with the values held in registers (M32).
