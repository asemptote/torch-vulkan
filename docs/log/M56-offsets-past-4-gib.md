# Offsets past 4 GiB and 2^31 elements (M56)

TorchBench's pytorch_unet failed its accuracy check on vk (#142): its
`Conv2d(128, 64, 3, padding=1)` on a `[2, 128, 640, 959]` input came out
with a relative error of 0.85, without an error. The convolution's im2col
columns are `[2, 1152, 613760]`, 1.41e9 float32 elements, 5.27 GiB. The
threshold was the columns' bytes passing 2^32, not their elements passing
2^31: the same convolution with 96 input channels (3.95 GiB) was right.

## The cause

The kernels take each tensor as a buffer device address in their push
constants and index it with 32-bit integers: slang compiles `p[i]` and
`p + i` to an `OpPtrAccessChain` whose element index is that integer. On
NVIDIA's driver (615.71, H100 NVL) the byte offset from the address wraps
at 4 GiB. Main on the H100, a float32 tensor of 1.2e9 elements:

| | result | right |
|---|---|---|
| `x.fill_(1)`, then the sum of `x[2**30:]` (through its own address) | 0 | 126,258,176 |
| `x.sum()` | 1.2e9 (the first 4 GiB read twice) | 1.2e9 (by chance) |
| `torch.arange(1.2e9)[-4:]` | 0, 0, 0, 0 | the last four |
| `bmm` whose last matrix starts past 4 GiB | rel. error 0.67 | |
| #142's convolution, forward / weight gradient | 0.85 / 0.70 | |

Element 2^30 of a float32 tensor was read and written at its element 0, so
every kernel was affected, not only the columns' copy: #135's pieces of a
product did not change it. Mesa's drivers compute these addresses in 64
bits (lavapipe was right).

Two rewrites of the SPIR-V were tried on the H100 (`fill_` and `sum` past 4
GiB, float32 and float64):

- each index widened to 64 bits before the access chain (`OpSConvert` /
  `OpUConvert`): still wrapped;
- the address computed as integers, `OpConvertPtrToU`, a 64-bit
  multiply-add with the pointer type's `ArrayStride`, `OpConvertUToPtr`:
  right.

So NVIDIA's driver keeps a 32-bit byte offset per access chain, whatever
the index's width; integer address arithmetic is out of its reach.

## 64-bit addresses where a dispatch needs them

`crates/vkcore/src/wide_address.rs` rewrites every access chain on a device
address that way (`integer_addresses`), adding the `Int64` capability, the
64-bit types and the stride constants it needs. Indices keep their meaning:
a signed index is sign-extended, an unsigned one zero-extended, a 64-bit one
used as it is.

Only dispatches that need it run the rewritten kernel. `Runtime::wide_addresses`
(lib.rs) looks each 8-byte word of the push constants up in the pool's live
allocations, as the barriers do (M37: kernels get every address from their
push constants): where one has more than 4 GiB of its allocation after it,
the kernel cache builds and runs the kernel's 64-bit form (keyed apart from
the usual one). The lookup is skipped while the pool's blocks span 4 GiB or
less. A tensor of 4 GiB or less, or an address in its last 4 GiB, takes the
kernel as compiled.

On the H100 the rewritten kernels ran as fast as the usual ones in every
case measured (`TORCH_VK_WIDE_ADDRESSES=1`, best of 3 x 30 calls, ms):

| | add 64M | mm 4096 | mm 4096 float16 | conv [8, 64, 128²] | sum(1) 4096² | softmax 4096² | transpose copy 4096² |
|---|--:|--:|--:|--:|--:|--:|--:|
| as compiled | 0.276 | 8.64 | 0.884 | 1.71 | 0.048 | 0.140 | 0.135 |
| 64-bit addresses | 0.279 | 8.50 | 0.888 | 1.75 | 0.048 | 0.146 | 0.136 |

They still run only where needed: not every driver was measured, and a
kernel as compiled is the code its author tuned.

A device without `shaderInt64` (Mali GPUs) cannot run the 64-bit form: such
a dispatch is refused with an error naming the kernel. Whether Mali's
driver wraps at 4 GiB was not measured.

`TORCH_VK_WIDE_ADDRESSES=1` (a developer switch) runs every kernel in its
64-bit form. `cargo test`'s `every_kernel_builds` builds both forms of every
kernel, and a unit test checks that every access chain of every kernel is
understood and rewritten; all 412 rewritten modules pass `spirv-val`.

## Offsets past 2^31 elements: an audit

With bytes past 4 GiB addressed, the 32-bit integers themselves remain the
limit: a kernel's element offsets must stay below 2^31 (signed) and its
counts below what its loops can step through. Every launch site in csrc was
checked for offsets or products the host did not bound. `vk.h` now names
the limit (`kKernelOffsets`, 2^31) and `offsets_fit(t)`. Found and changed
(each either runs in pieces or refuses: the CPU fallback where the op has
one, else an error naming the limit):

| launch | was unchecked | now |
|---|---|---|
| `tril`/`triu` (triangle.slang) | any size: `b * sb + i * sr + j * sc` in int | CPU fallback past 2^31 |
| gather, `index_select`, `take`, `x[idx]`, scatter, `index_put_`, `index_copy`, `index_fill_` (move.slang through index offsets) | the indexed tensor's span: its views have stride 0 on the indexed dim | error when its storage holds 2^31 elements or more from the view's start |
| accumulation (`index_add`, `scatter_add`/`reduce`, `index_put_` with accumulate: segreduce.slang) | the destination's span | the same error; `accumulate_rows` falls back to the general path |
| `embedding_dense_backward`'s rows (segreduce.slang) | `num_rows * width` checked against 2^62 | 2^31, else the CPU |
| `sum_rows_by_sorted_group` (sparse products) | `groups * width` | 2^31, else `index_add` |
| `nonzero` | `numel` below 2^32; coordinates `count * ndim` | both below 2^31 |
| scans: `cummax`/`cummin`, `logcumsumexp`, `mode` (rowscan.slang) | nothing | error past 2^31 |
| `searchsorted`, `bucketize` | nothing (`numel` cast to uint) | CPU fallback past 2^31 |
| dense linear algebra (factor.slang, getrf, geqrf, eig, stedc, ...: offsets in uint) | only the batch (65,535 matrices) | at most 2^30 elements a launch, its workspaces up to twice that: `cholesky_ex` and `cholesky_solve` in chunks of matrices, right-hand sides checked; else the CPU |
| FFT (fft.slang, fft_radix, fft_reg: `row * n`, `batch * n` in uint) | nothing | inputs and outputs below 2^28 complex values (Bluestein's rows are up to 4x longer, some passes index real parts), else the CPU |
| reductions with a stride-0 input (reduce.slang's chunk ends, argreduce's `int(i)`) | `n_out * n_red` below 2^32 | 2^31, else the CPU |
| `gemv4` (`k0 * WBITS` in uint) | the bits along a row | below 2^32, else another route |
| `_weight_int8pack_mm` (qmm8.slang) | A's span | M <= 8: checked (else the CPU); tiled: rows in pieces within A's offsets |
| pooling (pool.slang: `plane * iv`) | only the elements written | the elements read too; error past 2^31 |
| grid sampling | a plane past 2^31 (expanded input); the backward's corner tables | errors |
| `conv_direct` (conv4s.slang) | the padded weight | its size, else im2col |
| random fills and sampling, upsampling, the strided element-wise launches | counts up to 2^32 (a grid-stride loop's 32-bit counter can wrap within a step of 2^32: a hang) | 2^31 |

The convolution's columns: a batch whose im2col columns reach 2^31
elements runs in pieces of the batch, forward and weight gradient (#142's
model at batch 4: 2.8e9 elements of columns, 3 samples then 1). One sample
whose columns reach 2^31 is refused by the columns' copy.

The element-wise and copy launches (`launch.cpp`, `strided.h`) were already
bounded (each operand's span below 2^31, the outer dims on the host), as
were the matrix products (`max_offset` below 2^31 for every operand, M52's
pieces), softmax, the norms and losses (`numel` below 2^31), `sort`
(`total` below 2^31) and `cumsum`.

## Tests

`tests/python/test_addressing.py`:

- `test_tensor_past_4_gib`: one float32 tensor of 4 GiB + 4 MiB, each kernel
  given its start: fill, multiply, sums, max, a `bmm` whose last matrix is
  past 4 GiB, `arange`.
- `test_convolution_with_columns_past_4_gib`: #142's convolution at batch 2
  and 4 (columns in one piece, then in two), forward and weight gradient,
  with data whose sums are exact.
- These two skip on lavapipe (the host's memory) and where the device
  cannot allocate the tensors; on the H100 they take 16 s together.
- `test_wide_kernels_need_int64`: with `TORCH_VK_DISABLE=int64` and every
  dispatch wide, a kernel is refused by name.
- `test_wide_kernels`: `test_core.py`, `test_indexing.py` and
  `test_sorting.py` with `TORCH_VK_WIDE_ADDRESSES=1` (25 s on lavapipe).

Without the change, `test_tensor_past_4_gib` and the convolution's test
fail on the H100 at their first check. #142's repro gives a relative error
of 6.4e-7 against the CPU (0.85 before).
