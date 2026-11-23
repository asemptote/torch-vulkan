# Device memory past what the driver maps (M53)

BlenderbotForCausalLM's inference failed on the 48 GB L40S with "A device
memory allocation has failed" (#133), where CUDA peaks at 21 GB (float32)
and 10 GB (bfloat16). The cause was not one large tensor (no tensor of
the model comes near a per-allocation limit: its largest weights are about
100 MB) or fragmentation, but where torch-vulkan put every tensor: in
memory the host maps, of which NVIDIA's driver lets a process have much
less than the GPU's memory.

## Where tensors were

The pool (crates/vkcore/src/memory.rs) sub-allocates tensors from 64 MiB
blocks (larger tensors get a block each). Every block was host-visible and
persistently mapped: device-local where the device has such a memory type,
plain host memory otherwise. Each tensor's data pointer was its address in
the mapping, so `.cpu()`, `.item()`, uploads, the CPU fallback and storage
resizes read and wrote tensors with `memcpy`. On integrated GPUs (Intel,
Mali, the CPU's lavapipe), all memory is like that, and nothing is copied.

On discrete GPUs, device-local host-visible memory is what the host reaches
through the PCIe BAR, and drivers limit it:

| GPU (driver) | device-local heap | host-visible part | tensors held before |
|---|---|---|---|
| Tesla M4, 4 GB (dev host, no resizable BAR) | 3.75 GiB | a separate 246 MiB heap | none of 256 MiB: the first failed |
| H100 NVL, 94 GB (NVIDIA 615.71) | 93.6 GiB | the same heap, but allocations stop at 34.5 GiB | 138 x 256 MiB |
| L40S, 48 GB (NVIDIA 615.71) | (not measured) | not measured here (its queue was busy; the model benchmarks' memory test is queued, #133) | not BlenderbotForCausalLM's peak, about 24 GB on the H100 |

The H100's driver reports its host-visible type on the full 93.6 GiB heap
(BAR1 is 128 GiB) and `maxMemoryAllocationSize` is unlimited, yet the
139th 256 MiB block fails. On the L40S the same limit, smaller, falls below
Blenderbot's peak. The pool's existing recovery (release empty cached
blocks, retry) could not help: nothing was cached.

## Unmapped device memory

Blocks now have a placement (`memory::Placement`):

- `Mapped`, as before (host-visible, device-local where it can be, host
  cached too where it can be), tried first;
- `Unmapped`: device-local memory, not host-visible where the device has
  such a type (that is what the drivers limit), used once a mapped block
  is refused;
- `Host`: host-visible, host-cached, not device-local where it can be:
  staging for unmapped memory, and the device's error word (the host reads
  it after every synchronisation, so it lives in host memory).

The pool remembers a refused mapped block (`mapped_full`) and creates
unmapped blocks directly until a mapped block is released, so a process at
the limit pays for one failed allocation, not one per block.

### Host pointers that are keys

PyTorch's storages need a data pointer per allocation, and csrc finds an
allocation (and its GPU address) by that pointer. Unmapped allocations get
pointers from a range no process memory has, from 2^56 up (non-canonical
on x86-64, past Arm64's 48- or 52-bit addresses), never reused. A host
access through one faults at once rather than reading something else.
32-bit hosts (Termux on 32-bit Arm) have no unmapped memory; their GPUs are
integrated anyway.

### The host's reads and writes

Every place the host touched a tensor's memory now goes through three
calls (crates/pyvk/src/ffi.rs, `Runtime::read`, `write`, `copy`):

- `vkt_read(address, dst, n)`: after the queued work, `memcpy` from the
  mapping; for unmapped memory, a copy command into host staging memory
  (64 MiB at a time), wait, `memcpy`.
- `vkt_write(address, src, n)`: the reverse; the last piece's copy is left
  queued (later commands are ordered after it).
- `vkt_copy(dst, src, n)`: a copy command, ordered after the queued work
  like a kernel. The allocator's `copy_data` (copy on write) and storage
  resizes use it; they used to wait for the queue and `memcpy`.

In csrc (backend.cpp): uploads (`write_device`); downloads, `.item()` and
the CPU fallback's copies (`host_readable`: a view of the mapping, or a
copy of the memory the tensor's elements span, with its strides); strided
uploads of types with no copy kernel (the span read, updated on the host,
written back). Nothing else in csrc or the Python package dereferences a
vk tensor's data pointer; `TORCH_VK_BARRIERS=check`, which hashes live
memory through the mapping, skips unmapped allocations.

## Allocation errors and limits

- A request larger than `maxBufferSize` (one tensor is one buffer) is
  refused with both sizes named. `maxMemoryAllocationSize` is, per the
  specification, the size past which an allocation *may* fail: larger
  blocks are now tried (the pool used to refuse them up front), and the
  error names the limit when one fails.
- When no block can be made, the error names the request, the block size,
  what the pool holds (mapped and unmapped, in use and cached), and for
  each kind of memory why it refused, with its memory type and heap.
- Existing blocks are now chosen by best fit across all blocks (of the
  placement), not the first block with room: a large block left empty is
  no longer split by small requests while a tighter range exists, which
  kept it from ever being released.

## Testing

- `TORCH_VK_UNMAPPED=1` (developer switch) puts every tensor in unmapped
  memory, on any device: on lavapipe that is the same memory, not mapped.
  `tests/python/test_core.py::test_unmapped_memory` runs uploads and
  downloads (whole, strided, 80 MB in pieces), `.item()`, strided uploads
  with and without a copy kernel, a storage resize, copy on write, the
  device's index error and the CPU fallback under it;
  `test_allocation_past_the_device_says_so` checks the error for a request
  past every limit.
- `crates/vkcore/tests/runtime.rs`: unmapped memory through staging (two
  pieces), kernels on it, copies between placements; best fit leaving an
  empty large block alone.
- The whole Python suite with `TORCH_VK_UNMAPPED=1`: on lavapipe, 2008
  passed (all).

## Results

Filling a GPU with float32 tensors, each filled by a kernel
(`torch.empty(...).fill_()` until an allocation fails):

| GPU | main | this change |
|---|---|---|
| Tesla M4 (3.75 GiB) | fails at the first 256 MiB tensor | 3.5 GiB (14 x 256 MiB, all unmapped) |
| H100 NVL (93.6 GiB) | 34.5 GiB (138 x 256 MiB) | 92 GiB (92 x 1 GiB, 58 of them unmapped) |

The last error now reads, on the H100: "out of device memory allocating
1073741824 bytes (1.00 GiB) (a new block of ...); the pool holds ... (92.00
GiB) in 92 blocks (... 58.00 GiB of it unmapped), ... in use by tensors;
mapped memory: refused earlier, no mapped block released since (memory
type 4, heap 0 of ... 93.58 GiB); unmapped device memory: A device memory
allocation has failed (memory type 1, heap 0 ...)".

Host access to a 1 GiB tensor on the H100 (`.cpu()`, and `copy_` from the
CPU after an add there):

| | read | write |
|---|---|---|
| mapped | 0.24 to 0.29 GB/s | 4.2 GB/s |
| unmapped (through staging) | 3.3 GB/s | 6.1 GB/s |

Reads of mapped memory on a discrete GPU are uncached reads over PCIe,
and much slower than a copy command into cached host memory; writes are
write-combined and about even. Kernels read both kinds alike (the
products of mapped and unmapped tensors above gave the expected values).

BlenderbotForCausalLM (HuggingFace suite, eager inference) on the H100
passes with and without this change, at batch 4 and 16, float32 and
bfloat16: its peak, 24 GB in `nvidia-smi`, is under the H100's 34.5 GiB.
On the L40S, the model benchmarks' memory test measures that limit (#133).

## Left open

Since reads of unmapped memory are about 11 times faster on the H100, all
tensors could live in unmapped memory on discrete GPUs, as CUDA's do, with
mapped memory kept for integrated GPUs (where it is all the memory, and
copies would be pure cost). That changes the cost of every small read
(`.item()`: a copy command and a wait instead of a load), so it is left
for measurements on more workloads; `TORCH_VK_UNMAPPED=1` runs that way
now.
