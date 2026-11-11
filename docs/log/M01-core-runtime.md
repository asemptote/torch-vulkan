# Core runtime (M1)

- **Memory pool** (`memory.rs`): tensors are sub-allocated (256 B aligned,
  best fit, coalescing) from 64 MiB persistently mapped, device-addressable
  blocks; larger requests get their own block. Freed memory is cached and
  reused in stream order, like PyTorch's CUDA allocator; `trim()` returns
  empty blocks to Vulkan, and allocation retries after trimming when out of memory.
- **Kernel cache** (`kernels.rs`): one shared pipeline layout (no descriptor
  sets, 128-byte push-constant block). Pipelines are keyed by
  (kernel, specialisation constants); compiled code persists across runs in
  `%LOCALAPPDATA%\torch-vulkan\pipelines.bin` (override with `TORCH_VK_CACHE_DIR`).
- **Stream** (`stream.rs`): in-order like a CUDA stream. Commands are recorded
  with a full memory barrier between them and submitted in batches of up to
  256, tracked by a timeline semaphore; command buffers are recycled.
  The host must `synchronize()` before touching memory that queued work uses.
- **Validation** (`context.rs`): `TORCH_VK_VALIDATE=1` enables the Khronos
  validation layer with synchronisation and best-practices checks; every
  message is printed and counted, and the tests fail on any.

Measured on the UHD 620 (`cargo run --release --example dispatch_bench`):
synchronising after every small kernel costs **486 µs/op**; batching brings it
to **5.6 µs/op** (~87× less).
