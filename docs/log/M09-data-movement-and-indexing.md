# Data movement and indexing (M9)

- **Every dtype moves on the GPU.** Copies, `clone`, `contiguous`, fills and
  all data movement use `shaders/move.slang`, which works on elements by size
  (1, 2, 4, 8 or 16 bytes), so any dtype is covered. Casts now also cover
  uint8, int8, int16, float16 and bfloat16 (where the device has 8/16-bit
  storage; this one does). Float-to-small-integer casts wrap like the CPU's
  (-3.0 -> 253), and float16 conversions round to nearest even explicitly
  (Vulkan leaves that rounding to the driver).
- **Indexing** is two steps: `shaders/index.slang` turns index tensors into
  element offsets (wrapping negative indices where PyTorch does: `index`,
  `index_put_`, `take`, `put_`, `index_fill_`; rejecting them where it
  rejects them: `gather`, `index_select`, `scatter`, `index_copy`), then a
  gather or scatter in `move.slang` reads or writes the source restrided to
  the result's shape. Advanced indexing (integer and boolean, adjacent or not)
  follows PyTorch's `make_info`; boolean masks go through the GPU `nonzero`.
- **Out-of-range indices** are reported like CUDA's device asserts: the kernel
  sets an error word, and the next synchronise (`.cpu()`, `.item()`,
  `torch_vulkan.synchronize()`) raises `IndexError`. Queued work is not
  stopped, and the result of the bad op is undefined.
- **nonzero** counts with a GPU running sum (one sync for the result size, as
  on CUDA) and writes coordinates with `shaders/nonzero.slang`;
  `masked_select`, `masked_scatter_` and boolean indexing build on it.
- **cat, flip, roll, reflection/replication padding (1-3d)**, `channel_shuffle`,
  `equal`, `isin` (by broadcast comparison, up to 2^24 pairs) and
  **arange, linspace, logspace, eye** (`shaders/iota.slang`, matching
  PyTorch's formulas and float16 rounding) run on the GPU.
- Still on the CPU: ops that accumulate into repeated indices (`index_put_`
  with `accumulate=True`, `index_add`, `scatter_add`, `scatter_reduce`,
  `put_(accumulate=True)`, `bincount`, `histc`). This GPU has no 64-bit or
  float atomics, so these will use sorting (M10).

OpInfo sweep: 317 of 659 ops fully on the GPU (272 after M8), no mismatches.
