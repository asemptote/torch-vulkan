# Sorting, searching and statistics (M10)

- **Sort** (`shaders/sort_keys.slang`, `shaders/sort.slang`): values become
  unsigned keys in PyTorch's order (NaN largest, -0 equal to +0, inverted
  for descending), then a bitonic sort of (key, index) pairs runs over all
  rows at once, in shared memory for 1024-element blocks. Breaking ties by
  index makes the result exactly the stable sort, for every real dtype
  including the 8- and 16-bit ones.
- Built on it: `sort`, `argsort`, `msort`, `topk`, `kthvalue`, `median`,
  `nanmedian`, `mode` (`shaders/rowscan.slang` finds the longest run),
  `unique` (with inverse and counts) and `unique_consecutive`.
- `searchsorted` and `bucketize` (`shaders/searchsorted.slang`, a binary
  search per value with PyTorch's comparisons), `cummax`, `cummin` and
  `logcumsumexp` (sequential per row, matching the CPU's NaN and tie rules).
- **Accumulation into repeated indices** without atomics (this GPU has no
  64-bit or float atomics): source elements are sorted by destination and
  `shaders/segreduce.slang` folds each run in source order, the same order as
  the CPU's loop, so float sums agree bit for bit. Used by `index_add`,
  `index_reduce`, `scatter_add`, `scatter_reduce` (sum/prod/mean/amax/amin,
  with or without self), `scatter(reduce=...)`, `index_put_`/`put_` with
  `accumulate=True`, `bincount` and `histc`.
- Small reductions from existing kernels: `aminmax`, `var_mean`,
  `std_mean`, `nansum`, `trace`.
- Which of several equal elements `topk`, `kthvalue`, `median` and `mode`
  report is unspecified in PyTorch (the CPU uses unstable partial sorts); we
  return the stable-order one, and tests check values plus that the indices
  select them.

| sort, float32 | vk (UHD 620) | CPU (stable, 4 threads) |
|---|---|---|
| 1 x 2^20 | 78 ms | 533 ms |
| 64 x 2^16 | 145 ms | 546 ms |
| 4096 x 1000 | 61 ms | 326 ms |

OpInfo sweep: 346 of 659 ops fully on the GPU (317 after M9), no mismatches.
