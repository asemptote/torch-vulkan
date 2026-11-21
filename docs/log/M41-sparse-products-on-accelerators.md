# Sparse compressed products from PyTorch 2.15's accelerator tests (M41)

Issue #89. PyTorch 2.15 runs several formerly CUDA-only tests in
test_sparse_csr.py on any accelerator. On vk they found gaps in the sparse
compressed (CSR/CSC/BSR/BSC) support of M18: 43 failing tests in #83's
re-run of its failures, and one older failure,
`test_matmul_device_mismatch`. All are now fixed on the GPU, with no CPU
fallback: the whole of test_sparse_csr.py runs on the HD 530 with no aten
op on the CPU, and the only failures left are those in the runner's
`EXPECTED` list (printing's per-device files, complex `abs` that CPU and
CUDA fail as well).

As in M18, these are written with vk's tensor ops (gathers, sorted
searches, segmented sums), which run as the project's kernels.

## What PyTorch does, and vk now does

- **`bmm`/`baddbmm` with a compressed first operand** (`SparseCsrCUDA`
  only in PyTorch). The entries become a COO tensor over (batch, row,
  column) and the product runs as one block-diagonal sparse product, as
  vk's COO `bmm`. The tests build a CSR tensor whose indices have no batch
  dimension but whose shape has one (with the invariants unchecked);
  cuSPARSE reads that as one matrix for every batch (batch stride 0), and
  vk does too. `beta = 0` ignores `self`, NaN included, as for `addmm`.
- **`torch.sparse.mm(csr, dense, reduce)`** (`_sparse_mm_reduce_impl` and
  its backward, CPU-only in PyTorch) with `sum`, `mean`, `amax` and `amin`,
  following the CPU's `SpmmReduceKernel`: products in float32 for float16
  and bfloat16, rounded once; `mean` divides the rounded sum by the row's
  count; rows without entries are 0. Sums fold each row's entries in entry
  order with `segreduce` (one launch), as the CPU's loop. `arg_out` (the
  entry that gave each `amax`/`amin`, `nnz` for none) is computed only
  when autograd needs it, as on the CPU, and picks the first of equal
  entries, or the last NaN, as the CPU's strict comparison does. The
  backward is the CPU's: per-entry dot products (divided by the row count
  for `mean`), scatter-adds through `arg_out` for `amax`/`amin`.
- **`sparse_sampled_addmm`**: PyTorch's checks, in its order
  (`sparse_sampled_addmm_check_inputs`: strided `mat1`/`mat2`, matrices,
  batch sizes, then shapes). The result's indices are set directly, as
  PyTorch's resize and copy: building a tensor from them checks the
  invariants (when enabled), which an empty CSR tensor made with its
  checks off (uninitialised indices, as on the CPU and CUDA) fails.
- **`addmm(dense, compressed, compressed)`** returns a dense result:
  PyTorch's result is `empty(self.options())`, so it takes `self`'s
  layout; vk returned `mat1`'s.
- **BSR/BSC `to_dense`** checks the block sizes (`expected block sizes to
  be strictly positive`) as `sparse_compressed_to_dense`.
- **Operands on two devices.** The compressed-index validation raises the
  error of PyTorch's TensorIterator (`... at least two devices, vk:0 and
  cpu!`) instead of sending the indices to the CPU; the COO `addmm` (where
  a CPU operand arrives through the sparse dispatch key) checks devices as
  PyTorch's kernels do; and the backend's generic check names the device
  with its index (`vk:0`, as PyTorch's messages for other devices).
