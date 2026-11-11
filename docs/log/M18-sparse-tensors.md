# Sparse tensors (M18)

Sparse COO, CSR, CSC, BSR and BSC tensors on "vk"
(`crates/pyvk/csrc/sparse.cpp`): their indices and values are ordinary vk
tensors, so PyTorch's device-generic sparse code serves as is on
`SparsePrivateUse1` / `SparseCsrPrivateUse1` (construction, the getters,
copies, the zero-preserving unary functions, norms, cat, permute), and the
parts PyTorch writes per device are tensor ops here, running as Vulkan
kernels:

- coalescing (one int64 key per entry, a stable sort, each run's values
  summed in index order in one launch of `segreduce.slang`; entries already
  in order skip it), sums of two coalesced tensors as a merge without a
  sort (2-3x faster than sorting their concatenation at 10^4-10^6 entries),
  and PyTorch's `flatten_indices` stub (its coalesced-constructor check)
  registered for vk;
- every conversion between dense and the five layouts, batched and hybrid,
  with PyTorch's structure: from a sparse source the specified elements
  (explicit zeros included) decide the entries and blocks, from a dense one
  the nonzeros; the conversions PyTorch refuses fail with its errors;
- sparse + sparse and dense + sparse (coalesced operands give coalesced
  sums), products with scalars and with dense and sparse tensors,
  broadcasting as PyTorch's (a dense operand's batch dims become sparse
  dims), `sparse_mask` and its gradient projection;
- `mm`/`addmm`/`addmv`/`bmm`/`hspmm`/`torch.sparse.mm` for any mix of dense
  and compressed operands, with autograd and the dense ops' errors; sparse
  @ sparse from the entries (each entry of the first meets its row of the
  second, found by a sorted search; the products coalesced), so memory
  grows with the products, not the dense size; `sampled_addmm`, batched
  and broadcast;
- sparse `softmax`/`log_softmax` and their gradients (over the specified
  elements), the gradient of sparse `sum`, sum and prod of compressed
  tensors over dims (the entries accumulated by their kept indices; complex
  products pairwise by sorted runs), `triangular_solve` with a compressed
  matrix.

PyTorch's test_sparse.py and test_sparse_csr.py pass on vk but for tests
whose expectations name other devices (printing's per-device files,
error-message patterns for CPU/CUDA/MPS, products the CPU refuses), listed
with their reasons in `tools/pytorch_tests.py`. They run locally, not in
the cluster's full runs (their slowest gradchecks take minutes each).

The common conversions and masks of one matrix take few kernel launches
(an entry's row by a sorted search of the compressed indices, a sorted
check before coalescing): CSR to dense 9, dense to CSR 14, `sparse_mask`
with a CSR mask 10, coalescing entries already in order 10.

Not yet (M18): tiny sparse tensors still take several times the CPU's time
(an op is 5-50 kernel launches; COO + COO's merge is the most, about 50:
each step is its own small kernel).
