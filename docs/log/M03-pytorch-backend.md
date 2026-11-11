# PyTorch backend (M3)

- `crates/pyvk/csrc/backend.cpp`: a `c10::Allocator` over the Rust memory pool
  (`data_ptr()` is the host-mapped address; the GPU address is derived from the
  allocation context), a device guard, `PrivateUse1HooksInterface`, and aten
  kernels for `empty`/`empty_strided`, views (`view`, `as_strided`,
  `_reshape_alias`), `resize_` (grows storage, keeps contents), `set_`,
  `_copy_from`/`_copy_from_and_resize` (CPU <-> vk and vk <-> vk with
  broadcasting and dtype conversion), `_local_scalar_dense`, `fill_`, `zero_`.
- `crates/pyvk/csrc/ops.cpp`: the `.out` overloads of structured ops (PyTorch's
  generated functional/in-place variants call them): 19 unary ops plus `relu`,
  11 binary ops (scalar operands, `alpha`, scalar-first `sub`), 6 comparisons
  (Tensor and Scalar), all with PyTorch's type promotion. Cases without a
  kernel (e.g. float64 `exp`) run the same overload through
  `at::native::cpu_fallback`, so results stay correct.
- Every other op: a boxed PrivateUse1 fallback to the CPU, counted in
  `torch_vulkan.fallback_counts()` and warned about once per op. Its results
  are written back with `_copy_from_and_resize`, which resizes `out=`
  arguments (the empty-result bug of torch-directml and pytorch_ocl is tested).
- Strided launches merge contiguous dims; tensors with more than 4
  non-mergeable dims loop over the outer dims on the host (one dispatch each).
- `crates/pyvk/src/ffi.rs`: the C ABI from C++ to the Rust runtime (alloc, free,
  dispatch, synchronise); the C++ is compiled by `crates/pyvk/build.rs` against
  the libtorch of the target Python and linked into the same extension module.
- `python/torch_vulkan`: names the backend `vk`, registers `torch.vk`, adds
  `Tensor.vk()`, and exposes `synchronize`, `memory_stats`, `empty_cache`,
  `fallback_counts`.

The M2 prototype (a `__torch_dispatch__` tensor subclass) is replaced; its
kernels and tests carry over.

Kernels are Slang templates compiled per dtype (`// axis T: ...` lines in the
shader; see `build.rs`), with the op chosen by a specialisation constant:
`unary`, `binary`, `compare`, `cast` (all 25 pairs of f32/f64/i32/i64/bool;
also strided copies) and `fill`. Operands are addressed as pointer + up to 4
coalesced (size, stride) dims, so broadcasting, transposes and slices need no copies.

Precision: float32 math uses the GPU's built-ins, which Vulkan only requires
to be accurate to ~2^-11 absolute (sin/cos) or a few ULP; tests use those
bounds. float64 uses `_math64.slang` (see M4 and M8).

Measured (`python benchmarks/elementwise.py`, UHD 620 vs 4-thread CPU, torch 2.14.1):

| | vk (M3, C++) | vk (M2, Python) | CPU |
|---|---|---|---|
| per-op overhead, 1024 floats | 9 µs | 136 µs | 7 µs |
| `relu(x*2+1)`, 16M float32 | 27 ms | 29 ms | 27 ms |
| `relu(x*2+1)`, 16M float64 | 37 ms | 40 ms | 51 ms |

Large arrays are memory-bound (~15–22 GB/s on this integrated GPU); per-op
overhead is now close to the CPU's.
