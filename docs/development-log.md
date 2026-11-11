# Development log

How torch-vulkan was built, milestone by milestone: design notes, what each
milestone added, what was measured and what was left for later. Each
milestone has its own file in [log/](log/), named `MNN-topic.md`; a new
milestone adds a new file rather than editing an existing one.

## Milestones

The list below runs to M22; later milestones are the newer files in [log/](log/).

- [M22: Cooperative-matrix GEMM](log/M22-cooperative-matrix-gemm.md)
- [M21: Faster FFT kernels](log/M21-faster-fft-kernels.md)
- [M20: Subgroup GEMM](log/M20-subgroup-gemm.md)
- [M19: Matrix-vector products and LAPACK's eigh](log/M19-matrix-vector-products-and-lapacks-eigh.md)
- [M18: Sparse tensors](log/M18-sparse-tensors.md)
- [M17: Phones and small GPUs](log/M17-phones-and-small-gpus.md)
- [M16: Dedicated kernels](log/M16-dedicated-kernels.md)
- [M15: PyTorch's own tests](log/M15-pytorchs-own-tests.md)
- [M14: Complex completion and the remaining ops](log/M14-complex-completion-and-the-remaining-ops.md)
- [M13: Linear algebra on the GPU](log/M13-linear-algebra-on-the-gpu.md)
- [M12: Neural-network ops](log/M12-neural-network-ops.md)
- [M11: Random numbers](log/M11-random-numbers.md)
- [M10: Sorting, searching and statistics](log/M10-sorting-searching-and-statistics.md)
- [M9: Data movement and indexing](log/M09-data-movement-and-indexing.md)
- [M8: Element-wise ops](log/M08-element-wise-ops.md)
- [M7: Dense linear algebra](log/M07-dense-linear-algebra.md)
- [M6: Complex numbers and FFT](log/M06-complex-numbers-and-fft.md)
- [M5: Matrix products](log/M05-matrix-products.md)
- [M4: Reductions, scans, float64 math](log/M04-reductions-scans-float64-math.md)
- [M3: PyTorch backend](log/M03-pytorch-backend.md)
- [M1: Core runtime](log/M01-core-runtime.md)
- [M0: Verified on this machine](log/M00-verified-on-this-machine.md)

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| GPU API | Vulkan 1.1+ via [`ash`](https://crates.io/crates/ash) (1.2/1.3 features as extensions) | raw device pointers, explicit memory types, low dispatch overhead; runs on Intel, AMD, NVIDIA, and phone GPUs (Mali) |
| Core language | Rust (`crates/vkcore`) | Memory/queue/pipeline management; exposed through a C ABI so only a thin layer is tied to PyTorch's C++ ABI |
| PyTorch integration | C++ backend on `PrivateUse1` (`crates/pyvk/csrc`), calling the Rust core through a C ABI | PyTorch's backend API is C++-only; only this layer is tied to a PyTorch release |
| Device name | `vk` | PyTorch already has a built-in `vulkan` device type (its old, unmaintained mobile backend) |
| Tensor addressing | `VK_KHR_buffer_device_address`: tensors are `(pointer, offset, sizes, strides)` passed in push constants | Views and strides become pointer arithmetic, as in CUDA; no descriptor sets |
| Memory | Prefer device-local + host-visible + coherent memory | On integrated GPUs (UHD 620, Strix Halo) CPU and GPU share tensors with no copies |
| Kernels | [Slang](https://shader-slang.org) → SPIR-V (`crates/vkcore/shaders`, compiled by `build.rs`) | Generics, pointers, autodiff; can later also target CUDA/Metal |
| FFT | VkFFT (planned) | Complex + float64 FFT on Vulkan |
| Matmul | Own kernels, modelled on llama.cpp's `ggml-vulkan` | No mature Vulkan BLAS exists |

## Layout

```
Cargo.toml                 workspace
crates/vkcore/             Rust Vulkan core
  src/context.rs           device + queue, capabilities, validation reporting
  src/memory.rs            caching sub-allocator (Pool, Allocation, RangeAllocator)
  src/kernels.rs           pipeline cache + embedded SPIR-V table
  src/stream.rs            batched in-order command stream
  src/lib.rs               Runtime (all of the above) + helpers
  shaders/*.slang          compute kernels (templated per dtype), compiled to SPIR-V by build.rs
  shaders/opcodes.txt      operation codes shared by shaders and C++ (generated headers)
  shaders/_strided*.slang  shared push-constant layouts + strided index math
  shaders/_fmath.slang     float math by dtype: built-ins + _math32 (float32), _math64* (float64)
  shaders/test/            kernels for vkcore's own tests (feature test-kernels)
  tests/runtime.rs         GPU tests (batching, dependencies, reuse, copies, pool, cache)
  examples/smoke.rs        hardware check
  examples/dispatch_bench.rs  per-op overhead, synchronized vs batched
crates/pyvk/               extension module torch_vulkan._native
  csrc/backend.cpp         allocator, device guard, hooks, views, resize, copies, CPU fallback
  csrc/launch.cpp          strided element-wise launches (csrc/strided.h plans dims)
  csrc/ops.cpp             unary, binary and comparison aten kernels
  csrc/elementwise.cpp     activations, clamp, predicates, where, lerp/addcmul, compositions
  csrc/indexing.cpp        indexing, accumulation, nonzero, cat/flip/roll/padding, arange/linspace/eye
  csrc/sorting.cpp         sort and order statistics, unique, searchsorted, cummax/cummin
  csrc/random.cpp          Philox generator and random fills, multinomial, randperm, dropout
  csrc/convolution.cpp     convolution (im2col + GEMM), transposed, backward
  csrc/normalization.cpp   batch norm entry points, attention math path
  csrc/reduce.cpp          reductions, arg-reductions, scans
  csrc/linalg.cpp          matrix products (batched strided GEMM)
  csrc/fft.cpp             FFTs (Stockham + Bluestein, r2c/c2r)
  csrc/dense.cpp           LU, Cholesky, QR, Jacobi eigh/SVD, lstsq, eig, LDL, triangular solves, tril/triu
  csrc/special.cpp         special functions (Bessel, gamma family, zeta, polynomials)
  shaders/factor.slang     factorization panels, Jacobi rotations, LDL (one workgroup per matrix)
  shaders/eig.slang        Hessenberg reduction, shifted QR, Schur eigenvectors
  shaders/special.slang    special functions; _special.slang is generated by tools/port_special.py
  src/ffi.rs               C ABI from C++ to the Rust runtime
  src/lib.rs               Python functions (synchronize, stats, fallback counts)
  build.rs                 compiles csrc/ against the target Python's libtorch
python/torch_vulkan/       backend naming, torch.vk device module, _decomp.py (PyTorch
                           decompositions as kernels), _kernels.py (Python kernels)
tests/python/test_*.py     results on "vk" vs CPU (statistics for random ops)
tools/opinfo_coverage.py   runs PyTorch's OpInfo samples on "vk": GPU / fallback / mismatch per op
tools/port_special.py      regenerates shaders/_special.slang from torch's ATen/native/Math.h
tools/pytorch_tests.py     runs PyTorch's device-generic test files on "vk" (M15), with a CPU re-check
tools/cluster/             runs them on a PBS cluster (full-run.sh), and a dev copy there (dev.sh)
tools/dev/                 pt.py (one test file on "vk", tracebacks), optrace.py (ops differing from
                           the CPU inside a test), seedsweep.py (a test under shifted seeds)
.github/workflows/ci.yml   Linux CI on lavapipe (Mesa's CPU Vulkan) with validation
benchmarks/elementwise.py  per-op overhead, element-wise, reduction and scan throughput vs CPU
benchmarks/matmul.py       matrix multiply GFLOP/s vs CPU
benchmarks/fft.py          FFT timings vs CPU
benchmarks/linalg.py       solve / cholesky / inv timings vs CPU
pyproject.toml             maturin build of the Python package
```

## Building

Requirements: Rust (rustup, MSVC toolchain), Visual Studio Build Tools with C++,
a Vulkan 1.3 driver, and the Vulkan SDK (1.4.363 here: `slangc`, validation
layers, `vkconfig`). `build.rs` finds `slangc` via `SLANGC`, then `VULKAN_SDK`, then PATH.

```powershell
$env:PATH = "$HOME\.cargo\bin;$env:PATH"
cargo run --release --example smoke
```

Tests and examples under validation (synchronisation and best-practices checks included):

```powershell
$env:TORCH_VK_VALIDATE = "1"
cargo test --release
cargo run --release --example smoke
```

All 15 Rust tests (8 unit tests, 7 GPU tests) and both examples run with no
validation messages.

Python package (editable install into `.venv`, CPU build of PyTorch):

```powershell
uv venv .venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install numpy pytest maturin
.\.venv\Scripts\maturin develop --release --uv   # compiles csrc/ against this venv's torch
$env:TORCH_VK_VALIDATE = "1"; .\.venv\Scripts\python -m pytest -q
```

All Python tests pass with no validation messages. Op coverage:
`python tools/opinfo_coverage.py --jobs 4` (writes `tools/opinfo_coverage.json`).

## Development workflow

Checks, cheapest first; a change goes on to the next only once it passes:

1. Local: `.\.venv\Scripts\python -m pytest -q` (seconds), and the affected
   PyTorch tests with `python tools/dev/pt.py test_linalg.py -k "..."`.
2. Lavapipe, in the dev copy on the cluster (a second checkout with its own
   venv, rebuildable while a full run uses the main one), from the local
   repo root:

   ```sh
   tools/cluster/dev.sh build                  # sync the working tree, rebuild (~1 min for a C++ change)
   tools/cluster/dev.sh run ID...              # test ids through the runner, CPU re-check included
   tools/cluster/dev.sh pt test_nn.py -k ...   # pytest with tracebacks
   tools/cluster/dev.sh rss ID...              # time and peak memory per test
   tools/cluster/dev.sh py script.py           # any script, on lavapipe
   ```

   Lavapipe finds what real GPUs hide: races between a workgroup's SIMD
   chunks, loads on untaken paths, inaccurate built-ins.
3. Full run, once per batch of changes and before every push: on the head
   node, after `tools/cluster/sync.sh HOST --build -J GATEWAY`,
   `bash tools/cluster/full-run.sh NAME --gpu` (48 lavapipe shards and one
   H100 job, about an hour; see `tools/cluster/README.md`). Push only if no
   test that passed in an earlier run fails.

For a failure, `tools/dev/optrace.py` shows which ops differ from the CPU
inside a test, and `tools/dev/seedsweep.py` whether the CPU passes the same
test under other seeds. A failure that is not a vk bug (a tolerance tuned to
one seed or device, a CUDA-specific expectation, a time or memory limit of
the runner) gets an `EXPECTED` entry in `tools/pytorch_tests.py` with the
evidence as its reason, rather than a workaround.

## Roadmap

1. ~~**M1 core runtime**~~: done (see above). Deferred: finer-grained barriers
   (per-allocation dependency tracking) and a separate transfer queue.
2. ~~**M2 Python prototype**~~: done; superseded by M3.
3. ~~**M3 native backend**~~: done (see above). Not yet: multiple devices,
   pinned memory. (A GPU random-number generator came with M11; bool binary
   ops and 8/16-bit accumulation run on the GPU since.)
4. ~~**M4 reductions, scans, float64 math**~~: done (see above); the ops
   it left (`logcumsumexp`, `cummax`, `median`, `masked_select`, float64
   `pow`/`atan2`, the arg-reduction thread-per-output mode) came later. Not
   yet: Payne-Hanek reduction for huge sin/cos arguments.
5. ~~**M5 matrix products**~~: done (see above). Not yet: a faster
   subgroup/cooperative-matrix kernel. (Integer GEMM, 16-bit products in
   float32 and convolutions, 1-D to 3-D and transposed, run on the GPU.)
6. ~~**M6 complex numbers + FFT**~~: done (see above), with an own FFT
   rather than VkFFT (VkFFT's Vulkan backend needs the raw device/queue
   handles and runtime glslang). Faster kernels came with M21. (Complex
   reductions, scans, matmul and transcendentals came with M14.)
7. ~~**M7 dense linear algebra**~~: done (see above). Not yet: blocked QR
   (compact-WY updates), batched GPU paths, look-ahead, eigh/SVD on the GPU.
8. ~~**M8 element-wise completeness**~~ and infrastructure cleanup: done (see
   above). Since: float16/bfloat16 arithmetic (in float32, M16), complex
   transcendentals (M14), `polygamma`, Bessel and the other `torch.special`
   functions. Not yet: 8/16-bit integer arithmetic in its own width (it
   computes in int32).

Goal for M9-M14: every dense aten op runs on Vulkan (`TORCH_VK_STRICT=1` with
no fallbacks in the OpInfo sweep).

9. ~~**M9 data movement and indexing**~~: done (see above). (`repeat_interleave`
   with a tensor of repeats and `tril_indices`/`triu_indices` came with M17.)
10. ~~**M10 sorting, searching and statistics**~~: done (see above).
    (`histogram`/`histogramdd` came with M14, `unique(dim=...)` with M17.)
    Not yet: a radix sort for very long
    rows (bitonic is O(n log^2 n)), parallel scans for single long rows in
    cummax/cummin/logcumsumexp.
11. ~~**M11 random numbers**~~: done (see above). (`rrelu`'s noise came
    with M12; `poisson`, `_standard_gamma`, `binomial`, `_sample_dirichlet`
    with M17. `random_` on complex dtypes is an error on the CPU too.)
12. ~~**M12 neural-network ops**~~: done (see above), CTC loss,
    `segment_reduce` and integer convolution included (integer pooling
    without float64 and bicubic `grid_sample`'s backward came with M17,
    native kernels for the hottest composed ops with M16).
13. ~~**M13 linear algebra on the GPU**~~: done (see above). Not yet:
    general (nonsymmetric) `eig`/`eigvals`, `ldl_factor`/`ldl_solve`
    (Bunch-Kaufman pivoting). (Complex matrices came with M14.)
14. ~~**M14 complex completion**~~: done (see above); the OpInfo sweep runs
    every op on the GPU. Not yet: accuracy-faithful complex edge cases
    (infinities, branch cuts at signed zeros), a faster eig (multishift QR,
    aggressive deflation).
15. ~~**M15 PyTorch's own tests**~~: done (see above). Not yet: tensors of
    2^31 or more elements (32-bit element offsets; lavapipe's buffers stop at
    2 GiB), float32 subnormals on devices that flush them, and long float64
    kernels on Windows GPUs (the float64 zeta test outlasts the Intel UHD
    620's watchdog and loses the device). (lavapipe's slowest tests, the
    SVD-heavy ones and grid_sampler_3d's, got faster in M16.)
16. ~~**M16 dedicated kernels**~~: done (see above), rms_norm and the
    smooth-L1/Huber backward included.
17. ~~**M17 phones and small GPUs**~~: done (see above). (int64 products
    stay in the tiles: they are bound by the emulated 64-bit multiplies, and
    an int64 gemm4 ran at 0.3 GOP/s on the A13 against 2.5 in the tiles;
    making gemm4 generic over it also slowed its float32 path 4x.)
18. ~~**M18 sparse tensors**~~: done (see above). Not yet: a single merge
    kernel for COO + COO (positions, keys and values in one pass).
Testing throughout: PyTorch OpInfo samples compared against the CPU (`tools/opinfo_coverage.py`).
