# Barriers only where a dependency needs one (M37)

Issue #45. Until now `vkcore::Stream` put a memory barrier before every
command, so every kernel waited for the one before it, whether or not it
used its results. Kernels reach memory only through buffer device addresses
in their push constants, so the stream could not see what each one touches.
Now it can: each dispatch declares the memory it reads and writes, and the
stream puts a barrier before a command only when it conflicts with one
since the last barrier.

## Measured first

`benchmarks/barriers.py` times each case whole, synchronize included
(median of 300 runs), and counts the barriers per run:

- 10 or 1000 independent adds: `torch.add(a[i], 1, out=o[i])` on distinct
  tensors of 256 float32 values;
- 10 or 1000 dependent adds: `x.add_(1)` on one tensor;
- one `torch.optim.Adam(foreach=True)` step over the 32 parameters of a
  small MLP (8 layers of 256 x 256 linear, layer norm and GELU);
- that MLP's forward, and forward plus backward, on a batch of 32.

On main, H100 NVL, microseconds:

| case | main | main, `TORCH_VK_BARRIERS=full` |
|---|---|---|
| 10 independent adds | 74.3 | 77.0 |
| 10 dependent adds | 78.0 | 82.1 |
| 1000 independent adds | 3777 | 3882 |
| 1000 dependent adds | 4155 | 4241 |
| Adam step | 1367 | 1450 |
| MLP forward | 565 | 573 |
| MLP forward + backward | 2108 | 2155 |

- Independent kernels cost what dependent ones do: every command waits for
  the one before.
- With 1000 adds, about 3.8 us each, the host records about as fast as the
  GPU runs serialised kernels, so leaving out barriers can gain at most a
  fraction of a microsecond per op there. Short bursts, where the GPU's
  serialised run comes on top of the recording, have more to gain.
- The Adam step (32 tensors, 224 kernels) and the backward pass have many
  independent kernels; the MLP's forward is a dependent chain, which
  needs every barrier (fusion, #28, is the fix there).

## The stream (`vkcore/src/stream.rs`)

As in llama.cpp's Vulkan backend, the stream keeps the accesses of the
commands recorded since the last barrier: GPU address ranges, each read or
read and written ([`Access`]). A new command needs a barrier when:

- it reads or writes memory written since the last barrier (read after
  write, write after write), or writes memory read since then (write after
  read). All readers since the last barrier are kept, not only the last
  access: PyTorch's old Vulkan backend tracked one access per tensor and so
  missed write-after-read hazards;
- it is undeclared, or an undeclared command was recorded since the last
  barrier: an undeclared command is ordered against everything before and
  after it, as before (vkcore's own `Runtime::dispatch`, used by its tests
  and examples);
- it is the first transfer command (copy, fill) ever recorded (see below);
- or 128 accesses are kept: the barrier then empties the list, so checking
  a command stays a scan of a short list.

The barrier is one global `VkMemoryBarrier`, as in Dawn, from the compute
stage's writes to the compute stage's reads and writes (compute to
compute, as 0.2's narrowed barriers between dispatches). A pipeline barrier's
first scope is every command before it in submission order, earlier
submissions included, so once it is recorded every earlier write is
available and visible to every later command, and the list is emptied. Once
a transfer command has been recorded, barriers include the transfer stage
on both sides; the first transfer gets a barrier of its own, because writes
made visible by earlier barriers were made visible to the compute stage
only. (The C++ backend issues no transfers: its copies and fills are
kernels.)

The list is kept across submissions. 0.2 put a barrier before each batch's
first command because submission order alone orders nothing (lavapipe ran
a batch's first dispatch while the previous batch's last one still ran);
now that barrier is there exactly when the first command conflicts with
one before it. A synchronize does not empty the list either: the host has
waited, but a later command still needs a barrier on the GPU to see earlier
writes. That costs at most one barrier after each synchronize. The Intel
bound on submissions in flight (M34) is a wait on the host before a
submission and is unchanged.

Between barriers the GPU may run commands in any order or concurrently,
which is right for any two that touch disjoint memory or only read common
memory.

## Where the accesses come from (`vkcore/src/declare.rs`)

The design sketch for #45 captured accesses in csrc's `gpu_address` and
`gpu_address_out`, inside a scope per op. That depends on every address a
kernel uses having passed through them within the op's scope: an address
computed once and reused across ops (a cached FFT table, the device error
word) or a scope missing at one of hundreds of entry points would be a
silent race. Instead the runtime reads the addresses where the kernels do:

- **Kernels get every address from their push constants.** No shader loads
  an address from memory (no pointer-to-pointer types; checked over all 72
  shaders), and csrc's push-constant structs hold addresses as `uint64_t`
  fields, so at 8-byte-aligned offsets.
- So `Runtime::dispatch_declared` looks up every 8-byte word of a
  dispatch's push constants in the pool's live allocations (a map from GPU
  address to size, kept by `Pool::alloc` and `free`). A word inside an
  allocation is taken for an address into it, and the dispatch is declared
  to read and write **the whole allocation**: a kernel may index anywhere
  in it from that address (negative strides, data-dependent gathers), but
  not past it. A word that is not an address and happens to fall in an
  allocation only adds a barrier. A word inside the pool's memory but in
  no live allocation (freed or guard bytes) makes the dispatch undeclared.
- Every dispatch of the C++ backend goes through this (`vkt_dispatch`), so
  none is left undeclared, and there is nothing to annotate per op.

The cost of whole allocations is false dependencies: every reader also
counts as a writer, and writes to disjoint parts of one tensor conflict. For
the strided element-wise kernels (`csrc/launch.cpp`) the call site knows
better, and declares exactly ([`Declared`]: for an address in the push
constants, the bytes the kernel touches through it, and whether it writes
them):

- the kernels on the `_strided` layouts that touch each operand only at
  its own strided offsets, out written and the inputs read (unary, binary,
  their complex forms, cabs, cast, compare, special, ternary, where, iota,
  index, and move's copy and fill; checked in each shader): out is declared
  written over exactly the bytes it spans in that dispatch, and each input
  read over the bytes it spans. move's gather and scatter index with data
  and keep whole allocations;
- `copy8` (copies of five to eight dims) likewise.

A declaration is used only if it holds its own address and lies inside that
address's allocation; otherwise the whole allocation counts. So independent
element-wise ops, ops that read a common input, and the dispatches of the
loop over dims beyond four (one per outer index, each writing its own
slice) need no barrier between them.

Batched linear algebra needs nothing more: its kernels take the whole batch
in one dispatch (the grid's z), and its loops over algorithm steps are
dependent anyway.

## Checking it

Synchronisation validation cannot see accesses through buffer device
addresses, so it says nothing here (M25). The checks:

- **`TORCH_VK_BARRIERS=check`** runs each declared dispatch alone,
  synchronizing before and after, and fails if it changed any byte of live
  memory outside what it declared written (hashing all live allocations).
  It catches a kernel writing beyond its declaration, a strided kernel
  writing an input, and an address the scan did not find. It cannot see
  reads, and a write of the value already there.
- **Bitwise A/B.** With `TORCH_VK_READ_LOG=dir`, every read of device memory
  by the host (`.cpu()`, `.item()`) is logged with a hash of its bytes, per
  test (a pytest plugin, `tools/dev/barrier_ab.py`) or per OpInfo
  (`tools/opinfo_coverage.py`). Runs with the default barriers and with
  `TORCH_VK_BARRIERS=full` must read the same bits, except where two runs
  with the default barriers differ from each other too (uninitialized
  memory, float atomics).
- **Stress tests** (`tests/python/test_barriers.py`): random programs of 250
  ops (more with `TORCH_VK_STRESS_SEEDS` and `TORCH_VK_STRESS_STEPS`) mixing
  element-wise ops in place and out=, copies (also transposed), fills of
  rows, matrix products, reductions into a row, index_add_, gathers and
  index_put_, cumulative sums, concatenation, sorts of temporaries, and host
  reads, on views of one allocation that overlap, interleave (`S[0::2]`
  and `S[1::2]`), transpose, and have seven dims that do not merge, and on
  separate tensors. All values are small integers in float32, so the CPU's
  results must match bit for bit. Also: disjoint row writes followed by a
  reduction, many readers followed by a write, pool reuse of a freed
  temporary, host reads, and split-k products followed by independent work.
- vkcore's tests (`cargo test`) count the barriers between declared
  dispatches, check the push-constant lookup, and pass with the barriers in
  every mode.

Results (`tests/python`: 1612 tests on the NVIDIA GPUs, 1610 elsewhere;
the strict OpInfo sweep: 689 ops):

| | H100 NVL | L40S | HD 530 | lavapipe |
|---|---|---|---|---|
| tests/python, default / full / default again | all pass | all pass | all pass | all pass |
| read logs differing, default vs full | 0 of 1539 tests | 0 of 1539 | 0 of 1537 | 0 of 1537 |
| read logs differing, default vs default | 0 | 0 | 0 | 0 |
| OpInfo strict: all on the GPU, read logs differing | 689; 0 of 658 | 689; 0 of 658 | 689; 0 of 658 | 689; 0 of 658 |
| `TORCH_VK_BARRIERS=check` | not run | test_matmul, test_optim, test_barriers: 154 pass | tests/python: all pass | tests/python: all pass |
| stress tests, 200 programs of 400 ops | pass | pass | 30 x 250: pass | 6 x 250: pass |

- About 11,000 reads in tests/python and 8,100 in the OpInfo sweep
  compared bit for bit. No test read different bits in two runs with the
  same settings either, so the comparison is exact everywhere.
- The check reads all live device memory twice per kernel: 22 minutes for
  tests/python on the HD 530 and lavapipe, against under a minute and a
  half. On the NVIDIA GPUs the memory is read over PCIe, and the whole
  suite would take hours, so there it ran on the L40S over the tests of
  the paths only NVIDIA takes: cooperative-matrix and split-k products,
  plus the optimisers and the stress tests (30 minutes for 154 tests). It
  found nothing anywhere, and no address at a 4-byte offset.
- The table's runs used PyTorch 2.14.1. After moving to PyTorch 2.15, the
  A/B runs of tests/python (1625 tests) and the OpInfo sweep were repeated
  on the HD 530 and lavapipe: 0 of 1552 tests and 0 of 658 ops differing,
  no noise.
- PyTorch's test files on the H100 (the 37 files of the full runs,
  PyTorch 2.14.1), with the default barriers: 129,131 passed, 306 expected
  failures, 85 that fail on the CPU too, and no failure of vk's own, so
  nothing to re-run with `every`.

## Results

The same build in each mode, three rounds alternating, medians of the
three medians (`every` is main's barriers). Microseconds, and barriers per
run in brackets.

H100 NVL:

| case | tracked (default) | every (as main) | full |
|---|---|---|---|
| 10 independent adds | **51.2** (1) | 74.1 (10) | 77.6 (10) |
| 10 dependent adds | 79.7 (10) | 78.7 (10) | 82.5 (10) |
| 1000 independent adds | **3311** (16) | 3863 (1000) | 3947 (1000) |
| 1000 dependent adds | 4249 (1000) | 4180 (1000) | 4300 (1000) |
| Adam step | **986** (6) | 1393 (224) | 1459 (224) |
| MLP forward | 572 (24) | 567 (24) | 575 (24) |
| MLP forward + backward | 2133 (110) | 2151 (136) | 2194 (136) |

L40S:

| case | tracked (default) | every (as main) | full |
|---|---|---|---|
| 10 independent adds | **42.0** (1) | 54.8 (10) | 60.5 (10) |
| 10 dependent adds | 59.8 (10) | 59.0 (10) | 64.2 (10) |
| 1000 independent adds | **2784** (16) | 3183 (1000) | 3390 (1000) |
| 1000 dependent adds | 3607 (1000) | 3517 (1000) | 3677 (1000) |
| Adam step | **832** (6) | 1019 (224) | 1134 (224) |
| MLP forward | 447 (24) | 443 (24) | 454 (24) |
| MLP forward + backward | 1687 (110) | 1691 (136) | 1752 (136) |

HD 530 (Mesa ANV; a shared machine, so differences under about 10% are
noise there):

| case | tracked (default) | every (as main) | full |
|---|---|---|---|
| 10 independent adds | **208** (1) | 276 (10) | 282 (10) |
| 10 dependent adds | 298 (10) | 293 (10) | 296 (10) |
| Adam step | 4689 (6) | 4662 (224) | 4675 (224) |
| MLP forward + backward | 9970 (110) | 9888 (136) | 9905 (136) |

On the HD 530 a barrier left out between small kernels saves about 7.5 us
(10 adds), but the Adam step does not move: the host takes about 20 us to
record each of its kernels, far longer than the GPU takes to run them, with
or without barriers. Its 1000-add runs varied by a third from round to
round, more than any difference between the modes.

Per barrier left out:

- **A short burst of independent kernels:** 2.5 us on the H100 (10 adds,
  74.1 to 51.2 us for 9 barriers) and 1.4 us on the L40S. That is the
  serialisation itself: the drain and refill around each barrier.
- **An optimiser step:** Adam over 32 tensors goes from 224 barriers to 6
  and from 1393 to 986 us on the H100 (29% less; 1.9 us per barrier), 1019
  to 832 on the L40S (18%; 0.9 us per barrier). Each `_foreach_` op's
  kernels touch distinct tensors, and so do consecutive foreach ops until
  one comes back to a tensor an earlier one wrote.
- **Long runs of independent kernels:** 0.55 us per kernel on the H100 and
  0.4 on the L40S (1000 adds). Here the host is the limit: it records an
  add in about 3.3 us, and the GPU keeps up once it no longer waits
  between kernels.
- **Dependent chains:** nothing to leave out (every barrier is needed), and
  the bookkeeping costs the host 70 to 90 ns per kernel (1000 dependent adds:
  4180 to 4249 us on the H100). The MLP's forward pass, a dependent chain,
  is within 1% either way; with its backward pass, 26 of 136 barriers go
  and the time is within 1% too: the kernels there are larger, and the
  independent ones are few.

## Switches

- `TORCH_VK_BARRIERS=full`: a full compute and transfer barrier before every
  command, as before 0.2's narrowing; `every`: 0.2's barrier before every
  command; `check`: the check above. `torch_vulkan._native.barrier_stats()`
  counts commands, barriers and undeclared commands, and
  `_native.set_barrier_mode(...)` switches modes in a running process (for
  A/B tests).
- `TORCH_VK_READ_LOG=dir`: the read log above.

## Limits and what next

- **Whole allocations, read and written.** Outside the strided element-wise
  kernels every address counts for its whole allocation, read and written.
  Readers of one tensor (two matrix products of one input, a backward pass
  reading the forward's activations) are therefore ordered among
  themselves, and so are writes to disjoint parts of one tensor by other
  kernels. Declaring more kernels exactly (the matrix products first: out
  written, A and B read) would let them overlap; each needs its shader
  checked, and `TORCH_VK_BARRIERS=check` run over its tests.
- **The device error word** that indexing kernels set is one allocation all
  of them write, so they are ordered among themselves. It could become an
  access kind that does not conflict with itself (atomic stores of 1).
- **The check sees writes only.** A kernel reading outside its declaration
  (only possible for the exact declarations of the strided kernels, which
  were checked by reading the shaders) would not be caught by
  `TORCH_VK_BARRIERS=check`; the A/B runs and stress tests would catch its
  effect only where a race actually happens.
- **Host time.** Dependent chains gain nothing and pay 70 to 90 ns per
  kernel for the lookup and the scan. On the H100 most small ops are now
  bound by the host's recording (about 3.3 us per add), so the next gains
  are there and in submitting early (MPS and the old Vulkan backend commit
  every 16 kernels, so the GPU starts while the host records), and in
  fusion for dependent chains (#28).
- `queryEvent` still always returns true (real events on the timeline
  semaphore are separate work).
