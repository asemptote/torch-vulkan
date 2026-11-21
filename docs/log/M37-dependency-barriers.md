# Barriers only where a dependency needs one (M37)

Issue #45. Until now `vkcore::Stream` put a memory barrier before every
command, so every kernel waited for the one before it, whether or not it
used its results. Kernels reach memory only through buffer device addresses
in their push constants, so the stream could not see what each one touches.
Now it can: each dispatch declares the memory it reads and writes, and the
stream puts a barrier before a command only when it conflicts with one
since the last barrier.

## Measured first

RESULTS_BEFORE

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

CHECK_RESULTS

## Results

RESULTS_AFTER

## Switches

- `TORCH_VK_BARRIERS=full`: a full compute and transfer barrier before every
  command, as before 0.2's narrowing; `every`: 0.2's barrier before every
  command; `check`: the check above. `torch_vulkan._native.barrier_stats()`
  counts commands, barriers and undeclared commands, and
  `_native.set_barrier_mode(...)` switches modes in a running process (for
  A/B tests).
- `TORCH_VK_READ_LOG=dir`: the read log above.

## Limits and what next

LIMITS
