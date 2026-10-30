# Running PyTorch's test suites on "vk" on a cluster

`tools/pytorch_tests.py` runs PyTorch's own device-generic tests (`test_ops.py`,
`test_torch.py`, ...) on device `"vk"`. A full `test_ops.py` is ~33,000 tests
(~1 s each, mostly harness overhead), so it is split into shards and run in
parallel. These scripts do that on Linux machines that share a home directory
(here: workstations reached through a gateway and a PBS cluster).

## One-time setup (on any of the machines)

```sh
# the repo at ~/torch-vulkan (git clone, or a copy of a working tree)
bash ~/torch-vulkan/tools/cluster/setup.sh
```

`setup.sh` installs Rust and the Vulkan SDK (for `slangc`) in the home
directory, checks out PyTorch's `test/` directory at the installed torch's
commit into `~/pytorch-src`, creates `~/torch-vulkan/.venv` with torch, and
builds (Cargo's files on local `/tmp`; network home directories are slow for
them). Large, rarely read things (the SDK, the Rust toolchain) can live on a
bigger, slower volume behind symlinks; the venv should stay on the fast one,
since every test process imports torch from it.

## Running shards

- `run-shard.sh SHARD SHARD_COUNT DRIVER OUT_DIR TEST_FILE...`: one shard, at
  low priority; `DRIVER` is `lavapipe` (Mesa's CPU Vulkan driver: any
  machine, no GPU) or `nvidia`.
- `pbs-lavapipe.sh`: a PBS job array, one single-core lavapipe shard per job.
- `pbs-gpu.sh`: one GPU job running several shards side by side on its GPU.

Each script's header has a `qsub` example. Pass several test files in
`TEST_FILES` to run each one's slice in the same shard. To re-run only the
tests that failed, give the runner `--ids FAILURES.txt --name NAME` in
`TEST_FILES` instead of file names.

Outputs (per shard, in `OUT_DIR`): `<run>.jsonl` (each test's outcome, written
as it finishes, so crashes and hangs lose one test, not the shard),
`<run>.log`, and after the CPU re-check `<run>.cpu-too.txt` (failures that
also fail on the CPU).

## Merging

```sh
python tools/pytorch_tests.py --merge --out OUT_DIR
```

prints outcome totals and the most common vk-only failure messages over all
shards in `OUT_DIR`, and writes `<run>.failures.txt` (vk-only failures) and
`<run>.expected.txt` (failures matching the runner's `EXPECTED` list, with
the reason). Concatenating the `failures.txt` files gives the input for an
`--ids` re-run.

## Things learned the hard way

- A kernel that never finishes hangs the GPU (Linux compute drivers have no
  watchdog), and pytest's default timeout cannot interrupt a process waiting
  on the GPU: shards use `--timeout-method thread`, and the runner records the
  test as crashed and continues.
- PBS queues limit CPUs (and on GPU queues, jobs) per user per queue, not
  jobs; node memory, not CPUs, is often what limits how many shards run.
- Rebuilding the extension while shards are running can crash them (the
  shared library is replaced under them): wait for runs to finish first.
