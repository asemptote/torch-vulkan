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

After changing the code locally, `tools/cluster/sync.sh HOST --build -J
GATEWAY` copies the working tree (uncommitted changes included) to the shared
home directory and rebuilds there with `build.sh` (log in
`~/torch-vulkan-build.log`, ending with `build exit N`).

## Running shards

- `run-shard.sh SHARD SHARD_COUNT DRIVER OUT_DIR TEST_FILE...`: one shard, at
  low priority; `DRIVER` is `lavapipe` (Mesa's CPU Vulkan driver: any
  machine, no GPU) or `nvidia`.
- `pbs-lavapipe.sh`: a PBS job array, one single-core lavapipe shard per job.
- `pbs-gpu.sh`: one GPU job running several shards side by side on its GPU.
- `full-run.sh NAME [--gpu]` (on the head node): the standard full run, all
  thirteen test files in 48 lavapipe shards, and with `--gpu` one H100 job
  of 16 shards alongside (a separate queue). Each takes about an hour.

Each script's header has a `qsub` example. Pass several test files in
`TEST_FILES` to run each one's slice in the same shard. To re-run only the
tests that failed, give the runner `--ids FAILURES.txt --name NAME` in
`TEST_FILES` instead of file names.

Outputs (per shard, in `OUT_DIR`): `<run>.jsonl` (each test's outcome, written
as it finishes, so crashes and hangs lose one test, not the shard),
`<run>.log`, and after the CPU re-check `<run>.cpu-too.txt` (failures that
also fail on the CPU).

### Bounds-checked runs

With `TORCH_VK_VALIDATE=gpu` (and `VK_LAYER_PATH` pointing at the SDK's
`share/vulkan/explicit_layer.d`), the Khronos validation layer checks every
memory access the kernels make through buffer device addresses against the
live allocations. The runner then records any test during which the layer
reported something with the outcome `validation` and the layer's message.
This catches out-of-bounds and null accesses that real GPUs often survive
silently. The SDK's layer needs a newer `libstdc++` than some
distributions have; put one on `LD_LIBRARY_PATH`. Pass all three to
`qsub -v`.

## Merging

```sh
python tools/pytorch_tests.py --merge --out OUT_DIR
```

prints outcome totals and the most common vk-only failure messages over all
shards in `OUT_DIR`, and writes `<run>.failures.txt` (vk-only failures) and
`<run>.expected.txt` (failures matching the runner's `EXPECTED` list, with
the reason). Concatenating the `failures.txt` files gives the input for an
`--ids` re-run.

## Queued runs: the cluster plus workstation backfill

`queue-run.sh start NAME` (from the local repo root) spreads a full run over
every machine that can take part. Each driver family gets a queue of small
shards on the shared file system: `$RUNS/NAME` for lavapipe and
`$RUNS/NAME-nvidia`.

- Cluster workers take shards from the start of each queue: the PBS lavapipe
  array, plus an H100 job and an L40S job (separate queues).
- Workstation workers (`pool.sh`) take shards from the end, one per core on
  ws0, ws2 and ws3 (lavapipe). Each is ranked: worker k runs only while
  the cores other processes use (a per-machine `top` monitor) plus k + 1
  fit in the machine. As other users' load rises, the highest ranks pause
  their shards and come back in order as it falls; a shard paused for 20
  minutes goes back to the queue. ws3's Tesla M4 can join the NVIDIA run
  (`WS3_GPU_WORKERS`), but its 4 GB is too small for some tests, so by
  default it is left for quick checks (`dev.sh` with `DRIVER=nvidia-host`).

The two sides meet in the middle, so whatever starts soonest does the most,
and a busy cluster queue no longer holds the run up. Every worker claims a
shard by renaming its file into `claimed/` (`worker.sh`).

With `--from PREV`, `plan_shards.py` plans each queue from an earlier run:
its tests and their recorded times (the runner records each test's wall
time), packed longest first into one-file shards of about 5 minutes, and
numbered from the longest. The cluster then starts the slowest tests first
instead of finishing on them, and each shard needs one pytest start-up.
`queue-run.sh status NAME` shows the progress per machine. Results merge as
for fixed shards.

## The dev copy

`dev.sh` keeps a second checkout (`DEV_DIR`, with its own venv and Cargo
target directory) for experiments, so it can be rebuilt while a full run
uses `~/torch-vulkan`. From the local repo root: `dev.sh init` once, then
`dev.sh build` (sync and rebuild), `dev.sh run ID...`, `dev.sh pt FILE -k
EXPR`, `dev.sh rss ID...` (peak memory and time per test) and `dev.sh py
SCRIPT`; the script's header lists the settings.

## Things learned the hard way

- A kernel that never finishes hangs the GPU (Linux compute drivers have no
  watchdog), and pytest's default timeout cannot interrupt a process waiting
  on the GPU: shards use `--timeout-method thread`, and the runner records the
  test as crashed and continues.
- PBS queues limit CPUs (and on GPU queues, jobs) per user per queue, not
  jobs; node memory, not CPUs, is often what limits how many shards run.
- Rebuilding the extension while shards are running can crash them (the
  shared library is replaced under them): wait for runs to finish first.
- `cc` compiles csrc/ with its `parallel` feature: one file's change rebuilds
  all fourteen, about 4 minutes one after another and 45 s in parallel.
