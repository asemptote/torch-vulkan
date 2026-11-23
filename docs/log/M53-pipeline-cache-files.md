# The pipeline cache across versions (M53)

The compiled-pipeline cache (`VkPipelineCache`, `crates/vkcore/src/kernels.rs`)
was one file, `pipelines.bin` in the cache directory. Each process loaded
it whole, compiled what it needed, and wrote back everything the driver
held: the loaded data and the new pipelines. Nothing ever left it. Every
build whose kernels differed added its pipelines to the old ones, so the
file grew with every upgrade and every development build: 144 MB on the
development laptop (issue #50). Two GPUs sharing the directory also
overwrote each other's file, since the driver ignores data whose header
names another device and returns only its own.

## One file per build and device

- The file is now `pipelines-<key>.bin`. The key is an FNV-1a hash of:
  - the kernels' SPIR-V: `SPIRV_HASH`, which build.rs computes over every
    kernel's name and bytes, for compiled and prebuilt (Termux) kernels
    alike;
  - the identity the driver checks in the data's header: vendor and device
    IDs, driver version and `pipelineCacheUUID`.
- A new build, version or driver therefore starts an empty file of its own
  rather than adding to the last one. A file only holds the pipelines that
  one build compiled on one device. These are the kernels times the
  specialisation constants in use, a bounded set: a run of tests/python on
  the H100 leaves a 79 MB file.
- At exit a process writes its file only if the driver's data changed.
  Otherwise it just sets the file's modification time to now. Test runs
  start many short processes, which used to rewrite the whole file each
  time. Writes still go through a temporary file and a rename, so
  concurrent processes never read a partial file. The last writer wins, and
  a pipeline lost that way is compiled again next time.

## Pruning

The cache directory is pruned when a process creates its kernel cache, at
first use of the device:

- the unversioned `pipelines.bin` of releases up to 0.2 is removed;
- other `pipelines-*.bin` files unused for 30 days are removed: no process
  has written them or set their modification time since. These are earlier
  builds and versions, or GPUs and drivers no longer present;
- of the other files that remain, the most recently used are kept up to
  256 MB in all (about three such NVIDIA files), and the rest removed.
  Several installed versions or GPUs in use at once keep their files,
  within that cap. The current file is never removed;
- temporary files over a day old are removed: a writer died before
  renaming them.

Pruning is best effort. A file another process is reading or removing at
the same time costs at most a recompile.

`TORCH_VK_CACHE_DIR` still names the directory. Set but empty, it now means
no cache on disk (it used to write `pipelines.bin` into the working
directory). docs/api.md describes both.

## Tests

- vkcore: `prunes_stale_cache_files` checks the rules on a directory of
  dated files (the current file however old, ages, the cap, legacy and
  temporary files, files of other names left alone); `cache_keys` checks
  that the key changes with the SPIR-V and with the device's UUID.
