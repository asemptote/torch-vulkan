# Intel GPUs under Linux: i915's 20 s request watchdog (M34)

On an HD 530 (Mesa's ANV, Linux's i915), long runs of GPU work came back
with part of their results missing and no error (#72). M25 put this down to
submissions busy for "a couple of seconds" and capped batches at 2^13
workgroups on Intel GPUs. That cap helped, but the explanation was wrong.

## What happens

- **i915 cancels requests older than 20 s.** `i915.request_timeout_ms`
  (`CONFIG_DRM_I915_REQUEST_TIMEOUT`, 20000 by default) limits how long a
  submission may take, counted from when it is submitted, not from when it
  starts running. Work still unfinished then is cancelled: the dispatch
  running at that moment is cut off partway, and later ones in the
  submission never run. Its fence is signalled with `-EINTR`.
- **ANV reports success.** `vkWaitForFences`, `vkWaitSemaphores`, the
  timeline semaphore's counter and `vkDeviceWaitIdle` all return
  `VK_SUCCESS` (mesa#16166). The device stays usable. The kernel's error is
  there to read: a sync_file exported from the same `VkFence`
  (`VK_KHR_external_fence_fd`, `SYNC_IOC_FILE_INFO`) has status -4.
- **Queue time counts.** In a standalone C reproducer, 46 dispatches of
  0.44 s in one submission were cut at 20.0 s, and so were 50 of them in 50
  submissions queued back to back. Waiting on the host after each
  submission avoided it. A real hang (non-preemptible workgroups) is
  different: the engine is reset after about 7 s and ANV returns
  `VK_ERROR_DEVICE_LOST`.

## Why the batch cap seemed to fix it

The stream never waited on the host between batches, so with the cap a
2^27-entry float64 sort (172 dispatches, about 26 s of work) went as 172
submissions queued at once. What kept it under 20 s was i915 itself: it
blocks a submission while the context's ring is nearly full. In the C
reproducer, 50 submissions of 0.44 s queued at once, while with 90 the
submitting thread was held for 10.6 s and the work past 20 s was cut;
255 submissions of 0.11 s held it for 23 s and all passed, with about 5 s
of work still queued at the end. That bounds the queue at a few dozen
submissions, whatever they cost: the sort's passes of ~0.15 s stayed near
10 s queued, but slower submissions do not. Twenty float32 4096 x 4096
matmuls (1.45 s each) queued before one synchronize were cut at 20.0 s on
main, with the last 7 outputs never written.

## What the stream does now (`vkcore/src/stream.rs`)

On Intel GPUs under Linux only (vendor 0x8086, `target_os = "linux"`):

- **At most 2 submissions in flight** (`TORCH_VK_MAX_IN_FLIGHT`, 0 for no
  limit, the default elsewhere). Before submitting a batch, the stream waits
  until at most one earlier batch is unfinished. A batch then waits behind
  at most one other, so it finishes within two batches' time of its
  submission, and with batches of at most 2^13 workgroups that is normally
  well under a second. The GPU still has the next batch queued whenever one
  finishes. One in flight would leave the GPU idle while the host wakes up
  between batches: on the HD 530 a run of multiplies over 2^20 elements (two
  per batch) took 1.5 to 1.7 ms each instead of 0.86.
- **Cancellations are reported.** Each submission signals an exportable
  fence. When the stream retires a submission that was in flight for 1 s or
  more (i915 cancels only at 20 s), it exports the fence as a sync_file and
  reads its status. A negative status fails the call with an error naming
  the cause, and every later call fails too, as on a lost device: the
  cancelled work's outputs, and whatever was computed from them, are wrong.
  Shorter submissions only have their fence reset. The device needs
  `VK_KHR_external_fence_fd` with sync_file export, which ANV has; without
  it the stream only bounds the work in flight.

The 2^13-workgroup batch cap stays: it keeps each batch short, and with two
in flight that is what bounds the queue in time.

## Cost (HD 530)

Measured against main on the HD 530, which other jobs were sharing (best
of several runs; differences of a few percent are noise there):

- Throughput is unchanged: 2000 multiplies of 2^20 float32 elements, 865
  us each on both; 500 of 2^22, 3409 us (main 3414); 20000 tiny adds, 5.4
  us (5.1); 1024 x 1024 matmuls, 22.2 ms (22.0). The twenty 4096 x 4096
  matmuls of the test take 29.2 s, twenty times one.
- An op followed by a synchronize: 134 us against 128 (add, then
  `torch_vulkan.synchronize()`), and 131 to 144 us against 131 to 140 in
  `dispatch_bench`'s fills, about what main's numbers vary by. Reading
  every submission's sync_file status cost about 12 us here, which is why
  only submissions in flight for 1 s or more are checked: a fence's wait
  and reset cost 1 to 3 us.
- One submission per op with no synchronize (`dispatch_bench` with a batch
  of 1, the worst case for the bound): 26 to 31 us per op against 22 to 24
  without the bound.

Elsewhere nothing runs: the bound is 0, no fences are created, and the
extra work per submission is one `Instant::now()`.

## Tests

`tests/python/test_long_submissions.py`, opt-in (`TORCH_VK_LONG_TESTS=1`;
about 4 minutes on the HD 530): twenty queued 4096 x 4096 matmuls (wrong on
main, right now), the float64 sort of 2^27 entries, two float32 sorts of 2^27
queued together, and, on Intel under Linux, a float64 sort in one 26 s
submission (`TORCH_VK_MAX_BATCH_GROUPS=0`), which must now raise instead of
returning a half-sorted result (on main it surfaced only as an `IndexError`
from a later kernel's range check). `cargo test -p vkcore` checks the bound
on submissions in flight on every GPU.

## Limits

- A single kernel running 20 s is still cut off; the stream can only
  report it. Raising `i915.request_timeout_ms` (as Intel's compute runtime
  documents for OpenCL and Level Zero) is the way around that.
- Workgroup counts are not time. Batches of a few heavy kernels can still
  take 10 s each: six queued float64 4096 x 4096 matmuls (5.2 s each, two
  per batch) lost the device after 31 s with two batches in flight, and
  passed with `TORCH_VK_MAX_IN_FLIGHT=1` (32 s). Bounding batches by
  measured time (timestamps per kernel) would cover that.
- Only i915 was tried; the xe driver (newer Intel GPUs) has its own job
  timeouts. The bound and the check apply to any Intel GPU under Linux.
