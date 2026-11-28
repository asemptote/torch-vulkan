"""Back-pressure: the host does not run unboundedly ahead of the GPU (#100,
docs/log/M59).

The stream estimates each kernel's GPU time from timestamp queries and waits
before a submission while the estimated time queued would pass
TORCH_VK_MAX_QUEUED_MS, or the submissions in flight TORCH_VK_MAX_IN_FLIGHT.
Each case runs in its own process, with its own bounds.
"""
import os
import subprocess
import sys
import textwrap

import pytest

from torch_vulkan import _native


def stats():
    return dict(_native.queue_stats())


def run(code, **env):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_queue_stats():
    s = stats()
    for k in ("submissions", "waits", "wait_s", "timed", "max_queued_s", "max_in_flight", "queued_s",
              "limit_queued_s", "limit_in_flight", "measure_first"):
        assert k in s, k
    assert s["limit_queued_s"] >= 0 and s["limit_in_flight"] >= 0


# Matrix products sized to take about 2 ms each on this device, queued
# without a synchronize for about 0.5 s of GPU time.
LOOP = """
    import time, torch, torch_vulkan
    from torch_vulkan import _native
    n = 128
    while True:
        a = torch.randn(n, n, device="vk")
        b = torch.randn(n, n, device="vk")
        out = torch.empty(n, n, device="vk")
        torch.mm(a, b, out=out)
        torch_vulkan.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            torch.mm(a, b, out=out)
        torch_vulkan.synchronize()
        one = (time.perf_counter() - t) / 5
        if one > 2e-3 or n >= 4096:
            break
        n *= 2
    ref = (a.cpu().double() @ b.cpu().double()).float()
    k = max(10, int(0.5 / one))
    outs = [torch.empty(n, n, device="vk") for _ in range(4)]
    for i in range(k):
        torch.mm(a, b, out=outs[i % 4])
    queued = dict(_native.queue_stats())["queued_s"]
    t = time.perf_counter()
    torch_vulkan.synchronize()
    sync = time.perf_counter() - t
    s = dict(_native.queue_stats())
    for o in outs:
        torch.testing.assert_close(o.cpu(), ref, rtol=1e-3, atol=1e-3)
    print(f"RESULT n={n} k={k} one={one:.6f} sync={sync:.4f} queued={queued:.4f}", s)
"""


def parse(out):
    line = next(l for l in out.splitlines() if l.startswith("RESULT"))
    head, _, tail = line.partition(" {")
    fields = dict(f.split("=") for f in head.split()[1:])
    return {k: float(v) for k, v in fields.items()}, eval("{" + tail)


def test_queued_time_is_bounded():
    if stats()["limit_queued_s"] == 0 or not _timestamps():
        pytest.skip("no timestamp queries on this device's queue")
    r, s = parse(run(LOOP, TORCH_VK_MAX_QUEUED_MS="50", TORCH_VK_MAX_IN_FLIGHT="0"))
    assert s["limit_queued_s"] == pytest.approx(0.05)
    assert s["timed"] > 0
    # The estimate queued after any submission stays within the bound, or
    # is one batch alone (a batch is cut at a quarter of the bound, so one
    # product of about 2 ms is a batch of its own only past 12.5 ms).
    assert s["max_queued_s"] <= max(0.05, 2 * r["one"]) * 1.5, (r, s)
    assert r["queued"] <= max(0.05, 2 * r["one"]) * 1.5, (r, s)


def test_unbounded_switches():
    # TORCH_VK_MAX_QUEUED_MS=0 and TORCH_VK_MAX_IN_FLIGHT=0: no bounds, no
    # kernels timed, as before (developer switches).
    r, s = parse(run(LOOP, TORCH_VK_MAX_QUEUED_MS="0", TORCH_VK_MAX_IN_FLIGHT="0"))
    assert s["limit_queued_s"] == 0 and s["limit_in_flight"] == 0
    assert s["timed"] == 0 and s["waits"] == 0 and s["measure_first"] == 0


def test_submissions_in_flight_bounded():
    r, s = parse(run(LOOP, TORCH_VK_MAX_QUEUED_MS="0", TORCH_VK_MAX_IN_FLIGHT="3"))
    assert s["limit_in_flight"] == 3
    assert s["max_in_flight"] <= 3


def test_measure_first():
    # Intel's default under Linux (i915's watchdog, docs/log/M34), forced
    # here: still the right results, with at most one batch of unknown cost
    # in flight (checked in vkcore's tests).
    if not _timestamps():
        pytest.skip("no timestamp queries on this device's queue")
    r, s = parse(run(LOOP, TORCH_VK_MEASURE_FIRST="1"))
    assert s["measure_first"] == 1


def _timestamps():
    # The stream times kernels only where the queue has timestamps; after the
    # session's earlier tests, some kernel has been timed if it does.
    import torch
    import torch_vulkan

    x = torch.ones(1 << 12, device="vk")
    for _ in range(4):
        x.add_(1)
        torch_vulkan.synchronize()
    return stats()["timed"] > 0
