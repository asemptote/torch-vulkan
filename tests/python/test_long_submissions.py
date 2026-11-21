"""More than 20 s of GPU work queued at once (issue #72).

Under Linux, i915 cancels GPU work still unfinished 20 s after it was
submitted, and Mesa's ANV reports success anyway (mesa#16166), so on Intel
GPUs the stream keeps little work in flight (TORCH_VK_MAX_IN_FLIGHT) and
checks each submission's fence status. These tests queue that much work on
purpose: about 30 s each on an HD 530, minutes on a slow GPU, so they are
opt-in:

    TORCH_VK_LONG_TESTS=1 python -m pytest -v tests/python/test_long_submissions.py

Each case runs in its own process (a cancellation, once detected, fails every
later call, as a lost device would).
"""

import os
import subprocess
import sys
import textwrap

import pytest

from torch_vulkan import _native

pytestmark = pytest.mark.skipif(
    os.environ.get("TORCH_VK_LONG_TESTS", "") in ("", "0"),
    reason="long (>20 s of GPU work each): set TORCH_VK_LONG_TESTS=1",
)


def run(code, **env):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    print(proc.stdout + proc.stderr)  # (shown with -s)
    return proc.returncode, proc.stdout + proc.stderr


def test_many_queued_matmuls():
    # About 1.5 s each on the HD 530: 20 of them, queued without a host wait,
    # are 30 s of work. With nothing bounding the submissions in flight, the
    # ones still unfinished at 20 s were never written (left NaN).
    code, out = run("""
        import time, torch, torch_vulkan
        n, k = 4096, 20
        a = torch.rand(n, n, device="vk")
        b = torch.rand(n, n, device="vk")
        ref = a @ b
        outs = [torch.full((n, n), float("nan"), device="vk") for _ in range(k)]
        torch_vulkan.synchronize()
        t = time.time()
        for y in outs:
            torch.mm(a, b, out=y)
        torch_vulkan.synchronize()
        print(f"{k} matmuls: {time.time() - t:.1f} s", flush=True)
        bad = [i for i, y in enumerate(outs) if not torch.equal(y, ref)]
        print("wrong:", bad)
        assert not bad
    """)
    assert code == 0, out


@pytest.mark.float64
def test_float64_sort_2_27():
    # About 26 s of GPU work in 172 dispatches; in one submission it came back
    # unsorted on the HD 530.
    code, out = run("""
        import time, torch, torch_vulkan
        torch.manual_seed(3)
        x = torch.rand(1 << 27, dtype=torch.float64)
        xv = x.to("vk")
        torch_vulkan.synchronize()
        t = time.time()
        v = torch.sort(xv).values.cpu()
        print(f"float64 sort of 2^27: {time.time() - t:.1f} s", flush=True)
        assert torch.equal(v, torch.sort(x).values)
    """)
    assert code == 0, out


def test_two_queued_sorts():
    # Two 2^27-entry float32 sorts queued before one synchronize (~38 s).
    code, out = run("""
        import time, torch, torch_vulkan
        torch.manual_seed(2)
        xs = [torch.rand(1 << 27) for _ in range(2)]
        xv = [x.to("vk") for x in xs]
        torch_vulkan.synchronize()
        t = time.time()
        outs = [torch.sort(x).values for x in xv]
        torch_vulkan.synchronize()
        print(f"two float32 sorts of 2^27: {time.time() - t:.1f} s", flush=True)
        for x, v in zip(xs, outs):
            assert torch.equal(v.cpu(), torch.sort(x).values)
    """)
    assert code == 0, out


@pytest.mark.float64
@pytest.mark.skipif(
    not (sys.platform == "linux" and "Intel" in _native.device_name()),
    reason="i915's 20 s request watchdog: Intel GPUs under Linux",
)
def test_cancellation_is_reported():
    # One 26 s submission (no batch limit) cannot be saved by bounding the
    # submissions in flight: i915 cuts it at 20 s. The stream must then raise
    # rather than return the half-sorted result.
    code, out = run(
        """
        import torch, torch_vulkan
        x = torch.rand(1 << 27, dtype=torch.float64)
        try:
            v = torch.sort(x.to("vk")).values.cpu()
        except RuntimeError as e:
            assert "cancelled by the kernel driver" in str(e), e
            print("raised:", e)
        else:
            raise AssertionError("no error; sorted: %s" % torch.equal(v, torch.sort(x).values))
        """,
        TORCH_VK_MAX_BATCH_GROUPS="0",
    )
    assert code == 0, out
