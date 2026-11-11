"""The same workloads on device "vk" and on dpnp (Intel's SYCL NumPy, backed by
oneMKL), for comparing against vendor libraries on Intel GPUs.

    python benchmarks/vs_dpnp.py           # device "vk"
    python benchmarks/vs_dpnp.py --dpnp    # dpnp, from an environment that has it

Each result is checked against NumPy on the CPU (relative error, 2-norm).
"""

import argparse
import time

import numpy as np


def backend(use_dpnp):
    if use_dpnp:
        import dpctl
        import dpnp

        q = dpctl.SyclQueue("gpu")
        return dict(
            name=f"dpnp {dpnp.__version__}: {q.sycl_device.name}",
            to=lambda a: dpnp.asarray(a, sycl_queue=q),
            back=dpnp.asnumpy,
            sync=q.wait,
            exp=dpnp.exp,
            sin=dpnp.sin,
            sort=dpnp.sort,
            fft=dpnp.fft.fft,
            solve=dpnp.linalg.solve,
            eigvalsh=dpnp.linalg.eigvalsh,
            eigvals=dpnp.linalg.eigvals,
            fallbacks=lambda: None,
        )
    import torch

    import torch_vulkan as tv

    return dict(
        name=f"vk, torch {torch.__version__}: {tv.device_name()}",
        to=lambda a: torch.from_numpy(a).to("vk"),
        back=lambda t: t.cpu().numpy(),
        sync=tv.synchronize,
        exp=torch.exp,
        sin=torch.sin,
        sort=lambda a: torch.sort(a).values,
        fft=torch.fft.fft,
        solve=torch.linalg.solve,
        eigvalsh=torch.linalg.eigvalsh,
        eigvals=torch.linalg.eigvals,
        fallbacks=lambda: tv.fallback_counts() or "none",
    )


def timed(fn, sync, reps=3):
    result = fn()
    sync()
    start = time.perf_counter()
    for _ in range(reps):
        result = fn()
    sync()
    return (time.perf_counter() - start) / reps, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpnp", action="store_true", help="run on dpnp instead of device vk")
    xp = backend(parser.parse_args().dpnp)
    print(xp["name"])

    rng = np.random.default_rng(0)
    x32 = rng.standard_normal(4_000_000).astype(np.float32)
    x64 = x32.astype(np.float64)
    s = rng.standard_normal(1 << 20).astype(np.complex64)
    a32 = (rng.standard_normal((1024, 1024)) + 1024 * np.eye(1024)).astype(np.float32)
    b32 = rng.standard_normal(1024).astype(np.float32)
    a64, b64 = a32.astype(np.float64), b32.astype(np.float64)
    m32 = rng.standard_normal((2048, 2048)).astype(np.float32)
    m64 = m32.astype(np.float64)
    sym = rng.standard_normal((512, 512))
    sym = sym + sym.T
    gen = rng.standard_normal((256, 256))

    to = xp["to"]
    d = {k: to(v) for k, v in dict(x32=x32, x64=x64, s=s, a32=a32, b32=b32, a64=a64, b64=b64,
                                   m32=m32, m64=m64, sym=sym, gen=gen).items()}
    cases = [
        ("exp*sin f32 4M", lambda: xp["exp"](d["x32"]) * xp["sin"](d["x32"]), lambda: np.exp(x32) * np.sin(x32)),
        ("exp*sin f64 4M", lambda: xp["exp"](d["x64"]) * xp["sin"](d["x64"]), lambda: np.exp(x64) * np.sin(x64)),
        ("sum f32 4M", lambda: d["x32"].sum(), lambda: x32.sum(dtype=np.float64)),
        ("sort f32 4M", lambda: xp["sort"](d["x32"]), lambda: np.sort(x32)),
        ("fft c64 1M", lambda: xp["fft"](d["s"]), lambda: np.fft.fft(s)),
        ("solve f32 1024", lambda: xp["solve"](d["a32"], d["b32"]), lambda: np.linalg.solve(a32, b32)),
        ("solve f64 1024", lambda: xp["solve"](d["a64"], d["b64"]), lambda: np.linalg.solve(a64, b64)),
        ("eigvalsh f64 512", lambda: xp["eigvalsh"](d["sym"]), lambda: np.linalg.eigvalsh(sym)),
        ("eigvals f64 256", lambda: xp["eigvals"](d["gen"]), None),
        ("matmul f32 2048", lambda: d["m32"] @ d["m32"], lambda: m32 @ m32),
        ("matmul f64 2048", lambda: d["m64"] @ d["m64"], lambda: m64 @ m64),
    ]
    for label, fn, ref in cases:
        seconds, result = timed(fn, xp["sync"])
        line = f"  {label:18s} {seconds * 1000:10.1f} ms"
        if label.startswith("matmul"):
            line += f"  {2 * 2048**3 / seconds / 1e9:6.1f} GFLOP/s"
        if ref is not None:
            got = np.asarray(xp["back"](result), dtype=np.complex128).ravel()
            want = np.asarray(ref(), dtype=np.complex128).ravel()
            line += f"  rel. error {np.linalg.norm(got - want) / np.linalg.norm(want):.1e}"
        print(line)

    times = []
    for _ in range(20):
        start = time.perf_counter()
        float((to(np.arange(10, dtype=np.float32)) + 3).sum())
        times.append(time.perf_counter() - start)
    print(f"  small op round trip (upload, add, sum, read back): median {np.median(times) * 1000:.2f} ms")
    if xp["fallbacks"]() is not None:
        print(f"  CPU fallbacks: {xp['fallbacks']()}")


if __name__ == "__main__":
    main()
