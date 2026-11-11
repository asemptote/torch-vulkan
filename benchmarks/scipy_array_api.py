"""SciPy on device "vk" tensors through SciPy's array API support: which
functions run on the GPU, and whether they match SciPy on NumPy arrays.

    pip install scipy
    python benchmarks/scipy_array_api.py

SCIPY_ARRAY_API=1 is set here, before SciPy is imported. SciPy hands some
functions to PyTorch (special functions to torch.special, FFTs to torch.fft);
functions SciPy marks CPU-only raise for tensors on any other device.
"""

import os

os.environ["SCIPY_ARRAY_API"] = "1"

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.fft  # noqa: E402
import scipy.integrate  # noqa: E402
import scipy.linalg  # noqa: E402
import scipy.ndimage  # noqa: E402
import scipy.signal  # noqa: E402
import scipy.special  # noqa: E402
import scipy.stats  # noqa: E402
import torch  # noqa: E402

import torch_vulkan as tv  # noqa: E402


def main():
    print(f"SciPy {scipy.__version__}, torch {torch.__version__}: {tv.device_name()}")
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 3.0, 1000)
    m = rng.standard_normal((64, 64)) + 64 * np.eye(64)
    img = rng.standard_normal((128, 128))
    cases = [
        ("special.erf", lambda x, m, i: scipy.special.erf(x)),
        ("special.gammaln", lambda x, m, i: scipy.special.gammaln(x)),
        ("special.logsumexp", lambda x, m, i: scipy.special.logsumexp(x)),
        ("fft.fft", lambda x, m, i: scipy.fft.fft(x)),
        ("linalg.solve", lambda x, m, i: scipy.linalg.solve(m, x[:64])),
        ("linalg.inv", lambda x, m, i: scipy.linalg.inv(m)),
        ("stats.zscore", lambda x, m, i: scipy.stats.zscore(x)),
        ("stats.skew", lambda x, m, i: scipy.stats.skew(x)),
        ("stats.ttest_1samp", lambda x, m, i: scipy.stats.ttest_1samp(x, 1.0).statistic),
        ("signal.convolve", lambda x, m, i: scipy.signal.convolve(x, x[:16])),
        ("signal.welch", lambda x, m, i: scipy.signal.welch(x)[1]),
        ("ndimage.gaussian_filter", lambda x, m, i: scipy.ndimage.gaussian_filter(i, 2.0)),
        ("integrate.trapezoid", lambda x, m, i: scipy.integrate.trapezoid(x)),
    ]
    vx, vm, vi = (torch.from_numpy(a).to("vk") for a in (x, m, img))
    before = dict(tv.fallback_counts())
    on_gpu = 0
    for name, fn in cases:
        try:
            got = fn(vx, vm, vi)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:24s} {type(e).__name__}: {str(e).splitlines()[0][:70]}")
            continue
        want = fn(x, m, img)
        device = str(getattr(got, "device", "cpu"))
        on_gpu += device.startswith("vk")
        got = got.cpu().numpy() if torch.is_tensor(got) else np.asarray(got)
        print(f"  {name:24s} on {device:5s} max |difference| {np.max(np.abs(got - np.asarray(want))):.1e}")
    after = tv.fallback_counts()
    fallbacks = {k: n - before.get(k, 0) for k, n in after.items() if n != before.get(k, 0)}
    print(f"  {on_gpu}/{len(cases)} on the GPU; CPU fallbacks during these calls: {fallbacks or 'none'}")


if __name__ == "__main__":
    main()
