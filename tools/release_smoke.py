"""A release's smoke test, from its own artefacts (a wheel in a fresh venv,
or a Termux build from the release's SPIR-V bundle), with TORCH_VK_STRICT=1
so that any CPU fallback raises:

    TORCH_VK_STRICT=1 python tools/release_smoke.py

matmul, fft / rfft / irfft, eigvalsh, solve, qr (and float64 qr where the
GPU has float64), 20 Adam steps on a small MLP, and no fallbacks. It ends
with "SMOKE OK".
"""

import importlib.metadata
import os

import torch
import torch_vulkan as tv

assert os.environ.get("TORCH_VK_STRICT") == "1", "run with TORCH_VK_STRICT=1"
try:
    version = importlib.metadata.version("torch_vulkan")
except importlib.metadata.PackageNotFoundError:  # (a source tree on PYTHONPATH)
    version = getattr(tv, "__version__", "unknown (not installed)")
print("torch_vulkan", version, "from", os.path.dirname(tv.__file__), "torch", torch.__version__)
print("device", tv.device_name())
f64 = bool(tv._native.capabilities() & 1)
torch.manual_seed(0)


def close(got, want, tol):
    err = ((got.cpu() - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
    assert err < tol, err
    return err


a, b = torch.randn(256, 256), torch.randn(256, 256)
print("matmul", close(a.to("vk") @ b.to("vk"), a @ b, 1e-4))

x = torch.randn(4, 1 << 12, dtype=torch.complex64)
print("fft", close(torch.fft.fft(x.to("vk")), torch.fft.fft(x), 1e-5))
r = torch.randn(3, 1 << 15)
print("rfft", close(torch.fft.rfft(r.to("vk")), torch.fft.rfft(r), 1e-5))
print("irfft", close(torch.fft.irfft(torch.fft.rfft(r).to("vk")), r, 1e-5))

s = torch.randn(64, 64)
s = s @ s.T + 64 * torch.eye(64)
print("eigvalsh", close(torch.linalg.eigvalsh(s.to("vk")), torch.linalg.eigvalsh(s), 1e-4))
rhs = torch.randn(64, 3)
print("solve", close(torch.linalg.solve(s.to("vk"), rhs.to("vk")), torch.linalg.solve(s, rhs), 1e-4))

m = torch.randn(96, 48)
q, rr = torch.linalg.qr(m.to("vk"))
print("qr", close(q @ rr, m, 1e-4), close(q.T @ q, torch.eye(48), 1e-4))
if f64:
    md = m.double()
    q, rr = torch.linalg.qr(md.to("vk"))
    print("qr f64", close(q @ rr, md, 1e-12))

model = torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)).to("vk")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
xb, yb = torch.randn(128, 32, device="vk"), torch.randint(0, 10, (128,), device="vk")
losses = []
for _ in range(20):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    opt.step()
    losses.append(loss.item())
assert losses[-1] < losses[0], losses
print(f"training: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

fb = tv.fallback_counts()
print("fallback_counts", fb)
assert not fb, fb
print("SMOKE OK")
