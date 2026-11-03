"""Random number generation on "vk" (M11).

The numbers differ from the CPU's (a different generator), so these tests
check reproducibility, ranges and distribution statistics instead. With
N = 2^20 samples, sample means are within ~5e-3 sigma of the truth, so the
tolerances below fail only for real errors.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import torch_vulkan as tv
from _util import WIDE, check, on_gpu

N = 1 << 20


def test_seed_reproducibility_and_generators():
    with on_gpu():
        torch.manual_seed(123)
        a = torch.rand(1000, device="vk")
        b = torch.rand(1000, device="vk")
        torch.manual_seed(123)
        a2 = torch.rand(1000, device="vk")
        g = torch.Generator(device="vk").manual_seed(7)
        c = torch.randn(1000, device="vk", generator=g)
        g.manual_seed(7)
        c2 = torch.randn(1000, device="vk", generator=g)
    assert torch.equal(a.cpu(), a2.cpu())
    assert not torch.equal(a.cpu(), b.cpu())
    assert torch.equal(c.cpu(), c2.cpu())
    assert tv.initial_seed() == 123


def test_rng_state_roundtrip():
    torch.manual_seed(5)
    torch.rand(10, device="vk")
    state = torch.vk.get_rng_state()
    with on_gpu():
        x = torch.randn(100, device="vk")
    torch.vk.set_rng_state(state)
    with on_gpu():
        y = torch.randn(100, device="vk")
    assert torch.equal(x.cpu(), y.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_uniform_and_normal(dtype):
    torch.manual_seed(0)
    with on_gpu():
        u = torch.empty(N, device="vk", dtype=dtype).uniform_(-2, 3)
        z = torch.empty(N, device="vk", dtype=dtype).normal_(1.5, 2.0)
    u, z = u.cpu().double(), z.cpu().double()
    assert u.min() >= -2 and u.max() <= 3
    assert abs(u.mean() - 0.5) < 0.01 and abs(u.var() - 25 / 12) < 0.02
    assert abs(z.mean() - 1.5) < 0.01 and abs(z.std() - 2.0) < 0.01
    within = ((z - 1.5).abs() < 2.0).double().mean()  # P(|Z| < 1) = 0.6827
    assert abs(within - 0.6827) < 0.003


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_other_distributions(dtype):
    torch.manual_seed(1)
    with on_gpu():
        e = torch.empty(N, device="vk", dtype=dtype).exponential_(2.0)
        g = torch.empty(N, device="vk", dtype=dtype).geometric_(0.25)
        c = torch.empty(N, device="vk", dtype=dtype).cauchy_(1.0, 0.5)
        ln = torch.empty(N, device="vk", dtype=dtype).log_normal_(0.3, 0.4)
    e, g, c, ln = (t.cpu().double() for t in (e, g, c, ln))
    assert e.min() >= 0 and abs(e.mean() - 0.5) < 0.005
    assert g.min() >= 1 and torch.equal(g, g.round()) and abs(g.mean() - 4.0) < 0.03
    assert abs(c.median() - 1.0) < 0.005
    assert abs(((c - 1).abs() < 0.5).double().mean() - 0.5) < 0.003  # quartiles at median +- sigma
    assert ln.min() > 0 and abs(ln.log().mean() - 0.3) < 0.003 and abs(ln.log().std() - 0.4) < 0.003


@pytest.mark.parametrize("dtype,lo,hi", [(torch.int8, -100, 100), (torch.uint8, 3, 250), (torch.int32, -5, 7),
                                         (torch.int64, -(1 << 40), 1 << 40), (torch.float32, 0, 1000),
                                         (torch.float64, -50, 50), (torch.bool, 0, 2)])
def test_random_integers(dtype, lo, hi):
    torch.manual_seed(2)
    with on_gpu():
        r = torch.randint(lo, hi, (N,), device="vk", dtype=dtype)
    r = r.cpu().double()
    assert r.min() >= lo and r.max() <= hi - 1 and torch.equal(r, r.round())
    assert abs(r.mean() - (lo + hi - 1) / 2) < 0.01 * (hi - lo)
    if hi - lo <= 256:
        assert torch.unique(r).numel() == hi - lo  # every value occurs


def test_random_full_range():
    torch.manual_seed(3)
    with on_gpu():
        a = torch.empty(N, device="vk", dtype=torch.int64).random_()
        b = torch.empty(N, device="vk", dtype=torch.float32).random_()
        c = torch.empty(N, device="vk", dtype=torch.int16).random_(-7, None)  # [-7, max]
    a, b, c = a.cpu(), b.cpu(), c.cpu()
    assert a.min() >= 0 and a.max() > (1 << 62)  # [0, 2^63 - 1]
    assert b.min() >= 0 and b.max() <= (1 << 24) and b.max() > (1 << 23)
    assert c.min() >= -7 and c.max() > 30000


def test_bernoulli_and_dropout():
    torch.manual_seed(4)
    p = torch.linspace(0, 1, 9).repeat(N // 8)[:N]
    with on_gpu():
        b = torch.bernoulli(p.to("vk"))
        b2 = torch.empty(N, device="vk", dtype=torch.bool).bernoulli_(0.3)
        x = torch.ones(N, device="vk")
        d = F.dropout(x, 0.25, training=True)
    b, b2, d = b.cpu(), b2.cpu(), d.cpu()
    for k in range(9):
        assert abs(b[k::9].mean() - k / 8) < 0.01
    assert abs(b2.double().mean() - 0.3) < 0.003
    assert set(d.unique().tolist()) <= {0.0, torch.tensor(4 / 3).item()} and abs((d == 0).double().mean() - 0.25) < 0.003


def test_multinomial_and_randperm():
    torch.manual_seed(6)
    probs = torch.tensor([[0.1, 0.0, 0.6, 0.3], [1.0, 1.0, 1.0, 1.0]])
    with on_gpu():
        s = torch.multinomial(probs.to("vk"), 200_000, replacement=True)
        w = torch.multinomial(probs.to("vk"), 3, replacement=False)
        many = torch.multinomial(torch.ones(1000).to("vk"), 1000)
        perm = torch.randperm(5000, device="vk")
    s, w, many, perm = s.cpu(), w.cpu(), many.cpu(), perm.cpu()
    freq = torch.stack([torch.bincount(row, minlength=4) for row in s]).double() / s.size(1)
    assert (freq - probs / probs.sum(1, keepdim=True)).abs().max() < 0.005
    assert all(len(set(row.tolist())) == 3 and 1 not in row.tolist() for row in w[:1])
    assert torch.equal(many.sort().values, torch.arange(1000))
    assert torch.equal(perm.sort().values, torch.arange(5000)) and not torch.equal(perm, torch.arange(5000))


def test_complex_fills():
    torch.manual_seed(9)
    with on_gpu():
        z = torch.randn(N, device="vk", dtype=torch.complex64)
        u = torch.rand(N, device="vk", dtype=torch.complex64)
    z, u = z.cpu(), u.cpu()
    assert abs(z.real.var() - 0.5) < 0.005 and abs(z.imag.var() - 0.5) < 0.005  # E|z|^2 = 1
    assert u.real.min() >= 0 and u.imag.max() < 1 and abs(u.imag.mean() - 0.5) < 0.005


def test_normal_tensor_parameters():
    torch.manual_seed(8)
    mean = torch.tensor([0.0, 10.0, -5.0]).repeat(N // 4, 1)
    std = torch.tensor([1.0, 0.1, 3.0])
    with on_gpu():
        z = torch.normal(mean.to("vk"), std.to("vk"))
    z = z.cpu().double()
    assert (z.mean(0) - torch.tensor([0.0, 10.0, -5.0], dtype=torch.float64)).abs().max() < 0.02
    assert (z.std(0) - std.double()).abs().max() < 0.02


def _ks(a, b):
    """Kolmogorov-Smirnov distance of two samples."""
    a, b = a.double().sort().values, b.double().sort().values
    v = torch.cat([a, b]).sort().values
    return (torch.searchsorted(a, v, right=True) / len(a) - torch.searchsorted(b, v, right=True) / len(b)).abs().max().item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gamma_poisson_binomial_dirichlet(dtype):
    # Against the CPU's samples of the same distributions (2e5 each: a KS
    # distance above 0.01 is a real difference).
    torch.manual_seed(9)
    n = 200000
    full = lambda v: torch.full((n,), v, dtype=dtype)
    cases = [(f"gamma {a}", lambda d, a=a: torch._standard_gamma(full(a).to(d))) for a in [1e-3, 0.5, 1.0, 10.0, 1e3]]
    cases += [(f"poisson {r}", lambda d, r=r: torch.poisson(full(r).to(d))) for r in [0.3, 9.99, 10.0, 1e4]]
    cases += [(f"binomial {c} {p}", lambda d, c=c, p=p: torch.binomial(full(c).to(d), full(p).to(d)))
              for c, p in [(10, 0.3), (100, 0.05), (1000, 0.97), (1e6, 0.4)]]
    alpha = torch.tensor([0.01, 0.5, 7.0], dtype=dtype).expand(n // 4, -1).contiguous()
    for k in range(3):
        cases.append((f"dirichlet {k}", lambda d, k=k: torch._sample_dirichlet(alpha.to(d))[:, k]))
    for name, f in cases:
        with on_gpu():
            got = f("vk")
        assert torch.isfinite(got).all(), name
        assert _ks(got.cpu(), f("cpu")) < 0.01, name
    with on_gpu():
        edge = [torch._standard_gamma(torch.tensor([0.0, 1e-30, float("inf")], dtype=dtype, device="vk")),
                torch.binomial(torch.tensor([5.0, -1.0, 5.0, 0.0], dtype=dtype, device="vk"),
                               torch.tensor([0.5, 0.5, 1.5, -1.0], dtype=dtype, device="vk")),
                torch._sample_dirichlet(torch.tensor([[1e-30, 1e-30], [0.0, 0.0]], dtype=dtype, device="vk"))]
    tiny = torch.finfo(dtype).tiny
    check(edge[0], torch.tensor([tiny, tiny, float("inf")], dtype=dtype))
    assert edge[1].cpu()[1:].tolist() == [0.0, 5.0, 0.0]
    check(edge[2], torch.full((2, 2), 0.5, dtype=dtype))
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.poisson(torch.tensor([-1.0], dtype=dtype, device="vk"))


def test_gamma_and_dirichlet_gradients():
    torch.manual_seed(10)
    alpha = torch.cat([torch.rand(3000) * 0.9 + 0.05, torch.rand(3000) * 10 + 0.5, torch.rand(3000) * 100 + 8]).to(WIDE)
    x = torch._standard_gamma(alpha)
    with on_gpu():
        got = torch._standard_gamma_grad(alpha.to("vk"), x.to("vk"))
    check(got, torch._standard_gamma_grad(alpha, x), rtol=1e-3, atol=1e-6)
    a = torch.cat([torch.rand(3000) * 0.9 + 0.05, torch.rand(3000) * 10 + 0.5, torch.rand(3000) * 50 + 7]).to(WIDE)
    total = a + torch.cat([torch.rand(3000) * 0.9 + 0.05, torch.rand(3000) * 10 + 0.5, torch.rand(3000) * 50 + 7]).to(WIDE)
    xs = torch.distributions.Beta(a, total - a).sample().clamp(1e-4, 1 - 1e-4)
    with on_gpu():
        got = torch._dirichlet_grad(xs.to("vk"), a.to("vk"), total.to("vk"))
    # (float32 without float64: its rare cancellation near the Beta mean costs ~1%)
    check(got, torch._dirichlet_grad(xs, a, total), rtol=2e-2 if WIDE == torch.float32 else 1e-6, atol=1e-5)
    conc = torch.full((5000,), 2.0, device="vk", requires_grad=True)
    with on_gpu():
        torch.distributions.Gamma(conc, 1.0).rsample().sum().backward()
        torch.distributions.Dirichlet(torch.tensor([1.0, 2.0, 3.0], device="vk", requires_grad=True)).rsample((100,)).sum().backward()
    assert torch.isfinite(conc.grad).all()
