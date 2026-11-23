"""scaled_dot_product_attention on "vk" (M54): the fused kernel
(shaders/attention.slang, the "overrideable" backend) and the math path,
forward and backward, against the math path on the CPU in float64."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from _util import HAS_FLOAT64, on_gpu

import torch_vulkan as tv

FUSED = int(SDPBackend.OVERRIDEABLE)
MATH = int(SDPBackend.MATH)
INT16 = bool(tv._native.capabilities() & 2)
DTYPES = [torch.float32] + ([torch.bfloat16, torch.float16] if INT16 else [])
TOL = {
    torch.float32: dict(rtol=2e-5, atol=2e-5),
    torch.float16: dict(rtol=2e-3, atol=2e-3),
    torch.bfloat16: dict(rtol=1.6e-2, atol=1.6e-2),
}
# gradients of 16-bit inputs go through their rounded outputs (delta = rowsum(dO out))
GRAD_TOL = {
    torch.float32: dict(rtol=1e-4, atol=1e-4),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=5e-2, atol=5e-2),
}


def rand(shape, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def reference(q, k, v, mask=None, causal=False, scale=None, gqa=False, dropout_p=0.0, dropout_mask=None):
    """The math path on the CPU in float64 (a bool mask made additive first, as F.sdpa does)."""
    args = [x.double() for x in (q, k, v)]
    if mask is not None:
        mask = torch.where(mask, 0.0, -torch.inf) if mask.dtype == torch.bool else mask.double()
    return torch.ops.aten._scaled_dot_product_attention_math(
        *args, mask, dropout_p, causal, dropout_mask, scale=scale, enable_gqa=gqa)[0]


def run(backend, q, k, v, mask=None, causal=False, scale=None, gqa=False):
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale,
                                              enable_gqa=gqa)


def with_grads(fn, q, k, v, seed=7):
    """fn's output and the gradients of sum(out * r) for q, k, v (r fixed)."""
    leaves = [x.detach().requires_grad_() for x in (q, k, v)]
    out = fn(*leaves)
    r = rand(out.shape, seed).to(out.device, out.dtype)
    (out * r).sum().backward()
    return [out.detach()] + [x.grad for x in leaves]


# B, H, Hk, L, S, E, EV
SHAPES = [
    (2, 3, 3, 37, 37, 16, 16),
    (1, 2, 2, 64, 64, 64, 64),
    (2, 2, 2, 70, 130, 64, 32),
    (1, 1, 1, 1, 5, 8, 8),
    (1, 2, 2, 129, 3, 40, 80),
    (1, 2, 2, 50, 66, 128, 128),
    (1, 1, 1, 20, 33, 256, 200),
    (2, 4, 2, 45, 45, 32, 32),  # GQA
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_matches_math(dtype, shape):
    B, H, Hk, L, S, E, EV = shape
    q = rand((B, H, L, E), 1, dtype)
    k = rand((B, Hk, S, E), 2, dtype)
    v = rand((B, Hk, S, EV), 3, dtype)
    gqa = H != Hk
    qv, kv, vv = (x.to("vk") for x in (q, k, v))
    assert torch.ops.aten._fused_sdp_choice(qv, kv, vv, enable_gqa=gqa) == FUSED
    want = reference(q, k, v, gqa=gqa)
    with on_gpu():
        fused = run(SDPBackend.OVERRIDEABLE, qv, kv, vv, gqa=gqa)
        math = run(SDPBackend.MATH, qv, kv, vv, gqa=gqa)
    assert fused.dtype == dtype and fused.shape == (B, H, L, EV)
    torch.testing.assert_close(fused.cpu().double(), want, **TOL[dtype])
    torch.testing.assert_close(fused.cpu(), math.cpu(), **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[2], SHAPES[4], SHAPES[7]])
@pytest.mark.parametrize("causal", [False, True])
def test_fused_gradients(dtype, shape, causal):
    B, H, Hk, L, S, E, EV = shape
    q = rand((B, H, L, E), 1, dtype)
    k = rand((B, Hk, S, E), 2, dtype)
    v = rand((B, Hk, S, EV), 3, dtype)
    gqa = H != Hk
    want = with_grads(lambda a, b, c: reference(a, b, c, causal=causal, gqa=gqa), q, k, v)
    with on_gpu():
        got = with_grads(lambda a, b, c: run(SDPBackend.OVERRIDEABLE, a, b, c, causal=causal, gqa=gqa),
                         *(x.to("vk") for x in (q, k, v)))
    for g, w in zip(got, want):
        assert g.dtype == dtype
        torch.testing.assert_close(g.cpu().double(), w.double(), **GRAD_TOL[dtype])


def masks(B, H, L, S):
    g = torch.Generator().manual_seed(5)
    keep = torch.rand((B, 1, L, S), generator=g) > 0.3
    keep[0, 0, 3] = False  # a fully masked row: zeros, as the math path's _safe_softmax
    padding = torch.ones((B, 1, 1, S), dtype=torch.bool)
    padding[-1, ..., S // 2:] = False
    return {
        "bool": keep,
        "additive": torch.randn((B, H, L, S), generator=g),
        "padding": padding,  # broadcast along queries and heads
        "additive_2d": torch.randn((L, S), generator=g),
        "column": torch.randn((L, 1), generator=g),  # broadcast along keys (stride 0)
        "sliced": torch.randn((L + 7, S + 9), generator=g)[3:L + 3, 2:S + 2],  # row stride != S
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["bool", "additive", "padding", "additive_2d", "column", "sliced"])
def test_fused_masks(dtype, kind):
    B, H, L, S, E = 2, 2, 70, 75, 32
    q, k, v = rand((B, H, L, E), 1, dtype), rand((B, H, S, E), 2, dtype), rand((B, H, S, E), 3, dtype)
    mask = masks(B, H, L, S)[kind]
    if mask.dtype != torch.bool:
        mask = mask.to(dtype)
    want = with_grads(lambda a, b, c: reference(a, b, c, mask), q, k, v)
    with on_gpu():
        got = with_grads(lambda a, b, c: run(SDPBackend.OVERRIDEABLE, a, b, c, mask.to("vk")),
                         *(x.to("vk") for x in (q, k, v)))
    for g, w in zip(got, want):
        torch.testing.assert_close(g.cpu().double(), w.double(), **GRAD_TOL[dtype])
    if kind == "bool":
        assert (got[0][0, :, 3] == 0).all()


def test_fused_scale_and_layouts():
    # a negative scale; q, k, v as views of one qkv projection (ViT's layout),
    # a non-unit stride along the head dim; the output's layout [B, L, H, EV]
    B, L, H, E = 2, 50, 3, 24
    qkv = rand((B, L, 3, H, E), 4).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2].transpose(-1, -2).contiguous().transpose(-1, -2)
    want = with_grads(lambda a, b, c: reference(a, b, c, scale=-0.3), q, k, v)
    with on_gpu():
        got = with_grads(lambda a, b, c: run(SDPBackend.OVERRIDEABLE, a, b, c, scale=-0.3),
                         *(x.to("vk") for x in (q, k, v)))
    for g, w in zip(got, want):
        torch.testing.assert_close(g.cpu().double(), w.double(), **GRAD_TOL[torch.float32])
    assert got[0].transpose(1, 2).is_contiguous()


def test_fused_unbatched_and_broadcast_heads():
    # 3-d inputs, and one key/value head shared by every query head without enable_gqa
    q, k, v = rand((4, 9, 16), 1), rand((1, 11, 16), 2), rand((1, 11, 8), 3)
    want = reference(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
    with on_gpu():
        got = run(SDPBackend.OVERRIDEABLE, q.to("vk"), k.to("vk"), v.to("vk"))
    torch.testing.assert_close(got.cpu().double(), want, **TOL[torch.float32])


@pytest.mark.parametrize("causal", [False, True])
def test_fused_dropout(causal):
    # the kernel's dropout mask (return_debug_mask), against the math path
    # given that mask, forward and backward
    B, H, L, S, E, p = 2, 2, 70, 70, 32, 0.3
    q, k, v = rand((B, H, L, E), 1), rand((B, H, S, E), 2), rand((B, H, S, E), 3)
    qv, kv, vv = (x.to("vk") for x in (q, k, v))
    op = torch.ops.aten._scaled_dot_product_fused_attention_overrideable
    out, lse, cq, ck, mq, mk, seed, offset, keep = op(qv, kv, vv, None, p, causal, True)
    keep = keep.cpu()
    assert keep.shape == (B, H, L, S) and keep.dtype == torch.bool
    assert abs(keep.float().mean().item() - (1 - p)) < 0.02
    leaves = [x.double().requires_grad_() for x in (q, k, v)]
    want = reference(*leaves, causal=causal, dropout_p=p, dropout_mask=keep)
    torch.testing.assert_close(out.cpu().double(), want.detach(), **TOL[torch.float32])
    g = rand(out.shape, 9)
    want.backward(g.double())
    bwd = torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward
    grads = bwd(g.to("vk"), qv, kv, vv, None, [True, True, True, False], out, lse, cq, ck, mq, mk, p, causal, seed,
                offset)
    for got, leaf in zip(grads[:3], leaves):
        torch.testing.assert_close(got.cpu().double(), leaf.grad, **GRAD_TOL[torch.float32])
    # and through autograd: the same mask again for the same generator state
    tv.manual_seed(123)
    a = [x.detach().requires_grad_() for x in (qv, kv, vv)]
    o1 = F.scaled_dot_product_attention(*a, dropout_p=p, is_causal=causal)
    o1.backward(g.to("vk"))
    tv.manual_seed(123)
    o2 = op(qv, kv, vv, None, p, causal, True)
    torch.testing.assert_close(o1.detach().cpu(), o2[0].cpu())


def test_routes():
    q = torch.randn(1, 2, 8, 16, device="vk")
    choice = torch.ops.aten._fused_sdp_choice
    assert choice(q, q, q) == FUSED
    assert choice(q, q, q, dropout_p=0.1) == FUSED
    if HAS_FLOAT64:
        assert choice(q.double(), q.double(), q.double()) == MATH  # float64: the math path
    assert choice(q[0], q[0], q[0]) == MATH  # (3-d: sdpa adds the batch dim before choosing)
    m = torch.zeros(8, 8, device="vk", requires_grad=True)
    assert choice(q, q, q, m) == MATH  # a mask that needs a gradient
    with sdpa_kernel([SDPBackend.MATH]):
        assert choice(q, q, q) == MATH
    big = torch.randn(1, 1, 8, 300, device="vk")
    assert choice(big, big, big) == MATH  # head dim past 256
    with sdpa_kernel([SDPBackend.OVERRIDEABLE]):
        with pytest.raises(RuntimeError, match="No available kernel"):
            choice(big, big, big)


def test_forced_math():
    # TORCH_VK_SDPA=math prefers the math path (in a subprocess)
    script = ("import torch, torch_vulkan\n"
              "q = torch.randn(1, 2, 8, 16, device='vk')\n"
              "print(torch.ops.aten._fused_sdp_choice(q, q, q))\n")
    out = subprocess.run([sys.executable, "-c", script], env=dict(os.environ, TORCH_VK_SDPA="math"),
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == str(MATH)


def test_fused_long_sequence_memory():
    # 4 heads of 2048 x 2048 scores take 64 MiB in float32 on the math path;
    # the fused kernel's working memory is its output and logsumexp (2 MiB)
    q = torch.randn(1, 4, 2048, 64, device="vk")
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    out = F.scaled_dot_product_attention(q, q, q)
    assert torch.accelerator.max_memory_allocated() - before < 4 << 20
    with sdpa_kernel([SDPBackend.MATH]):
        F.scaled_dot_product_attention(q, q, q)
    assert torch.accelerator.max_memory_allocated() - before > 64 << 20
    ref = reference(q[:, :1, :128].cpu(), q[:, :1].cpu(), q[:, :1].cpu())
    torch.testing.assert_close(out[:, :1, :128].cpu().double(), ref, **TOL[torch.float32])
