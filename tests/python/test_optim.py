"""Fused optimiser steps (torch.optim with fused=True) on "vk" against the
CPU's fused kernels: the same parameters, gradients and options, a few
steps, and no CPU fallback on the GPU side."""

import pytest
import torch
import torch_vulkan

from _util import HAS_FLOAT64, make

DTYPES = [torch.float32, torch.float16, torch.bfloat16] + ([torch.float64] if HAS_FLOAT64 else [])

CASES = [
    (torch.optim.Adam, {}),
    (torch.optim.Adam, {"weight_decay": 0.1, "amsgrad": True}),
    (torch.optim.Adam, {"maximize": True, "betas": (0.3, 0.6)}),
    (torch.optim.AdamW, {}),
    (torch.optim.AdamW, {"weight_decay": 0.2, "amsgrad": True, "maximize": True}),
    (torch.optim.SGD, {"lr": 0.1}),
    (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "weight_decay": 0.01}),
    (torch.optim.SGD, {"lr": 0.1, "momentum": 0.9, "nesterov": True, "maximize": True}),
    (torch.optim.SGD, {"lr": 0.1, "momentum": 0.5, "dampening": 0.3}),
    (torch.optim.Adagrad, {"lr": 0.1}),
    (torch.optim.Adagrad, {"lr": 0.1, "lr_decay": 0.05, "weight_decay": 0.1, "maximize": True}),
]


def tol(dtype):
    return {torch.float16: 2e-3, torch.bfloat16: 2e-2}.get(dtype, 1e-5)


def run(opt, kwargs, device, dtype, steps=3, shapes=((5, 7), (33,), (2, 3, 4)), lr_tensor=False):
    params = [torch.nn.Parameter(make(s, torch.float32, seed=i).to(dtype).to(device)) for i, s in enumerate(shapes)]
    kw = dict(kwargs)
    if lr_tensor:
        kw["lr"] = torch.tensor(kw.get("lr", 1e-3))
    o = opt(params, fused=True, **kw)
    for step in range(steps):
        for i, p in enumerate(params):
            p.grad = make(p.shape, torch.float32, seed=100 * step + i).to(dtype).to(device)
        o.step()
    return [p.detach().cpu() for p in params], [{k: v.cpu() for k, v in o.state[p].items()} for p in params]


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("opt,kwargs", CASES, ids=lambda c: getattr(c, "__name__", None) or repr(c))
def test_fused_step(opt, kwargs, dtype):
    before = torch_vulkan.fallback_counts()
    got = run(opt, kwargs, "vk", dtype)
    assert torch_vulkan.fallback_counts() == before
    want = run(opt, kwargs, "cpu", dtype)
    for g, w in zip(got[0], want[0]):
        torch.testing.assert_close(g, w, atol=tol(dtype), rtol=tol(dtype))
    for gs, ws in zip(got[1], want[1]):
        for k in ws:
            torch.testing.assert_close(gs[k], ws[k], atol=tol(dtype), rtol=tol(dtype))


@pytest.mark.parametrize("opt", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.Adagrad])
def test_fused_tensor_lr(opt):
    kwargs = {"lr": 0.05, "momentum": 0.9} if opt is torch.optim.SGD else {"lr": 0.05}
    got = run(opt, kwargs, "vk", torch.float32, lr_tensor=True)
    want = run(opt, kwargs, "cpu", torch.float32, lr_tensor=True)
    for g, w in zip(got[0], want[0]):
        torch.testing.assert_close(g, w)


@pytest.mark.parametrize("opt", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD, torch.optim.Adagrad])
@pytest.mark.parametrize("found_inf", [0.0, 1.0])
def test_fused_grad_scale_found_inf(opt, found_inf):
    # (as torch.amp.GradScaler passes them: the gradients unscaled in place,
    # and the whole step skipped where found_inf is 1)
    results = []
    for device in ("vk", "cpu"):
        p = torch.nn.Parameter(make((6, 5), torch.float32).to(device))
        g = make((6, 5), torch.float32, seed=3).to(device) * 8
        p.grad = g.clone()
        kwargs = {"lr": 0.1, "momentum": 0.9} if opt is torch.optim.SGD else {"lr": 0.1}
        o = opt([p], fused=True, **kwargs)
        o.grad_scale = torch.tensor(8.0, device=device)
        o.found_inf = torch.tensor(found_inf, device=device)
        o.step()
        results.append((p.detach().cpu(), p.grad.cpu()))
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
