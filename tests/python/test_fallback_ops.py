"""Ops that used to run on the CPU (found by tools/pytorch_tests.py's
per-test fallback record), now on the GPU: each against the CPU, with no
fallback recorded."""

import pytest
import torch
import torch_vulkan

from _util import make


def no_fallback(f):
    before = torch_vulkan.fallback_counts()
    r = f()
    after = torch_vulkan.fallback_counts()
    assert after == before, {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}
    return r


@pytest.mark.parametrize("ord", [2, 1, 0, 3.5])
@pytest.mark.parametrize("shape,dim", [((0,), None), ((3, 0), 1), ((0, 4), 0), ((0, 4), None)])
def test_vector_norm_empty(shape, dim, ord):
    x = torch.empty(shape, dtype=torch.float64)
    got = no_fallback(lambda: torch.linalg.vector_norm(x.to("vk"), ord, dim))
    torch.testing.assert_close(got.cpu(), torch.linalg.vector_norm(x, ord, dim))


def test_vector_norm_empty_inf_raises_as_cpu():
    with pytest.raises(RuntimeError, match="empty"):
        torch.linalg.vector_norm(torch.empty(0, device="vk"), float("inf"))


def test_efficient_zero_tensor():
    z = no_fallback(lambda: torch._efficientzerotensor((2, 3), device="vk", dtype=torch.float32))
    assert z.device.type == "vk" and z._is_zerotensor()
    torch.testing.assert_close((z + torch.ones(2, 3, device="vk")).cpu(), torch.ones(2, 3))


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
def test_random_unsigned(dtype):
    t = torch.empty(4000, dtype=dtype, device="vk")
    no_fallback(lambda: t.random_(3, 40))
    v = t.cpu().to(torch.int64)
    assert v.min().item() >= 3 and v.max().item() < 40 and len(v.unique()) == 37
    no_fallback(lambda: t.random_(7))
    v = t.cpu().to(torch.int64)
    assert v.min().item() >= 0 and v.max().item() < 7
    with pytest.raises(RuntimeError, match="out of bounds"):
        t.random_(-1, 5)


COMPLEX32_OPS = [
    ("add", lambda a, b: a + b), ("sub", lambda a, b: a - b), ("mul", lambda a, b: a * b),
    ("eq", lambda a, b: a == b), ("ne", lambda a, b: a != b), ("abs", lambda a, b: a.abs()),
    ("isnan", lambda a, b: a.isnan()), ("neg", lambda a, b: -a), ("conj_physical", lambda a, b: a.conj_physical()),
    ("where", lambda a, b: torch.where(a.real > 0, a, b)), ("tril", lambda a, b: a.tril()),
    ("masked_fill", lambda a, b: a.masked_fill(a.real > 0, 2)),
]


@pytest.mark.parametrize("name,f", COMPLEX32_OPS, ids=[n for n, _ in COMPLEX32_OPS])
def test_complex32(name, f):
    a = torch.complex(make((4, 5), torch.float32), make((4, 5), torch.float32, seed=1)).to(torch.complex32)
    b = torch.complex(make((4, 5), torch.float32, seed=2), make((4, 5), torch.float32, seed=3)).to(torch.complex32)
    b[0, 0] = a[0, 0]
    got = no_fallback(lambda: f(a.to("vk"), b.to("vk")))
    want = f(a, b)
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.cpu(), want, atol=1e-3, rtol=1e-3)


def test_complex32_unsupported_still_raises():
    a = torch.ones(3, dtype=torch.complex32, device="vk")
    with pytest.raises(RuntimeError):
        a.exp()


@pytest.mark.parametrize("layout", [torch.sparse_csr, torch.sparse_csc])
def test_compressed_indices_validated_on_gpu(layout):
    d = make((5, 6), torch.float32) * (make((5, 6), torch.float32, seed=1) > 0.3)
    s = d.to_sparse(layout=layout)
    ci, pi = (s.crow_indices(), s.col_indices()) if layout == torch.sparse_csr else (s.ccol_indices(), s.row_indices())
    make_it = torch.sparse_csr_tensor if layout == torch.sparse_csr else torch.sparse_csc_tensor
    t = no_fallback(lambda: make_it(ci.to("vk"), pi.to("vk"), s.values().to("vk"), d.shape, check_invariants=True))
    torch.testing.assert_close(t.to_dense().cpu(), d)


@pytest.mark.parametrize("bad", ["first", "last", "step", "plain", "order"])
def test_compressed_indices_invalid_raise_as_cpu(bad):
    crow = torch.tensor([0, 2, 3, 5])
    col = torch.tensor([0, 2, 1, 0, 3])
    if bad == "first":
        crow = torch.tensor([1, 2, 3, 5])
    elif bad == "last":
        crow = torch.tensor([0, 2, 3, 4])
    elif bad == "step":
        crow = torch.tensor([0, 3, 2, 5])
    elif bad == "plain":
        col = torch.tensor([0, 2, 1, 0, 9])
    else:
        col = torch.tensor([2, 0, 1, 0, 3])
    vals = torch.ones(5)
    with pytest.raises(RuntimeError) as cpu:
        torch.sparse_csr_tensor(crow, col, vals, (3, 4), check_invariants=True)
    with pytest.raises(RuntimeError) as vk:
        torch.sparse_csr_tensor(crow.to("vk"), col.to("vk"), vals.to("vk"), (3, 4), check_invariants=True)
    assert str(vk.value) == str(cpu.value)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize("max_length", [4, 2])
def test_jagged_padded_round_trip(dtype, max_length):
    offsets = torch.tensor([0, 3, 3, 4, 7])
    values = make((7, 2, 3), torch.float32).to(dtype)
    want = torch.ops.aten._jagged_to_padded_dense_forward(values, [offsets], [max_length], 5.0)
    got = no_fallback(lambda: torch.ops.aten._jagged_to_padded_dense_forward(values.to("vk"), [offsets.to("vk")],
                                                                            [max_length], 5.0))
    torch.testing.assert_close(got.cpu(), want)
    if max_length == 4:
        back = no_fallback(lambda: torch.ops.aten._padded_dense_to_jagged_forward(got, [offsets.to("vk")], 7))
        torch.testing.assert_close(back.cpu(), values)


def test_jagged_padding_inf_integer():
    offsets = torch.tensor([0, 1, 3])
    values = torch.arange(3)
    want = torch.ops.aten._jagged_to_padded_dense_forward(values, [offsets], [3], float("-inf"))
    got = torch.ops.aten._jagged_to_padded_dense_forward(values.to("vk"), [offsets.to("vk")], [3], float("-inf"))
    torch.testing.assert_close(got.cpu(), want)


def test_padded_to_jagged_too_long_raises_as_cpu():
    offsets = torch.tensor([0, 5])
    with pytest.raises(RuntimeError, match="found batch item of length 5"):
        torch.ops.aten._padded_dense_to_jagged_forward(torch.ones(1, 3, device="vk"), [offsets.to("vk")])


@pytest.mark.parametrize("shape,ns", [((0, 1), (1,)), ((2, 0), (0,)), ((0, 3, 2), (3, 2))])
def test_layer_norm_empty(shape, ns):
    for dev in ("cpu", "vk"):
        x = torch.randn(shape, device=dev, requires_grad=True)
        w = torch.ones(ns, device=dev, requires_grad=True)
        b = torch.zeros(ns, device=dev, requires_grad=True)
        f = (lambda: torch.nn.functional.layer_norm(x, ns, w, b).sum().backward()) if dev == "vk" else None
        if dev == "vk":
            no_fallback(f)
            torch.testing.assert_close(w.grad.cpu(), want_w)
            o, m, r = no_fallback(lambda: torch.native_layer_norm(x, ns, w, b, 1e-5))
            torch.testing.assert_close(m.cpu(), want_m, equal_nan=True)
            torch.testing.assert_close(r.cpu(), want_r, equal_nan=True)
        else:
            torch.nn.functional.layer_norm(x, ns, w, b).sum().backward()
            want_w = w.grad
            _, want_m, want_r = torch.native_layer_norm(x, ns, w, b, 1e-5)


def test_group_norm_empty_batch():
    x = torch.randn(0, 4, 3, device="vk", requires_grad=True)
    w = torch.ones(4, device="vk", requires_grad=True)
    y = no_fallback(lambda: torch.nn.functional.group_norm(x, 2, w))
    no_fallback(lambda: y.sum().backward())
    assert y.shape == (0, 4, 3)
    torch.testing.assert_close(w.grad.cpu(), torch.zeros(4))
