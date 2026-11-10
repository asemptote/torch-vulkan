"""Ops that used to run on the CPU (found by tools/pytorch_tests.py's
per-test fallback record), now on the GPU: each against the CPU, with no
fallback recorded."""

import pytest
import torch
import torch_vulkan

from _util import make, make_complex


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


def test_index_with_cpu_indices():
    x = make((5, 4), torch.float32)
    mask = torch.tensor([True, False, True, False, True])
    rows = torch.tensor([3, 0])
    v = x.to("vk")
    torch.testing.assert_close(no_fallback(lambda: v[mask]).cpu(), x[mask])
    torch.testing.assert_close(no_fallback(lambda: v[rows]).cpu(), x[rows])
    no_fallback(lambda: v.index_put_((mask,), torch.full((3, 4), 7.0, device="vk")))
    torch.testing.assert_close(v.cpu(), x.index_put((mask,), torch.full((3, 4), 7.0)))


def test_nonzero_many_dims():
    x = (make((2, 1, 3, 1, 2, 2, 1, 3, 1, 2, 1, 2), torch.float32) > 0)
    torch.testing.assert_close(no_fallback(lambda: torch.nonzero(x.to("vk"))).cpu(), torch.nonzero(x))


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_scalar(side):
    s = torch.sort(make((20,), torch.float32)).values
    for value in (0.3, -5, 2):
        got = no_fallback(lambda: torch.searchsorted(s.to("vk"), value, side=side))
        torch.testing.assert_close(got.cpu(), torch.searchsorted(s, value, side=side))
    p = torch.randperm(20)
    u = s[torch.argsort(p)]
    got = no_fallback(lambda: torch.searchsorted(u.to("vk"), 0.1, sorter=torch.argsort(u).to("vk"), out_int32=True))
    torch.testing.assert_close(got.cpu(), torch.searchsorted(u, 0.1, sorter=torch.argsort(u), out_int32=True))


@pytest.mark.parametrize("op", [torch.logical_and, torch.logical_or, torch.logical_xor])
def test_logical_mixed_dtypes(op):
    a = make((6,), torch.float32) * (make((6,), torch.float32, seed=1) > 0)
    b = (make((6,), torch.float32, seed=2) > 0).to(torch.bfloat16)
    got = no_fallback(lambda: op(a.to("vk"), b.to("vk")))
    torch.testing.assert_close(got.cpu(), op(a, b))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_linspace_logspace_complex(dtype):
    for steps in (0, 1, 2, 7):
        got = no_fallback(lambda: torch.linspace(1 - 2j, -3 + 4j, steps, dtype=dtype, device="vk"))
        torch.testing.assert_close(got.cpu(), torch.linspace(1 - 2j, -3 + 4j, steps, dtype=dtype))
    got = torch.logspace(0.5 - 1j, 1 + 2j, 6, dtype=dtype, device="vk")
    torch.testing.assert_close(got.cpu(), torch.logspace(0.5 - 1j, 1 + 2j, 6, dtype=dtype))


def test_unique_internal():
    x = torch.tensor([3, 1, 3, 7, 1])
    for sorted_ in (True, False):
        got = no_fallback(lambda: torch._unique(x.to("vk"), sorted=sorted_, return_inverse=True))
        want = torch._unique(x, sorted=sorted_, return_inverse=True)
        torch.testing.assert_close(got[0].cpu(), want[0])
        torch.testing.assert_close(got[1].cpu(), want[1])


def test_add_complex_and_bool_alpha():
    a, b = torch.tensor([1 + 1j, 2 - 1j]), torch.tensor([3j, 1 + 0j])
    got = no_fallback(lambda: torch.add(a.to("vk"), b.to("vk"), alpha=2 - 1j))
    torch.testing.assert_close(got.cpu(), torch.add(a, b, alpha=2 - 1j))
    got = no_fallback(lambda: torch.sub(a.to("vk"), b.to("vk"), alpha=1j))
    torch.testing.assert_close(got.cpu(), torch.sub(a, b, alpha=1j))
    p, q = torch.tensor([True, False, False]), torch.tensor([True, True, False])
    for alpha in (True, False):
        got = no_fallback(lambda: torch.add(p.to("vk"), q.to("vk"), alpha=alpha))
        torch.testing.assert_close(got.cpu(), torch.add(p, q, alpha=alpha))
    with pytest.raises(RuntimeError, match="complex"):
        torch.add(torch.ones(2, device="vk"), torch.ones(2, device="vk"), alpha=1j)


def test_pinv_of_a_vector_raises_as_cpu():
    with pytest.raises(RuntimeError, match=r"linalg.pinv\(Float\{\[3\]\}\): expected a tensor with 2 or more dimensions"):
        torch.linalg.pinv(torch.ones(3, device="vk"))


def _grads(f, *inputs):
    """d(sum f(inputs) * w)/d(inputs) on the CPU and on vk, w fixed."""
    out = []
    for dev in ("cpu", "vk"):
        xs = [x.detach().to(dev).requires_grad_(i == 0) for i, x in enumerate(inputs)]
        y = f(*xs)
        w = (make_complex(y.shape, y.dtype) if y.dtype.is_complex else make(y.shape, y.dtype)).to(dev)
        loss = (y * w).sum()
        gs = torch.autograd.grad(loss.real if loss.is_complex() else loss, [x for x in xs if x.requires_grad])
        out.append([g.cpu() for g in gs])
    return out


@pytest.mark.parametrize("which", ["softshrink", "hardshrink"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
def test_shrink_backward(which, dtype):
    x = make((40,), torch.float32).to(dtype)
    f = getattr(torch.nn.functional, which)
    if dtype == torch.float16:  # (the forward stays the CPU's for float16)
        gx = torch.ops.aten.softshrink_backward if which == "softshrink" else torch.ops.aten.hardshrink_backward
        g = make((40,), torch.float32, seed=5).to(dtype)
        got = no_fallback(lambda: gx(g.to("vk"), x.to("vk"), 0.5))
        torch.testing.assert_close(got.cpu(), gx(g, x, 0.5))
        return
    cpu, vk = _grads(lambda t: f(t, 0.5), x)
    torch.testing.assert_close(vk[0], cpu[0])


@pytest.mark.parametrize("dims,shape,pad", [(1, (2, 3, 5), (2, 1)), (2, (1, 2, 4, 3), (1, 2, 0, 3)),
                                            (3, (1, 1, 2, 3, 2), (1, 1, 2, 0, 1, 2)), (2, (2, 5, 4), (-1, 2, 1, -1))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_replication_pad_backward(dims, shape, pad, dtype):
    x = make_complex(shape, dtype) if dtype.is_complex else make(shape, dtype)
    before = torch_vulkan.fallback_counts()
    cpu, vk = _grads(lambda t: torch.nn.functional.pad(t, pad, mode="replicate"), x)
    assert {k: v for k, v in torch_vulkan.fallback_counts().items() if "replication" in k} == \
        {k: v for k, v in before.items() if "replication" in k}
    torch.testing.assert_close(vk[0], cpu[0])


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("weighted", [False, True])
def test_multi_margin_loss_backward(p, reduction, weighted):
    x = make((5, 4), torch.float64)
    t = torch.tensor([0, 3, 1, 1, 2])
    w = make((4,), torch.float64, positive=True) if weighted else None
    cpu, vk = _grads(lambda a, b, *ww: torch.nn.functional.multi_margin_loss(
        a, b, p=p, margin=0.7, weight=ww[0].detach() if ww else None, reduction=reduction), x, t, *([w] if weighted else []))
    torch.testing.assert_close(vk[0], cpu[0])


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(3, 5), (5,)])
def test_multilabel_margin_loss_backward(reduction, shape):
    x = make(shape, torch.float64)
    t = torch.tensor([[3, 0, -1, 1, 2], [1, 1, 4, -1, 0], [-1, 2, 0, 0, 0]])[: shape[0] if len(shape) == 2 else 1]
    t = t.reshape(shape)
    cpu, vk = _grads(lambda a, b: torch.nn.functional.multilabel_margin_loss(a, b, reduction=reduction), x, t)
    torch.testing.assert_close(vk[0], cpu[0])


def test_complex32_fft():
    x = torch.complex(make((16,), torch.float32), make((16,), torch.float32, seed=1)).to(torch.complex32)
    got = no_fallback(lambda: torch.fft.fft(x.to("vk")))
    assert got.dtype == torch.complex32
    want = torch.fft.fft(x.to(torch.complex64))
    torch.testing.assert_close(got.cpu().to(torch.complex64), want, atol=2e-2, rtol=2e-2)


CL = torch.channels_last


@pytest.mark.parametrize("case", ["conv_x", "conv_w", "pad_reflect", "pad_replicate", "batch_norm", "group_norm"])
def test_channels_last_layouts_as_cpu(case):
    # (outputs and input gradients in the CPU's memory formats)
    x = make((2, 3, 6, 6), torch.float32)
    w = make((4, 3, 3, 3), torch.float32, seed=1)
    x_cl = case != "conv_w"
    fns = {
        "conv_x": lambda t, ww: torch.nn.functional.conv2d(t, ww),
        "conv_w": lambda t, ww: torch.nn.functional.conv2d(t, ww.to(memory_format=CL)),
        "pad_reflect": lambda t, ww: torch.nn.functional.pad(t, (1, 1, 2, 1), mode="reflect"),
        "pad_replicate": lambda t, ww: torch.nn.functional.pad(t, (1, 1, 2, 1), mode="replicate"),
        "batch_norm": lambda t, ww: torch.nn.functional.batch_norm(t, None, None, training=True),
        "group_norm": lambda t, ww: torch.nn.functional.group_norm(t, 3),
    }
    layouts = []
    for dev in ("cpu", "vk"):
        t = (x.to(memory_format=CL) if x_cl else x.clone()).to(dev).requires_grad_()
        y = fns[case](t, w.to(dev))
        y.backward(torch.ones_like(y))
        layouts.append((y.is_contiguous(memory_format=CL), t.grad.is_contiguous(memory_format=CL)))
    assert layouts[1] == layouts[0]


def test_strided_nested_tensor_autograd():
    # (PyTorch's device-generic NestedTensor kernels for vk nested tensors: no
    # CPU fallback, and autograd through them, which used to abort the process)
    a = make((1, 2, 4), torch.float64)
    b = make((2, 2, 4), torch.float64, seed=1)
    grads = []
    for dev in ("cpu", "vk"):
        x, y = a.clone().to(dev).requires_grad_(), b.clone().to(dev).requires_grad_()
        nt = torch.nested.as_nested_tensor([x, y])
        ln = torch.nn.functional.layer_norm(nt, (4,), torch.ones(4, dtype=torch.float64, device=dev),
                                            torch.zeros(4, dtype=torch.float64, device=dev))
        out = torch.nested.to_padded_tensor(torch.nn.functional.relu(ln) * 2, 0)
        if dev == "vk":
            before = torch_vulkan.fallback_counts()
        out.sum().backward()
        if dev == "vk":
            assert torch_vulkan.fallback_counts() == before
        grads.append((out.detach().cpu(), x.grad.cpu(), y.grad.cpu()))
    for g, w in zip(grads[1], grads[0]):
        torch.testing.assert_close(g, w)


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min", "prod"])
def test_segment_reduce_backward(reduce):
    d = torch.tensor([1.0, 0.0, 3.0, 3.0, float("nan"), 2.0, 0.0, 0.0, 5.0, -1.0], dtype=torch.float64)
    lengths = torch.tensor([2, 3, 0, 4, 1])
    grads = []
    for dev in ("cpu", "vk"):
        x = d.clone().to(dev).requires_grad_()
        y = torch.segment_reduce(x, reduce, lengths=lengths.to(dev))
        if dev == "vk":
            before = torch_vulkan.fallback_counts()
        y.backward(torch.tensor([1.0, -2.0, 3.0, 4.0, 0.5], dtype=torch.float64, device=dev))
        if dev == "vk":
            assert torch_vulkan.fallback_counts() == before
        grads.append(x.grad.cpu())
    torch.testing.assert_close(grads[1], grads[0], equal_nan=True)


@pytest.mark.parametrize("mask_type", [0, 1, 2])
def test_masked_softmax(mask_type):
    x = make((2, 3, 4, 4), torch.float32)
    m = {0: make((4, 4), torch.float32, seed=1) > 0.5, 1: make((2, 4), torch.float32, seed=2) > 0.3,
         2: make((2, 3, 4, 4), torch.float32, seed=3) > 0.2}[mask_type]
    if mask_type == 1:
        m[0] = True  # (a batch with every key masked: rows of NaN, as the CPU's)
    got = no_fallback(lambda: torch._masked_softmax(x.to("vk"), m.to("vk"), 3, mask_type))
    torch.testing.assert_close(got.cpu(), torch._masked_softmax(x, m, 3, mask_type), equal_nan=True)


def test_native_multi_head_attention_fast_path():
    # (nn.MultiheadAttention's inference fast path on vk, nested input too)
    torch.manual_seed(0)
    mha = torch.nn.MultiheadAttention(16, 4, batch_first=True).eval()
    x = torch.randn(3, 5, 16)
    with torch.no_grad():
        want = mha(x, x, x, need_weights=False)[0]
        mha_vk = mha.to("vk")
        got = no_fallback(lambda: mha_vk(x.to("vk"), x.to("vk"), x.to("vk"), need_weights=False)[0])
    torch.testing.assert_close(got.cpu(), want, atol=1e-5, rtol=1e-5)
