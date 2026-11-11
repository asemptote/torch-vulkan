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
    got = no_fallback(lambda: torch.logspace(0.5 - 1j, 1 + 2j, 6, dtype=dtype, device="vk"))
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.uint8])
@pytest.mark.parametrize("out_size", [(3, 5), (16, 11)])
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_lanczos_aa(dtype, out_size, align_corners):
    x = make((2, 3, 8, 7), torch.float32, positive=True) * 100
    x = x.to(dtype)
    f = torch.ops.aten._upsample_lanczos2d_aa
    got = no_fallback(lambda: f(x.to("vk"), out_size, align_corners))
    want = f(x, out_size, align_corners)
    if dtype == torch.uint8:
        assert (got.cpu().int() - want.int()).abs().max() <= 1
    else:
        # (float32: the CPU forms the weights and the two passes in float32;
        # values near 100 with negative lobes agree to a few ulps of 100)
        tol = 1e-4 if dtype == torch.float32 else None
        torch.testing.assert_close(got.cpu(), want, atol=tol, rtol=tol)
    if dtype == torch.float64:
        g = make(want.shape, torch.float64, seed=4)
        b = torch.ops.aten._upsample_lanczos2d_aa_backward
        got_b = no_fallback(lambda: b(g.to("vk"), out_size, x.shape, align_corners))
        torch.testing.assert_close(got_b.cpu(), b(g, out_size, x.shape, align_corners))


FLOAT8 = [torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]


@pytest.mark.parametrize("f8", FLOAT8, ids=str)
@pytest.mark.parametrize("src", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64, torch.bool],
                         ids=str)
def test_float8_casts(f8, src):
    # every float8 code to the other dtypes, and values that round, saturate,
    # overflow and are NaN, inf or -0 to float8: bit for bit as the CPU's
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(f8)
    x = torch.cat([make((300,), torch.float32) * 3, make((300,), torch.float32, seed=1) * 300,
                   torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"), 448.0, 464.0, 57344.0, 61440.0,
                                 1e-9, -1e-9, 2.0 ** -9, 2.0 ** -10, 2.0 ** -17, 1e30])]).to(src)
    got = no_fallback(lambda: x.to("vk").to(f8))
    assert torch.equal(got.cpu().view(torch.uint8), x.to(f8).view(torch.uint8))
    back = no_fallback(lambda: codes.to("vk").to(src))
    torch.testing.assert_close(back.cpu(), codes.to(src), equal_nan=True, atol=0, rtol=0)


def test_out_on_another_device_raises_without_cpu():
    # (PyTorch's test_out case 3: a CPU out= for vk inputs raises, from vk's
    # own check, with nothing run on the CPU)
    x = make((3,), torch.float32).to("vk")
    for f in (lambda: torch.exp(x, out=torch.empty(3)), lambda: torch.cat([x, x], out=torch.empty(6)),
              lambda: torch.sort(x, out=(torch.empty(3), torch.empty(3, dtype=torch.long)))):
        before = torch_vulkan.fallback_counts()
        with pytest.raises(RuntimeError, match="same device"):
            f()
        assert torch_vulkan.fallback_counts() == before


def test_where_keeps_operand_layout():
    # (as TensorIterator: a fresh output in the first full-shape operand's dim
    # order; compiled graphs view it with the strides FakeTensor predicted)
    x = make((2, 3, 4), torch.float32).transpose(0, 1)
    y = make((4,), torch.float32, seed=1)
    for f in (lambda a, b: torch.where(a > 0, a, b), lambda a, b: torch.where(a > 0, a, 0.0)):
        assert f(x.to("vk"), y.to("vk")).stride() == f(x, y).stride()


@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
@pytest.mark.parametrize("op", [torch.eq, torch.ne, torch.lt, torch.le, torch.gt, torch.ge], ids=lambda f: f.__name__)
def test_compare_unsigned(dtype, op):
    big = {torch.uint16: 65535, torch.uint32: 2**32 - 1, torch.uint64: 2**63 + 5}[dtype]
    a = torch.tensor([0, 1, 7, big, big, 3], dtype=dtype)
    b = torch.tensor([0, 2, 7, big, 1, big], dtype=dtype)
    if op not in (torch.eq, torch.ne):  # (the CPU raises for these; vk raises the same, itself)
        with pytest.raises(NotImplementedError) as cpu:
            op(a, b)
        before = torch_vulkan.fallback_counts()
        with pytest.raises(NotImplementedError) as vk:
            op(a.to("vk"), b.to("vk"))
        assert str(vk.value) == str(cpu.value) and torch_vulkan.fallback_counts() == before
        return
    torch.testing.assert_close(no_fallback(lambda: op(a.to("vk"), b.to("vk"))).cpu(), op(a, b))
    for s in (0, 7, big):
        torch.testing.assert_close(no_fallback(lambda: op(a.to("vk"), s)).cpu(), op(a, s))


def test_scalar_minus_complex_and_float8_isnan():
    c = torch.complex(make((4,), torch.float32), make((4,), torch.float32, seed=1))
    torch.testing.assert_close(no_fallback(lambda: 1 - c.to("vk")).cpu(), 1 - c)
    torch.testing.assert_close(no_fallback(lambda: torch.rsub(c.to("vk"), 2, alpha=3)).cpu(), torch.rsub(c, 2, alpha=3))
    x = torch.tensor([1.0, float("nan"), float("inf"), 0.0])
    for f8 in FLOAT8:
        torch.testing.assert_close(no_fallback(lambda: torch.isnan(x.to(f8).to("vk"))).cpu(), torch.isnan(x.to(f8)))


def test_rescued_fallback_cases():
    # (cases that went to the CPU in full-43, now on vk: CPU scalars among vk
    # tensors, out= of another dtype, empty inputs, empty scatter indices,
    # bool powers and linspace, argmax without dim, float8 equality, unsigned
    # unique_dim, complex32 norms, embedding_renorm_)
    b = torch.tensor([True, False, True])
    f = make((3,), torch.float32)
    d = make((3,), torch.float64, seed=1)
    cases = [
        lambda dev: b.to(dev) * True,
        lambda dev: torch.bitwise_xor(b.to(dev), True),
        lambda dev: torch.clamp_max(f.to(dev), d.to(dev), out=torch.empty(3, dtype=torch.complex64, device=dev)),
        lambda dev: torch.empty(0, 3, dtype=torch.int32, device=dev).normal_(),
        lambda dev: torch.scatter(make((2, 3), torch.float32).to(dev), 0, torch.empty(0, dtype=torch.long, device=dev),
                                  make((2, 3), torch.float32).to(dev)),
        lambda dev: b.to(dev) ** False,
        lambda dev: torch.linspace(2, 3, 1, dtype=torch.bool, device=dev),
        lambda dev: torch.argmax(make((2, 3), torch.float32).to(dev), keepdim=True),
        lambda dev: torch.tensor([1.0, float("nan")]).to(torch.float8_e4m3fn).to(dev) ==
        torch.tensor([1.0, float("nan")]).to(torch.float8_e4m3fn).to(dev),
        lambda dev: torch.unique(torch.tensor([[3, 1], [3, 1], [0, 2**15]], dtype=torch.uint16).to(dev), dim=0),
        lambda dev: torch.linalg.vector_norm(make_complex((4,), torch.complex64).to(torch.complex32).to(dev)),
        lambda dev: torch.embedding_renorm_(make((5, 3), torch.float32).to(dev) * 3, torch.tensor([[0, 2], [2, 4]]).to(dev),
                                            1.0, 2.0),
    ]
    for case in cases:
        want = case("cpu")
        got = no_fallback(lambda: case("vk"))
        torch.testing.assert_close(got.cpu() if isinstance(got, torch.Tensor) else got, want, equal_nan=True,
                                   atol=1e-2 if want.dtype == torch.float16 else None,
                                   rtol=1e-2 if want.dtype == torch.float16 else None)
    with pytest.raises(NotImplementedError, match="linspace_cpu"):
        torch.linspace(0, 3, 4, dtype=torch.bool, device="vk")


@pytest.mark.parametrize("include_self", [True, False])
def test_complex_scatter_products(include_self):
    g = torch.Generator().manual_seed(0)
    self_ = torch.randn(4, 5, dtype=torch.complex64, generator=g)
    src = torch.randn(6, 5, dtype=torch.complex64, generator=g)
    index = torch.randint(0, 4, (6, 5), generator=g)
    cases = [
        lambda d: d(self_).scatter_reduce(0, d(index), d(src), "prod", include_self=include_self),
        lambda d: d(self_).scatter(0, d(index), d(src), reduce="multiply"),
        lambda d: d(self_).scatter(0, d(index), 1.5, reduce="multiply"),
    ]
    if include_self:  # (the CPU raises otherwise: no complex kernel)
        cases.append(lambda d: d(self_).index_reduce(0, d(index[:, 0]), d(src), "prod"))
    for f in cases:
        got = no_fallback(lambda: f(lambda t: t.to("vk")))
        torch.testing.assert_close(got.cpu(), f(lambda t: t))


@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_empty_index(reduce):
    x = torch.arange(6.0).view(2, 3)
    idx = torch.empty(0, dtype=torch.long)
    got = no_fallback(lambda: x.to("vk").scatter_reduce(0, idx.to("vk"), x.to("vk"), reduce))
    torch.testing.assert_close(got.cpu(), x.scatter_reduce(0, idx, x, reduce))
    got = no_fallback(lambda: x.to("vk").scatter(0, idx.to("vk"), x.to("vk"), reduce="add"))
    torch.testing.assert_close(got.cpu(), x)


def test_index_fill_cpu_index_isin_dtypes_where_unsigned():
    x = torch.arange(12.0).view(3, 4)
    idx = torch.tensor([0, 2])
    got = no_fallback(lambda: x.to("vk").index_fill_(1, idx, -1.0))
    torch.testing.assert_close(got.cpu(), x.clone().index_fill_(1, idx, -1.0))
    e, t = torch.tensor([1.0, 2.5, 3.0], dtype=torch.half), torch.tensor([2.5, 3.0])
    got = no_fallback(lambda: torch.isin(e.to("vk"), t.to("vk")))
    torch.testing.assert_close(got.cpu(), torch.isin(e, t))
    for dt in (torch.uint16, torch.uint32, torch.uint64):
        a, b = torch.tensor([1, 60000, 3]).to(dt), torch.tensor([4, 5, 65535]).to(dt)
        c = torch.tensor([True, False, True])
        got = no_fallback(lambda: torch.where(c.to("vk"), a.to("vk"), b.to("vk")))
        assert got.dtype == dt and torch.equal(got.cpu(), torch.where(c, a, b))


@pytest.mark.parametrize("transpose", [False, True])
def test_csr_to_coo_batched(transpose):
    crow = torch.tensor([[0, 1, 3], [0, 2, 3]]).repeat(2, 1, 1)
    col = torch.tensor([[1, 0, 1], [0, 1, 1]]).repeat(2, 1, 1)
    for c, k in ((crow[0], col[0]), (crow, col)):
        got = no_fallback(lambda: torch._convert_indices_from_csr_to_coo(c.to("vk"), k.to("vk"), transpose=transpose))
        assert torch.equal(got.cpu(), torch._convert_indices_from_csr_to_coo(c, k, transpose=transpose))


def _same(got, want):
    for g_, w_ in zip(got, want):
        assert g_.dtype == w_.dtype and torch.equal(g_.cpu(), w_)


def test_empty_cases_without_cpu():
    v = lambda t: t.to("vk")
    s = torch.tensor([False, True])
    got = no_fallback(lambda: torch.searchsorted(v(s), torch.empty(0, 3, dtype=torch.bool, device="vk")))
    assert got.shape == (0, 3) and got.dtype == torch.long
    with pytest.raises(NotImplementedError, match="not implemented for 'Bool'"):
        torch.searchsorted(v(s), v(s))
    e = torch.empty([1] * 9 + [0], dtype=torch.bool)
    assert no_fallback(lambda: v(e).all()).item() is True
    got = no_fallback(lambda: v(e).any(dim=(0, 9), keepdim=True))
    assert torch.equal(got.cpu(), e.any(dim=(0, 9), keepdim=True))
    a, b = torch.empty(2, 0, dtype=torch.bool), torch.empty(0, 3, dtype=torch.bool)
    _same([no_fallback(lambda: torch.mm(v(a), v(b)))], [torch.mm(a, b)])
    _same([no_fallback(lambda: torch.bmm(v(a)[None], v(b)[None]))], [torch.bmm(a[None], b[None])])
    for dt in (torch.half, torch.int32, torch.bool):
        for shape in ((2, 0, 0), (2, 3, 0)):
            m = torch.empty(shape, dtype=dt)
            _same(no_fallback(lambda: torch.linalg.lu(v(m))), torch.linalg.lu(m))
        m = torch.empty(2, 0, 0, dtype=dt)
        _same(no_fallback(lambda: torch.linalg.lu_factor_ex(v(m))), torch.linalg.lu_factor_ex(m))
    m = torch.empty(0, 0, dtype=torch.half)
    _same(no_fallback(lambda: torch.linalg.cholesky_ex(v(m))), torch.linalg.cholesky_ex(m))
    t = torch.empty(4, dtype=torch.uint16, device="vk")
    no_fallback(lambda: t.random_(65530, None))  # ([65530, 65535])
    assert t.cpu().to(torch.int64).min().item() >= 65530


@pytest.mark.parametrize("shape,dim", [((4, 6), 1), ((6, 3), 0), ((2, 0, 4), 2)])
def test_glu_jvps(shape, dim):
    g = torch.Generator().manual_seed(0)
    x, dx = torch.randn(shape, generator=g), torch.randn(shape, generator=g)
    glu = torch.nn.functional.glu(x, dim)
    gg, dgg = torch.randn(glu.shape, generator=g), torch.randn(glu.shape, generator=g)
    gx = torch.ops.aten.glu_backward(gg, x, dim)
    v = lambda t: t.to("vk")
    got = no_fallback(lambda: torch.ops.aten.glu_jvp(v(glu), v(x), v(dx), dim))
    torch.testing.assert_close(got.cpu(), torch.ops.aten.glu_jvp(glu, x, dx, dim))
    got = no_fallback(lambda: torch.ops.aten.glu_backward_jvp(v(gx), v(gg), v(x), v(dgg), v(dx), dim))
    torch.testing.assert_close(got.cpu(), torch.ops.aten.glu_backward_jvp(gx, gg, x, dgg, dx, dim))


def test_compute_linear_combination():
    x, c = torch.arange(6.0).view(3, 2), torch.arange(6.0).view(2, 3)
    got = no_fallback(lambda: torch.ops.aten._compute_linear_combination(x.to("vk"), c.to("vk")))
    torch.testing.assert_close(got.cpu(), torch.ops.aten._compute_linear_combination(x, c))


ERROR_CASES = {
    "group_norm_int": lambda d: torch.ops.aten.native_group_norm(torch.ones(2, 4, 3, dtype=torch.int, device=d), None,
                                                                None, 2, 4, 3, 2, 1e-5),
    "random_complex": lambda d: torch.ones(3, dtype=torch.complex64, device=d).random_(0, 10),
    "random_complex_full": lambda d: torch.ones(3, dtype=torch.complex64, device=d).random_(),
    "sort_complex": lambda d: torch.sort(torch.ones(3, dtype=torch.complex64, device=d)),
    "nll_target": lambda d: torch.ops.aten.nll_loss_forward(torch.ones(2, 3, device=d), torch.ones(2, device=d), None,
                                                            1, -100),
    "masked_scatter_mask": lambda d: torch.ones(2, 2, device=d).masked_scatter_(
        torch.ones(2, 2, dtype=torch.uint8, device=d), torch.ones(4, device=d)),
    "nan_to_num_out": lambda d: torch.nan_to_num(torch.ones(2, device=d), out=torch.empty(2, dtype=torch.long, device=d)),
    "mm_dtypes": lambda d: torch.mm(torch.ones(2, 2, dtype=torch.int, device=d), torch.ones(2, 2, device=d),
                                    out=torch.empty(2, 2, dtype=torch.int, device=d)),
    "mm_out_dtype": lambda d: torch.mm(torch.ones(2, 2, device=d), torch.ones(2, 2, device=d),
                                       out=torch.empty(2, 2, dtype=torch.int, device=d)),
    "eigvals_1d": lambda d: torch.linalg.eigvals(torch.ones(3, device=d)),
    "cholesky_inverse_nonsquare": lambda d: torch.cholesky_inverse(torch.ones(2, 3, device=d)),
    "sum_complex32": lambda d: torch.ones(3, dtype=torch.complex32, device=d).sum(),
    "linear_combination_empty": lambda d: torch.ops.aten._compute_linear_combination(torch.ones((), device=d),
                                                                                     torch.ones(2, 2, device=d)),
}


@pytest.mark.parametrize("name", list(ERROR_CASES))
def test_cpu_errors_raised_on_vk(name):
    """The CPU's error type and message, raised on vk with no fallback."""
    f = ERROR_CASES[name]
    with pytest.raises(Exception) as cpu:
        f("cpu")
    before = torch_vulkan.fallback_counts()
    with pytest.raises(type(cpu.value)) as vk:
        f("vk")
    assert torch_vulkan.fallback_counts() == before
    assert str(vk.value).splitlines()[0] == str(cpu.value).splitlines()[0]


def test_op_without_cpu_kernel_raises_without_cpu():
    q = torch.ones(1, 2, 1, 4, device="vk")
    before = torch_vulkan.fallback_counts()
    with pytest.raises(NotImplementedError):
        torch.ops.aten._efficient_attention_forward(q, q, q, None, None, None, None, None, 0.0, 0, False)
    assert torch_vulkan.fallback_counts() == before


@pytest.mark.parametrize("dtype", [None, torch.int64, torch.uint8, torch.int8, torch.half, torch.bfloat16, torch.bool])
def test_prod_dtype(dtype):
    x = torch.tensor([1.5, 2.0, 3.0, -1.0])
    got = no_fallback(lambda: torch.prod(x.to("vk"), dtype=dtype))
    want = torch.prod(x, dtype=dtype)
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.cpu(), want)
    b = torch.tensor([True, True, False])
    got = no_fallback(lambda: torch.prod(b.to("vk")))
    assert got.dtype == torch.int64 and got.item() == 0


@pytest.mark.parametrize("dtype", [torch.complex32, torch.float16, torch.bfloat16])
def test_lazy_bits_resolved_on_gpu_16bit(dtype):
    x = torch.randn(5, dtype=torch.complex64 if dtype == torch.complex32 else torch.float32).to(dtype)
    views = [lambda t: t._neg_view()] + ([lambda t: t.conj()] if dtype == torch.complex32 else [])
    for view in views:
        got = no_fallback(lambda: view(x.to("vk")).clone())
        want = view(x).clone()
        assert not got.is_conj() and not got.is_neg()
        assert torch.equal(got.cpu().view(torch.int16 if dtype != torch.complex32 else torch.int32),
                           want.view(torch.int16 if dtype != torch.complex32 else torch.int32))


@pytest.mark.parametrize("k", [2, 3])
def test_adaptive_max_pool_out(k):
    x = torch.randn(1, 2, *([5] * k))
    op = getattr(torch.ops.aten, f"adaptive_max_pool{k}d").out
    out, idx = torch.empty(0, device="vk"), torch.empty(0, dtype=torch.long, device="vk")
    no_fallback(lambda: op(x.to("vk"), [3] * k, out=out, indices=idx))
    want, want_idx = op(x, [3] * k, out=torch.empty(0), indices=torch.empty(0, dtype=torch.long))
    torch.testing.assert_close(out.cpu(), want)
    assert torch.equal(idx.cpu(), want_idx)


BCOMPLEX32_CASES = {
    "c64_to": lambda t, d: t.to(torch.complex64).to(torch.bcomplex32),
    "to_c64": lambda t, d: t.to(torch.complex64),
    "to_c32": lambda t, d: t.to(torch.complex32),
    "real_to_f32": lambda t, d: t.real.to(torch.float32),
    "eq": lambda t, d: t == t,
    "isnan": lambda t, d: t.isnan(),
    "where": lambda t, d: torch.where(torch.tensor([True, False, True], device=d), t, torch.zeros_like(t)),
}


@pytest.mark.skipif(not hasattr(torch, "bcomplex32"), reason="torch without bcomplex32")
@pytest.mark.parametrize("name", list(BCOMPLEX32_CASES))
def test_bcomplex32(name):
    x = torch.tensor([1 + 2j, float("nan") + 0j, -3.5 + 0.25j]).to(torch.bcomplex32)
    f = BCOMPLEX32_CASES[name]
    got = no_fallback(lambda: f(x.to("vk"), "vk"))
    want = f(x, "cpu")
    assert got.dtype == want.dtype
    widen = (lambda t: t.to(torch.complex64)) if want.is_complex() else (lambda t: t)
    torch.testing.assert_close(widen(got.cpu()), widen(want), equal_nan=True)


def test_d45_regressions():
    v = lambda t: t.to("vk")
    # an out= on another device raises, even for an empty input (no rescue first)
    with pytest.raises(RuntimeError, match="device"):
        torch.nan_to_num(torch.empty(1, 0, 3, device="vk"), out=torch.empty(0))
    # nonzero's out= must be int64, as the CPU's
    with pytest.raises(RuntimeError, match="scalar type Long"):
        torch.nonzero(torch.ones(3, device="vk"), out=torch.empty([], device="vk"))
    with pytest.raises(RuntimeError, match="nonzero_static: Expected out tensor to have scalar type Long"):
        torch.nonzero_static(torch.ones(3, device="vk"), size=2, out=torch.empty(0, device="vk"))
    x = torch.tensor([0.0, 1, 1])
    out = torch.empty(0, dtype=torch.long, device="vk")
    no_fallback(lambda: torch.nonzero_static(v(x), size=4, fill_value=-7, out=out))
    assert torch.equal(out.cpu(), torch.nonzero_static(x, size=4, fill_value=-7))
    # histc of integers, as CUDA's; its range errors
    got = no_fallback(lambda: torch.histc(torch.tensor([1, 2, 1], dtype=torch.int8, device="vk"), bins=4, min=5, max=5))
    assert got.dtype == torch.int8 and got.tolist() == [2, 0, 0, 1]
    with pytest.raises(RuntimeError, match="max must be larger than min"):
        torch.histc(torch.ones(3, dtype=torch.int32, device="vk"), bins=4, min=1, max=0)
    # _compute_linear_combination's out= adds to out, as the CPU's
    xs, c = torch.rand(3, 2, 2), torch.rand(1, 3)
    want = torch.ones(1, 2, 2)
    torch._compute_linear_combination(xs, c, out=want)
    got = torch.ones(1, 2, 2, device="vk")
    no_fallback(lambda: torch._compute_linear_combination(v(xs), v(c), out=got))
    torch.testing.assert_close(got.cpu(), want)
