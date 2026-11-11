"""Quantised matrix products on "vk" vs the CPU: _weight_int8pack_mm,
_dyn_quant_pack_4bit_weight / _dyn_quant_matmul_4bit and _scaled_mm."""

import pytest
import torch

import torch_vulkan as tv
from _util import on_gpu

# (uint8 / int8 / float8 and float16 / bfloat16 tensors need 8- and 16-bit
# storage; without it every op on them, copies too, runs on the CPU)
pytestmark = pytest.mark.skipif((tv._native.capabilities() & 12) != 12, reason="no 8- or 16-bit storage")
FLOAT8 = [torch.float8_e4m3fn, torch.float8_e5m2]


def gen(seed):
    return torch.Generator().manual_seed(seed)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("m,k,n", [(32, 64, 48), (1, 4096, 300), (5, 100, 17), (70, 33, 5), (65, 1, 66)])
def test_weight_int8pack_mm(dtype, m, k, n):
    # shaders/qmm8.slang: weights dequantised as float(b) * scale while staged,
    # float32 sums (in another order than the CPU's), rounded once
    a = torch.randn(m, k, generator=gen(1)).to(dtype)
    b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=gen(2))
    s = (torch.rand(n, generator=gen(3)) / 50).to(dtype)
    # (the reference without the CPU's kernel: its AVX2 / AVX-512 bfloat16
    # path reads 16 values of k at a time and crashes for k not a multiple of
    # 16, as on the CI's runners: pytorch/pytorch#204474)
    want = (a.float() @ (b.float() * s.float()[:, None]).t()).to(dtype)
    with on_gpu():
        got = torch._weight_int8pack_mm(a.to("vk"), b.to("vk"), s.to("vk"))
        # a row slice (A's rows strided), as LLaMA's LM head passes it
        sliced = torch._weight_int8pack_mm(a.to("vk")[::2], b.to("vk"), s.to("vk"))
    mag = a.float().abs() @ (b.float() * s.float()[:, None]).abs().t()
    tol = torch.finfo(dtype).eps * want.float().abs() + 2**-20 * mag
    assert got.dtype == want.dtype and got.shape == want.shape
    assert ((got.cpu().float() - want.float()).abs() <= tol).all()
    assert ((sliced.cpu().float() - want[::2].float()).abs() <= tol[::2]).all()


def test_weight_int8pack_mm_errors():
    a, b, s = torch.randn(4, 8, device="vk"), torch.zeros(3, 8, dtype=torch.int8, device="vk"), torch.ones(3, device="vk")
    with pytest.raises(RuntimeError, match="expect B to be int8 tensor"):
        torch._weight_int8pack_mm(a, b.float(), s)
    with pytest.raises(RuntimeError, match="expect scales to be 1d tensor with size 3"):
        torch._weight_int8pack_mm(a, b, s[:2])
    with pytest.raises(RuntimeError, match="A must be contiguous on the last dimension"):
        torch._weight_int8pack_mm(a.t().contiguous().t(), b[:, :4], s)


def pack_case(n, k, bs, seed, bias):
    w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen(seed))
    nb = 1 if bs == k else k // bs
    sc = torch.rand(n, nb, generator=gen(seed + 1)) / 10
    sc = sc if bs == k else sc.to(torch.bfloat16)
    return w, sc, (torch.randn(n, generator=gen(seed + 2)) if bias else None)


@pytest.mark.parametrize("m,k,n,bs,bias", [(1, 64, 4096, 32, False), (32, 128, 11008, 32, True), (5, 96, 70, 96, True),
                                           (3, 100, 9, 100, False), (4, 256, 16, 64, True), (9, 64, 33, 64, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dyn_quant_matmul_4bit_exact(m, k, n, bs, bias, dtype):
    # shaders/qmm4.slang copies the CPU's reference kernels' arithmetic: the
    # results (and the packed weights) are equal bit for bit
    if dtype == torch.bfloat16 and bs != k:
        pytest.skip("bfloat16 inputs are channelwise only (the CPU's check)")
    w, sc, bvec = pack_case(n, k, bs, 7, bias)
    x = (torch.randn(m, k, generator=gen(11)) * 3).to(dtype)
    x[0, : k // 2] = 0.0  # a row's quantisation with zeros, and one all negative
    if m > 1:
        x[1] = -x[1].abs()
    packed = torch._dyn_quant_pack_4bit_weight(w, sc, bvec, bs, k, n)
    want = torch._dyn_quant_matmul_4bit(x, packed, bs, k, n)
    with on_gpu():
        packed_vk = torch._dyn_quant_pack_4bit_weight(w.to("vk"), sc.to("vk"), None if bvec is None else bvec.to("vk"),
                                                      bs, k, n)
        got = torch._dyn_quant_matmul_4bit(x.to("vk"), packed_vk, bs, k, n)
        got_cpu_packed = torch._dyn_quant_matmul_4bit(x.to("vk"), packed.to("vk"), bs, k, n)
    assert torch.equal(packed_vk.cpu(), packed)
    assert got.dtype == want.dtype
    assert torch.equal(got.cpu(), want)
    assert torch.equal(got_cpu_packed.cpu(), want)


def test_dyn_quant_errors():
    x = torch.randn(2, 64, device="vk")
    w, sc, _ = pack_case(8, 64, 32, 3, False)
    with pytest.raises(RuntimeError, match="Group size should be multiple of 32"):
        torch._dyn_quant_pack_4bit_weight(w.to("vk"), sc.to("vk"), None, 48, 64, 8)
    with pytest.raises(RuntimeError, match="expect weight to be kByte"):
        torch._dyn_quant_pack_4bit_weight(w.to("vk").char(), sc.to("vk"), None, 32, 64, 8)
    packed = torch._dyn_quant_pack_4bit_weight(w.to("vk"), sc.to("vk"), None, 32, 64, 8)
    with pytest.raises(RuntimeError, match="expect input to be float32 or bfloat16 tensor"):
        torch._dyn_quant_matmul_4bit(x.bfloat16(), packed, 32, 64, 8)
    with pytest.raises(RuntimeError, match="Invalid packed weight tensor size"):
        torch._dyn_quant_matmul_4bit(x, packed[:100], 32, 64, 8)


@pytest.mark.parametrize("fa,fb", [(FLOAT8[0], FLOAT8[0]), (FLOAT8[1], FLOAT8[0]), (FLOAT8[0], FLOAT8[1])])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32, torch.float16, torch.float8_e4m3fn, None])
@pytest.mark.parametrize("rowwise", [False, True])
def test_scaled_mm(fa, fb, out_dtype, rowwise):
    # the CPU's emulated path on the GPU: exact upcasts, scales, a float32
    # product, bias, the cast; within the float32 sums' order (the CPU's BLAS
    # and the GPU's GEMM add in different orders; even products of few bits
    # carry more than float32 holds once summed, as the H100 showed), then
    # one rounding to the output type
    a = torch.randn(16, 32, generator=gen(1)).to(fa)
    b = torch.randn(48, 32, generator=gen(2)).to(fb).t()
    sa = torch.rand(16, 1, generator=gen(3)) + 0.5 if rowwise else torch.tensor(1.25)
    sb = torch.rand(1, 48, generator=gen(4)) + 0.5 if rowwise else torch.tensor(0.75)
    bias = torch.randn(48, generator=gen(5)).to(torch.bfloat16) if out_dtype == torch.bfloat16 else None
    scale_result = torch.tensor(2.0) if out_dtype == torch.float8_e4m3fn else None
    kw = dict(bias=bias, out_dtype=out_dtype, scale_result=scale_result)
    want = torch._scaled_mm(a, b, sa, sb, **kw)
    vkw = {k: (v.to("vk") if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}
    with on_gpu():
        got = torch._scaled_mm(a.to("vk"), b.to("vk"), sa.to("vk"), sb.to("vk"), **vkw)
        out = torch.empty(0, dtype=want.dtype, device="vk")
        torch._scaled_mm(a.to("vk"), b.to("vk"), sa.to("vk"), sb.to("vk"), **vkw, out=out)
    assert got.dtype == want.dtype and got.shape == want.shape
    mag = (a.float() * sa).abs() @ (b.float() * sb).abs() / (2.0 if scale_result is not None else 1.0)
    eps = torch.finfo(want.dtype).eps
    tol = eps * want.float().abs() + 2**-20 * mag
    for res in (got, out):
        assert ((res.cpu().float() - want.float()).abs() <= tol).all()


def test_scaled_mm_errors():
    a = torch.randn(4, 8).to(FLOAT8[0]).to("vk")
    b = torch.randn(6, 8).to(FLOAT8[0]).to("vk").t()
    one = torch.tensor(1.0, device="vk")
    with pytest.raises(ValueError, match="Invalid scaling configuration"):
        torch._scaled_mm(a, b, torch.ones(3, device="vk"), one)
    with pytest.raises(ValueError, match="out_dtype must match output matrix type"):
        torch._scaled_mm(a, b, one, one, out_dtype=torch.float32, out=torch.empty(0, device="vk", dtype=torch.half))
    with pytest.raises(RuntimeError, match="Bias must be size 6"):
        torch._scaled_mm(a, b, one, one, bias=torch.ones(5, device="vk"), out_dtype=torch.float32)
