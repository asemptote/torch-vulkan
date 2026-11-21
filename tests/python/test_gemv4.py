"""Matrix products with one to eight vectors (shaders/gemv4.slang: 16-byte
loads, float32 sums): float32, float16 and bfloat16 mm / addmm / addmv /
linear, int8 weights (_weight_int8pack_mm) and 4-bit weights
(_convert_weight_to_int4pack, _weight_int4pack_mm).

gemv4 runs by default on Arm GPUs (Mali); TORCH_VK_GEMV=vector runs it on
any GPU, which is how the desktop and lavapipe runs exercise it. Without it
the same tests check the other paths."""

import pytest
import torch

import torch_vulkan as tv
from _util import on_gpu

pytestmark = pytest.mark.skipif((tv._native.capabilities() & 12) != 12, reason="no 8- or 16-bit storage")
FLOATS = [torch.float32, torch.float16, torch.bfloat16]


def gen(seed):
    return torch.Generator().manual_seed(seed)


def close(got, want64, mag, dtype):
    """got (vk, dtype) against a float64 reference: float32 sums in another
    order (relative to the sum of |products|), then one rounding to dtype."""
    got = got.cpu().double()
    tol = 2**-16 * mag + torch.finfo(dtype).eps * want64.abs() + 1e-30
    bad = (got - want64).abs() > tol
    assert not bad.any(), f"{int(bad.sum())} of {bad.numel()} off; worst {float(((got - want64).abs() / tol).max()):.3g} x tol"


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("k", [1, 3, 8, 33, 1001, 4096])
@pytest.mark.parametrize("ncol", [1, 2, 3, 5, 8])
def test_linear(dtype, k, ncol):
    # x @ W.t() (the decoding step of nn.Linear): W [rows, k] row-major
    rows = 77
    w = torch.randn(rows, k, generator=gen(1)).to(dtype)
    x = torch.randn(ncol, k, generator=gen(2)).to(dtype)
    bias = torch.randn(rows, generator=gen(3)).to(dtype)
    with on_gpu():
        got = torch.nn.functional.linear(x.to("vk"), w.to("vk"))
        got_b = torch.nn.functional.linear(x.to("vk"), w.to("vk"), bias.to("vk"))
    want = x.double() @ w.double().t()
    mag = x.double().abs() @ w.double().abs().t()
    assert got.dtype == dtype and got.shape == (ncol, rows)
    close(got, want, mag, dtype)
    close(got_b, want + bias.double(), mag + bias.double().abs(), dtype)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("k", [5, 64, 300])
@pytest.mark.parametrize("ncol", [1, 4, 7])
def test_mm_columns(dtype, k, ncol):
    # W @ X (X [k, ncol] row-major: the vectors are copied k-contiguous)
    rows = 130
    w = torch.randn(rows, k, generator=gen(4)).to(dtype)
    x = torch.randn(k, ncol, generator=gen(5)).to(dtype)
    with on_gpu():
        got = w.to("vk") @ x.to("vk")
        got_t = (x.to("vk").t() @ w.to("vk").t()).t()
    want, mag = w.double() @ x.double(), w.double().abs() @ x.double().abs()
    close(got, want, mag, dtype)
    close(got_t, want, mag, dtype)


@pytest.mark.parametrize("dtype", FLOATS)
def test_addmm_addmv_strided(dtype):
    # alpha, beta, a broadcast bias, W's rows strided (a slice: unaligned rows)
    k, rows = 37, 50
    big = torch.randn(rows, k + 3, generator=gen(6)).to(dtype)
    w = big[:, 1 : k + 1]
    x = torch.randn(3, k, generator=gen(7)).to(dtype)
    v = torch.randn(k, generator=gen(8)).to(dtype)
    c = torch.randn(rows, generator=gen(9)).to(dtype)
    with on_gpu():
        wv = big.to("vk")[:, 1 : k + 1]
        got = torch.addmm(c.to("vk"), x.to("vk"), wv.t(), beta=0.5, alpha=2.0)
        got_mv = torch.addmv(c.to("vk"), wv, v.to("vk"), beta=-1.0, alpha=0.25)
        got_mv0 = torch.addmv(torch.full((rows,), float("nan"), dtype=dtype, device="vk"), wv, v.to("vk"), beta=0.0)
    xd, wd, cd = x.double(), w.double(), c.double()
    close(got, 0.5 * cd + 2.0 * xd @ wd.t(), cd.abs() + 2.0 * xd.abs() @ wd.abs().t(), dtype)
    want_mv = -cd + 0.25 * wd @ v.double()
    close(got_mv, want_mv, cd.abs() + 0.25 * wd.abs() @ v.double().abs(), dtype)
    close(got_mv0, wd @ v.double(), wd.abs() @ v.double().abs(), dtype)


@pytest.mark.parametrize("dtype", FLOATS)
def test_big_row_count(dtype):
    # more rows than one row block, and a 4096-long k
    w = torch.randn(4100, 4096, generator=gen(10)).to(dtype)
    x = torch.randn(4096, generator=gen(11)).to(dtype)
    with on_gpu():
        got = w.to("vk") @ x.to("vk")
    close(got, w.double() @ x.double(), w.double().abs() @ x.double().abs(), dtype)


def test_lstm_step():
    # an LSTM run step by step: [1, input] @ W.t() per step
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(40, 24, batch_first=True, bidirectional=True)
    x = torch.randn(1, 9, 40)
    want, _ = lstm(x)
    with on_gpu():
        got, _ = lstm.to("vk")(x.to("vk"))
    assert torch.allclose(got.cpu(), want, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("m", [1, 2, 3, 8])
@pytest.mark.parametrize("k", [1, 17, 64, 1000])
def test_int8pack(dtype, m, k):
    n = 45
    a = torch.randn(m, k, generator=gen(12)).to(dtype)
    b = torch.randint(-128, 128, (n, k), dtype=torch.int8, generator=gen(13))
    s = (torch.rand(n, generator=gen(14)) / 50).to(dtype)
    with on_gpu():
        got = torch._weight_int8pack_mm(a.to("vk"), b.to("vk"), s.to("vk"))
    wd = b.double() * s.float().double()[:, None]  # (dequantised in float32: exact in float64)
    close(got, a.double() @ wd.t(), a.double().abs() @ wd.abs().t(), dtype)


def quantise4(w, group):
    """_group_quantize_tensor's packing (torch.testing's): w [n, k] to bytes
    [n, k / 2] (even k in the high nibble) and scales and zeros [k / group, n, 2]."""
    n, k = w.shape
    g = w.reshape(-1, group)
    lo, hi = g.amin(1, keepdim=True), g.amax(1, keepdim=True)
    scale = (hi - lo).clamp(min=1e-6) / 15
    zero = lo + scale * 8
    q = g.sub(lo).div(scale).round().clamp(0, 15).to(torch.int32).reshape(n, k)
    packed = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8)
    sz = torch.cat([scale.reshape(n, -1, 1), zero.reshape(n, -1, 1)], 2).transpose(0, 1).contiguous()
    return q, packed, sz


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("m", [1, 2, 5, 8, 9, 33])
@pytest.mark.parametrize("k,group", [(32, 32), (256, 64), (512, 128), (1024, 256), (4096, 32)])
def test_int4pack(dtype, m, k, group):
    n = 48
    w = torch.randn(n, k, generator=gen(15))
    q, packed, sz = quantise4(w, group)
    sz = sz.to(dtype)
    a = torch.randn(m, k, generator=gen(16)).to(dtype)
    with on_gpu():
        b4 = torch._convert_weight_to_int4pack(packed.to("vk"), 2)
        assert b4.dtype == torch.int32 and b4.shape == (n // 8, k // 32, 32, 1)
        got = torch._weight_int4pack_mm(a.to("vk"), b4, group, sz.to("vk"))
    szf = sz.float()
    scale = szf[:, :, 0].t().repeat_interleave(group, 1)
    zero = szf[:, :, 1].t().repeat_interleave(group, 1)
    wd = ((q - 8).float() * scale + zero).double()  # (as the CPU dequantises: in float32)
    assert got.dtype == dtype and got.shape == (m, n)
    close(got, a.double() @ wd.t(), a.double().abs() @ wd.abs().t(), dtype)


def test_int4pack_errors():
    w = torch.zeros(16, 32, dtype=torch.uint8, device="vk")
    with pytest.raises(RuntimeError, match="innerKTiles need to be 2, 4, or 8"):
        torch._convert_weight_to_int4pack(w, 3)
    with pytest.raises(RuntimeError, match="expect weight to be kByte"):
        torch._convert_weight_to_int4pack(w.int(), 2)
    b4 = torch._convert_weight_to_int4pack(w, 2)
    a = torch.zeros(2, 64, device="vk")
    sz = torch.zeros(2, 16, 2, device="vk")
    with pytest.raises(RuntimeError, match="expect qGroupSize to be 32, 64, 128 or 256"):
        torch._weight_int4pack_mm(a, b4, 16, sz)
    with pytest.raises(RuntimeError, match="expect qScaleAndZeros to be 3d tensor"):
        torch._weight_int4pack_mm(a, b4, 32, sz[:, :8])
