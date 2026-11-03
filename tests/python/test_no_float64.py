"""float32 ops whose PyTorch implementations use float64 temporaries, and the
TypeError for float64 tensors, on GPUs without float64 arithmetic (run the
suite with TORCH_VK_DISABLE=float64 to test those paths on any GPU)."""

import pytest
import torch
import torch.nn.functional as F

from torch_vulkan import _kernels, _nofloat64
from _util import HAS_FLOAT64, check, make, make_complex, on_gpu


@pytest.mark.skipif(HAS_FLOAT64, reason="this GPU has float64 arithmetic")
@pytest.mark.parametrize("make_wide", [lambda: torch.zeros(3, dtype=torch.float64, device="vk"),
                                       lambda: torch.randn(3, device="vk").double(),
                                       lambda: torch.randn(3, dtype=torch.complex128).to("vk")])
def test_float64_tensors_are_a_type_error(make_wide):
    with pytest.raises(TypeError, match="no float64 arithmetic"):
        make_wide()


def test_float64_free_quantile_matches_pytorch():
    # (_quantile_impl directly, so that GPUs with float64 test it too; it
    # equals PyTorch's quantile on the same device bit for bit, which differs
    # from the CPU's by the last bit of lerp in linear mode)
    x = make((5, 7, 33), torch.float32)
    x[0, 1, 3] = float("nan")
    x[2, :, 5] = float("nan")
    q = torch.tensor([0.0, 0.25, 0.3, 0.71, 1.0])
    for mode in ["linear", "lower", "higher", "midpoint", "nearest"]:
        for nan in (False, True):
            ref = torch.nanquantile if nan else torch.quantile
            for qq, dim, keepdim in [(0.3, None, False), (1 / 3, 1, True), (q, -1, False), (q, None, True), (0.999999, 0, False)]:
                got = _nofloat64._quantile_impl(nan, x.to("vk"), qq.to("vk") if isinstance(qq, torch.Tensor) else qq,
                                                dim, keepdim, mode)
                want = ref(x, qq, dim=dim, keepdim=keepdim, interpolation=mode)
                tol = dict(rtol=1e-6, atol=1e-6) if mode == "linear" else dict(rtol=0, atol=0)
                check(got, want, equal_nan=True, **tol)


def test_float64_free_linalg_and_zeta():
    m = make((3, 6, 4), torch.float32)
    m[1, :, 3] = m[1, :, 0]  # rank-deficient
    for t in (m, make_complex((3, 6, 4), torch.complex64)):
        with on_gpu():
            got = [_nofloat64._pinv_impl(t.to("vk"), None, None, False), _nofloat64._matrix_rank_impl(t.to("vk"), None, None, False),
                   _nofloat64._matrix_rank_impl(t.to("vk"), 0.5, 0.0, False)]
        check(got[0], torch.linalg.pinv(t), rtol=1e-4, atol=1e-4)
        check(got[1], torch.linalg.matrix_rank(t))
        check(got[2], torch.linalg.matrix_rank(t, atol=0.5, rtol=0.0))
    s = torch.rand(20) * 6 + 1.1
    check(torch.special.zeta(s.to("vk"), 2.0), torch.special.zeta(s, 2.0), rtol=1e-5, atol=1e-6)
    check(torch.special.zeta(3.0, s.to("vk")), torch.special.zeta(3.0, s), rtol=1e-5, atol=1e-6)


def test_float64_free_helpers():
    x = make((2, 3, 9, 7), torch.float32)
    vals = torch.cat([make((500,), torch.float32) * 1e3, torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan"),
                                                                       1e-40, -3e-45, 1.17e-38, 3.4e38])])
    with on_gpu():
        bits = _kernels._float_bits_as_double(vals.to("vk"))
        tri = [torch.triu_indices(5, 7, 1, device="vk"), torch.tril_indices(6, 4, -1, device="vk"),
               torch.triu_indices(4, 4, -9, device="vk"), torch.tril_indices(3, 3, -5, device="vk")]
        pooled = F.adaptive_avg_pool2d(x.to("vk"), (4, 3))
        empty_bn = F.batch_norm(torch.zeros(0, 3, device="vk"), torch.zeros(3, device="vk"), torch.ones(3, device="vk"),
                                training=True)
    check(bits, vals.double().view(torch.int64))
    for got, want in zip(tri, [torch.triu_indices(5, 7, 1), torch.tril_indices(6, 4, -1), torch.triu_indices(4, 4, -9),
                               torch.tril_indices(3, 3, -5)]):
        check(got, want)
    check(pooled, F.adaptive_avg_pool2d(x, (4, 3)), rtol=1e-6, atol=1e-6)
    assert empty_bn.shape == (0, 3)
    assert "device='vk:0'" in repr(x.to("vk"))
