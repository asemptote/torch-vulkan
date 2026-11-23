"""torchvision's nms and roi_align on "vk" (docs/log/M57): compared with
torchvision's CPU kernels. Skipped without torchvision."""

import pytest
import torch

import torch_vulkan as tv
from _util import HAS_FLOAT64, on_gpu

torchvision = pytest.importorskip("torchvision")
from torchvision import ops  # noqa: E402

DTYPES = [torch.float32] + ([torch.float64] if HAS_FLOAT64 else [])


def make_boxes(n, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * 100
    wh = torch.rand(n, 2, generator=g) * 40
    boxes = torch.cat([xy, xy + wh], 1)
    # (degenerate boxes: zero width, zero area, a point, one inverted: the
    # CPU's IoU is then 0, NaN or negative)
    boxes[1] = torch.tensor([10.0, 10.0, 10.0, 30.0])
    boxes[2] = torch.tensor([5.0, 5.0, 5.0, 5.0])
    boxes[3] = boxes[4] + 0.5
    boxes[5] = torch.tensor([30.0, 30.0, 20.0, 20.0])
    # (clusters of near-identical boxes, so that many overlap)
    boxes[10:40] = boxes[10] + torch.rand(30, 4, generator=g)
    return boxes.to(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 1.0, -0.5])
def test_nms(dtype, threshold):
    n = 700
    boxes = make_boxes(n, dtype)
    g = torch.Generator().manual_seed(1)
    scores = (torch.rand(n, generator=g) * 20).round() / 20  # (many ties: kept in index order)
    scores = scores.to(dtype)
    want = ops.nms(boxes, scores, threshold)
    with on_gpu():
        got = ops.nms(boxes.to("vk"), scores.to("vk"), threshold)
    assert got.dtype == torch.int64 and got.device.type == "vk"
    assert torch.equal(got.cpu(), want)


def test_nms_threshold_between_floats():
    # (float32 IoUs compared with the double threshold, as the CPU compares
    # them: an IoU of exactly 0.5 is not above 0.5, nor above 0.49999999999)
    boxes = torch.tensor([[0.0, 0.0, 2.0, 1.0], [1.0, 0.0, 3.0, 1.0], [0.0, 0.0, 3.0, 1.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    for t in (0.5, 0.49999999999, 0.50000000001, 1 / 3, 2 / 3, 0.6666666666):
        want = ops.nms(boxes, scores, t)
        with on_gpu():
            got = ops.nms(boxes.to("vk"), scores.to("vk"), t)
        assert torch.equal(got.cpu(), want), t


def test_nms_edge_cases():
    with on_gpu():
        empty = ops.nms(torch.empty(0, 4, device="vk"), torch.empty(0, device="vk"), 0.5)
        one = ops.nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="vk"), torch.tensor([0.5], device="vk"), 0.5)
    assert empty.shape == (0,) and empty.dtype == torch.int64
    assert one.tolist() == [0]
    # NaN scores sort first (descending), as on the CPU; NaN coordinates
    boxes = make_boxes(50, torch.float32)
    scores = torch.rand(50)
    scores[[3, 17]] = float("nan")
    boxes[7, 2] = float("nan")
    with on_gpu():
        got = ops.nms(boxes.to("vk"), scores.to("vk"), 0.4)
    assert torch.equal(got.cpu(), ops.nms(boxes, scores, 0.4))
    # (the CPU's errors)
    for b, s in [(torch.zeros(3, 5), torch.zeros(3)), (torch.zeros(3, 4), torch.zeros(4)),
                 (torch.zeros(3, 4), torch.zeros(3, 1))]:
        with pytest.raises(RuntimeError) as cpu_error:
            ops.nms(b, s, 0.5)
        with pytest.raises(RuntimeError) as vk_error:
            ops.nms(b.to("vk"), s.to("vk"), 0.5)
        assert str(vk_error.value).split("\n")[0] in str(cpu_error.value)  # (the CPU's, without its location)


@pytest.mark.parametrize("n", [3000, 6000])
def test_batched_nms(n):
    # (torchvision's coordinate offset below 20,000 coordinates, its loop
    # over classes above)
    boxes = make_boxes(n, torch.float32, seed=2)
    scores = torch.rand(n, generator=torch.Generator().manual_seed(3))
    idxs = torch.randint(0, 5, (n,), generator=torch.Generator().manual_seed(4))
    want = ops.batched_nms(boxes, scores, idxs, 0.5)
    got = ops.batched_nms(boxes.to("vk"), scores.to("vk"), idxs.to("vk"), 0.5)
    assert torch.equal(got.cpu(), want)


def make_rois(k, batch, height, width, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(k, 2, generator=g) * torch.tensor([width, height]) * 1.2 - 4
    wh = torch.rand(k, 2, generator=g) * torch.tensor([width, height]) * 0.8
    b = torch.randint(0, batch, (k, 1), generator=g).float()
    rois = torch.cat([b, xy, xy + wh], 1)
    rois[0, 1:] = torch.tensor([3.0, 3.0, 3.2, 3.1])  # (smaller than a pixel)
    rois[1, 1:] = torch.tensor([5.0, 6.0, 4.0, 2.0])  # (inverted)
    rois[2, 1:] = torch.tensor([-30.0, -30.0, -20.0, -25.0])  # (outside)
    return rois.to(dtype)


ROI_CASES = [
    # (output size, spatial scale, sampling ratio, aligned)
    ((7, 7), 1.0, 2, False),
    ((7, 7), 0.5, 2, True),
    ((5, 3), 0.25, -1, False),
    ((14, 14), 0.125, 0, True),
    ((2, 2), 1.0, 1, False),
    ((3, 4), 0.7, -1, True),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ROI_CASES)
def test_roi_align(dtype, case):
    size, scale, sampling, aligned = case
    x = torch.randn(2, 3, 23, 31, generator=torch.Generator().manual_seed(5), dtype=dtype)
    rois = make_rois(40, 2, 23 / scale, 31 / scale, dtype=dtype)
    want = ops.roi_align(x, rois, size, scale, sampling, aligned)
    with on_gpu():
        got = ops.roi_align(x.to("vk"), rois.to("vk"), size, scale, sampling, aligned)
    # (the CPU's operations in its order, divisions correctly rounded: bit for bit)
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", ROI_CASES)
def test_roi_align_backward(dtype, case):
    size, scale, sampling, aligned = case
    x = torch.randn(2, 3, 23, 31, generator=torch.Generator().manual_seed(6), dtype=dtype, requires_grad=True)
    rois = make_rois(40, 2, 23 / scale, 31 / scale, seed=1, dtype=dtype)
    grad = torch.randn(40, 3, *size, generator=torch.Generator().manual_seed(7), dtype=dtype)
    ops.roi_align(x, rois, size, scale, sampling, aligned).backward(grad)
    xv = x.detach().to("vk").requires_grad_()
    with on_gpu():
        ops.roi_align(xv, rois.to("vk"), size, scale, sampling, aligned).backward(grad.to("vk"))
    # (each input element's contributions added in the CPU's order: bit for bit)
    torch.testing.assert_close(xv.grad.cpu(), x.grad, rtol=0, atol=0)


def test_roi_align_edge_cases():
    x = torch.randn(2, 4, 9, 11)
    with on_gpu():
        empty = ops.roi_align(x.to("vk"), torch.empty(0, 5, device="vk"), (7, 7), 1.0, 2, False)
    assert empty.shape == (0, 4, 7, 7)
    # boxes as a list per image (torchvision converts them to rois on the device)
    boxes = [torch.tensor([[1.0, 1.0, 6.0, 5.0]]), torch.tensor([[0.0, 2.0, 9.0, 8.0], [2.0, 2.0, 3.0, 3.0]])]
    want = ops.roi_align(x, boxes, (3, 3), 1.0, -1, True)
    with on_gpu():
        got = ops.roi_align(x.to("vk"), [b.to("vk") for b in boxes], (3, 3), 1.0, -1, True)
    torch.testing.assert_close(got.cpu(), want, rtol=1e-6, atol=1e-6)
    # non-contiguous input; the gradient with no regions
    xt = x.transpose(2, 3)
    rois = make_rois(5, 2, 11, 9)
    with on_gpu():
        got = ops.roi_align(xt.to("vk"), rois.to("vk"), (2, 2), 1.0, 2, False)
    torch.testing.assert_close(got.cpu(), ops.roi_align(xt, rois, (2, 2), 1.0, 2, False), rtol=1e-6, atol=1e-6)
    xv = x.to("vk").requires_grad_()
    ops.roi_align(xv, torch.empty(0, 5, device="vk"), (2, 2), 1.0, 2, False).sum().backward()
    assert xv.grad.abs().sum().item() == 0


def test_roi_align_batch_index_out_of_range():
    x = torch.randn(2, 3, 8, 8, device="vk")
    rois = torch.tensor([[2.0, 0.0, 0.0, 4.0, 4.0]], device="vk")
    with pytest.raises((RuntimeError, IndexError)):
        ops.roi_align(x, rois, (2, 2), 1.0, 2, False)
        tv.synchronize()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_roi_align_16_bit(dtype):
    # (in float32 on the GPU, rounded once, where the CPU computes float16 in
    # float16 and rejects bfloat16; nms raises the CPU's error)
    x = torch.randn(2, 3, 23, 31, generator=torch.Generator().manual_seed(8)).to(dtype)
    rois = make_rois(20, 2, 23, 31).to(dtype)
    want = ops.roi_align(x.float(), rois.float(), (5, 5), 1.0, 2, True).to(dtype)
    with on_gpu():
        got = ops.roi_align(x.to("vk"), rois.to("vk"), (5, 5), 1.0, 2, True)
    assert got.dtype == dtype
    torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="not implemented for"):
        ops.nms(rois[:, 1:].to("vk"), rois[:, 0].to("vk"), 0.5)


def test_multiscale_roi_align():
    # (the module Mask R-CNN's heads use)
    pooler = ops.MultiScaleRoIAlign(["0", "1"], 7, 2)
    g = torch.Generator().manual_seed(9)
    feats = {"0": torch.randn(2, 4, 32, 32, generator=g), "1": torch.randn(2, 4, 16, 16, generator=g)}
    boxes = [torch.tensor([[1.0, 2.0, 40.0, 50.0], [10.0, 10.0, 120.0, 100.0]]), torch.tensor([[0.0, 0.0, 64.0, 64.0]])]
    sizes = [(128, 128), (128, 128)]
    want = pooler(feats, boxes, sizes)
    with on_gpu():
        got = pooler({k: v.to("vk") for k, v in feats.items()}, [b.to("vk") for b in boxes], sizes)
    torch.testing.assert_close(got.cpu(), want, rtol=1e-6, atol=1e-6)
