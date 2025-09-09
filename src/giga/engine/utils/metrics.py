import torch
import torchvision.transforms.v2 as tv2
from jaxtyping import Float, Int
from torch import Tensor
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.ops import masks_to_boxes


def prepare_bboxes(mask: Float[Tensor, "b 1 h w"]) -> Int[Tensor, "b 4"]:
    _bboxes = masks_to_boxes(mask.squeeze(1)).int()
    bboxes = torch.zeros_like(_bboxes)
    bboxes[:, 0] = _bboxes[:, 1]
    bboxes[:, 1] = _bboxes[:, 0]
    bboxes[:, 2] = _bboxes[:, 3] - _bboxes[:, 1]
    bboxes[:, 3] = _bboxes[:, 2] - _bboxes[:, 0]
    return bboxes


def eval_psnr(
    pred: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
    mask: Float[Tensor, "b 1 h w"] | None = None,
    **kwargs,
) -> Tensor:
    if mask is not None:
        pred = pred * mask
        target = target * mask
        bboxes = prepare_bboxes(mask)

        batch_size = pred.shape[0]
        scores = torch.zeros(batch_size, device=pred.device)
        for idx in range(batch_size):
            pred_crop = tv2.functional.crop(pred[idx], *bboxes[idx].tolist())
            target_crop = tv2.functional.crop(target[idx], *bboxes[idx].tolist())
            scores[idx] = peak_signal_noise_ratio(pred_crop, target_crop, **kwargs)
        score = scores.mean()
    else:
        score = peak_signal_noise_ratio(pred, target, **kwargs)
    return score


def eval_ssim(
    pred: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
    mask: Float[Tensor, "b 1 h w"] | None = None,
    **kwargs,
) -> Tensor:
    if mask is not None:
        pred = pred * mask
        target = target * mask
        bboxes = prepare_bboxes(mask)

        batch_size = pred.shape[0]
        scores = torch.zeros(batch_size, device=pred.device)
        for idx in range(batch_size):
            pred_crop = tv2.functional.crop(pred[idx], *bboxes[idx].tolist())
            target_crop = tv2.functional.crop(target[idx], *bboxes[idx].tolist())
            scores[idx] = structural_similarity_index_measure(pred_crop.unsqueeze(0), target_crop.unsqueeze(0), **kwargs)
        score = scores.mean()
    else:
        score = structural_similarity_index_measure(pred, target, **kwargs)
    return score


def eval_lpips(
    pred: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
    mask: Float[Tensor, "b 1 h w"] | None = None,
    model: LearnedPerceptualImagePatchSimilarity | None = None,
    net_type: str = "alex",
    **kwargs,
) -> Tensor:
    if model is None:
        model = LearnedPerceptualImagePatchSimilarity(net_type=net_type, **kwargs).to(pred.device)

    if mask is not None:
        pred = pred * mask
        target = target * mask
        bboxes = prepare_bboxes(mask)

        batch_size = pred.shape[0]
        scores = torch.zeros(batch_size, device=pred.device)
        for idx in range(batch_size):
            pred_crop = tv2.functional.crop(pred[idx], *bboxes[idx].tolist()).clamp(0, 1)
            target_crop = tv2.functional.crop(target[idx], *bboxes[idx].tolist()).clamp(0, 1)
            scores[idx] = model(pred_crop.unsqueeze(0) * 2.0 - 1, target_crop.unsqueeze(0) * 2.0 - 1)
        score = scores.mean()
    else:
        score = model(pred, target, normalize=True)
    return score
