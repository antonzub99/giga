# partially borrowed from
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def ssim_loss(
    pred: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
    window_size: int = 11,
    **kwargs,
) -> Float[Tensor, "..."]:
    return 1.0 - ssim(pred, target, window_size, **kwargs)


def masked_ssim_loss(
    pred: Float[Tensor, "b c h w"],
    target: Float[Tensor, "b c h w"],
    window_size: int = 11,
    **kwargs,
) -> Float[Tensor, "..."]:
    mask = target[:, 3:4, :, :] > 0.5
    return ssim_loss(pred[:, :3, :, :] * mask, target[:, :3, :, :] * mask, window_size, **kwargs)


def ssim(
    img1: Float[Tensor, "b c h w"],
    img2: Float[Tensor, "b c h w"],
    window_size: int = 11,
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (Float[Tensor, "b c h w"]): First image tensor.
        img2 (Float[Tensor, "b c h w"]): Second image tensor.
        window_size (int): Size of the Gaussian window to use for SSIM calculation.

    Returns:
        score (Float[Tensor, "..."]): SSIM score between img1 and img2.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    window = window.to(img1)
    return _ssim(img1, img2, window, window_size, channel)


def _ssim(
    img1: Float[Tensor, "b c h w"],
    img2: Float[Tensor, "b c h w"],
    window: Float[Tensor, "c 1 h w"],
    window_size: int,
    channel: int,
) -> Float[Tensor, "..."]:
    C1 = 0.01**2
    C2 = 0.03**2

    dtype = img1.dtype
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.clamp(F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq, min=0.0)
    sigma2_sq = torch.clamp(F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq, min=0.0)
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (((mu1_sq + mu2_sq).to(dtype) + C1) * ((sigma1_sq + sigma2_sq).to(dtype) + C2))
    ssim_map = ssim_map.clip(-1, 1)  # preventing numerical errors when training in half precision

    return ssim_map.mean()


def gaussian(window_size: int, sigma: float):
    gauss = torch.tensor([np.exp(-((((x - window_size) / 2) / (sigma * 2)) ** 2) / 2) for x in range(window_size)], dtype=torch.float32)
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> Float[Tensor, "c 1 h w"]:
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
