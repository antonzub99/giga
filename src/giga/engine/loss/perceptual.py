import torch
from einops import rearrange
from jaxtyping import Float, Int16
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class NoOpLPIPS(Module):
    """
    A placeholder LPIPS network that returns a zero tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros(1, device=x.device, dtype=x.dtype)


def cut_patches(
    image: Float[Tensor, "b c h w"],
    centers: Int16[Tensor, "b n 2"],
    cut_size: Int16[Tensor, "b n"],
    patch_size: int,
) -> Float[Tensor, "b n c s s"]:
    """
    Cut patches from the image based on the given centers and sizes.

    Args:
        image (Float[Tensor, "b c h w"]): The input image tensor.
        centers (Int16[Tensor, "b n 2"]): The centers of the patches to cut.
        cut_size (Int16[Tensor, "b n"]): The sizes of the patches to cut, assuming square patches - half of the square side.
        patch_size (int): The size of the patches to paste in the output.

    Returns:
        patches (Float[Tensor, "b n c s s"]): The cut patches from the image.

    """

    batch_size, num_patches, _ = centers.shape
    output_patches = torch.zeros((batch_size, num_patches, image.shape[1], patch_size, patch_size), device=image.device, dtype=image.dtype)

    for idx in range(batch_size):
        for pidx in range(num_patches):
            crop = cut_size[idx, pidx]
            x = torch.clamp(centers[idx, pidx, 0], crop, image.shape[-1] - crop)
            y = torch.clamp(centers[idx, pidx, 1], crop, image.shape[-2] - crop)
            patch = image[idx, ..., y - crop : y + crop, x - crop : x + crop]
            patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size), mode="bilinear", align_corners=False).squeeze(0)
            output_patches[idx, pidx] += patch

    return output_patches


def perceptual_patch_loss(
    x: Float[Tensor, "b c h w"],
    y: Float[Tensor, "b c h w"],
    center: Int16[Tensor, "b n 2"],
    radius: Int16[Tensor, "b n"],
    size: int,
    network: Module,
) -> Float[Tensor, "..."]:
    """
    Compute the perceptual loss between two images over patches with specified centers and sizes.

    Args:
        x (Float[Tensor, "b c h w"]): The first image tensor.
        y (Float[Tensor, "b c h w"]): The second image tensor.
        center (Int16[Tensor, "b n 2"]): The centers of the patches to cut.
        radius (Int16[Tensor, "b n"]): The radius of the patches to cut, assuming square patches.
        size (int): The size of the patches to paste in the output.
        network (Module): The perceptual network to compute the loss.

    Returns:
        score (Float[Tensor, "..."]): The perceptual loss score between the two images, computed over cut patches.
    """

    dtype = x.dtype
    x_patched = cut_patches(x, center, radius, size)
    y_patched = cut_patches(y, center, radius, size)

    x_patched = rearrange(x_patched, "b n c h w -> (b n) c h w")
    x_patched = x_patched.to(dtype) * 2.0 - 1.0
    y_patched = rearrange(y_patched, "b n c h w -> (b n) c h w")
    y_patched = y_patched.to(dtype) * 2.0 - 1.0

    score = network(x_patched[:, :3], y_patched[:, :3])
    return score
