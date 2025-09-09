import torch
from jaxtyping import Int
from torch import Tensor


def center_crop_bbox(
    resolution: tuple[int, int],
) -> Int[Tensor, "4"]:
    min_side = min(resolution)
    center_x = resolution[0] // 2
    center_y = resolution[1] // 2
    half_side = min_side // 2
    bbox = torch.tensor([center_x - half_side, center_y - half_side, min_side, min_side], dtype=torch.int32)
    return bbox


def sample_character_patches(masks: Tensor, num_patches: int, patch_size: int, patch_scales: list[float]) -> tuple[Tensor, Tensor]:
    """
    Samples patch centers and corresponding patch dimensions from a batch of mask tensors.
    This function iterates over a batch of masks and, for each mask, finds the indices where the mask
    has a positive value (assumed to be in the first channel). It then randomly samples a number of patch
    centers from these indices. For each sampled center, a patch dimension is computed based on a provided
    patch size and randomly selected scales from a list of patch_scales. The patch dimensions are stored as
    half the scaled patch size (to possibly represent a radius or half-width) along with the scale used.
    Args:
        masks (torch.Tensor): A tensor containing mask data with shape (N, C, H, W) or similar, where the
            function uses the first channel (index 0) to determine valid regions (non-zero values).
        num_patches (int): The number of patches to sample per mask.
        patch_size (int): The base size for patches. The final patch size is computed by scaling this value.
        patch_scales (list[float]): A list of scaling factors to apply to the patch_size for generating the
            dimensions of each patch.
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - centers_xy (torch.Tensor): A tensor of shape (N, num_patches, 2) containing the (x, y) coordinates
              of sampled patch centers (stored as int16).
            - patch_dims (torch.Tensor): A tensor of shape (N, num_patches) where for each patch, contains half the scaled patch size (as an int16).
    """

    centers_xy = torch.zeros((masks.shape[0], num_patches, 2), dtype=torch.int16)  # store centers of patches
    patch_dims = torch.zeros(
        (masks.shape[0], num_patches), dtype=torch.int16
    )  # store patch dims (1 number, square patches) + scale of the patch
    patch_scales = torch.tensor(patch_scales, dtype=torch.float32)
    for idx in range(masks.shape[0]):
        mask_indices = torch.argwhere(masks[idx, 0] > 0)
        sampled_indices = torch.randint(0, mask_indices.shape[0], (num_patches,))
        sampled_xy = torch.roll(mask_indices[sampled_indices], 1, dims=-1)
        centers_xy[idx] = sampled_xy.to(torch.int16)
        scale = patch_scales[torch.randint(0, len(patch_scales), (num_patches,))]
        patch_dims[idx, :] = (patch_size * scale / 2).to(torch.int16)
    return centers_xy, patch_dims
