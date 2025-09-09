from jaxtyping import Float
from torch import Tensor


def adjust_intrinsics_for_crop_and_resize(
    intrinsics: Float[Tensor, " ... 3 3"],
    original_resolution: Float[Tensor, " ... 2"],
    target_resolution: tuple[int, int],
) -> Float[Tensor, " ... 3 3"]:
    """
    Adjust camera intrinsics for center crop and resize operations.

    Args:
        intrinsics: Camera intrinsic matrix [3, 3] or [N, 3, 3]
        original_resolution: Original image resolution [width, height] or [N, 2]
        target_resolution: Target resolution as (width, height)

    Returns:
        Adjusted intrinsic matrix with same shape as input
    """
    is_batch = intrinsics.ndim == 3
    if not is_batch:
        intrinsics = intrinsics.unsqueeze(0)
        original_resolution = original_resolution.unsqueeze(0)

    adjusted_intrinsics = intrinsics.clone()

    if target_resolution is not None:
        target_w, target_h = target_resolution

    for i in range(intrinsics.shape[0]):
        K = intrinsics[i].clone()
        orig_w, orig_h = original_resolution[i].int().tolist()

        if not (target_w == orig_w and target_h == orig_h):
            crop_size = min(orig_w, orig_h)
            crop_offset_x = (orig_w - crop_size) // 2
            crop_offset_y = (orig_h - crop_size) // 2

            K[0, 2] -= crop_offset_x
            K[1, 2] -= crop_offset_y

            scale_x = target_w / crop_size
            scale_y = target_h / crop_size

            K[0, 0] *= scale_x
            K[1, 1] *= scale_y
            K[0, 2] *= scale_x
            K[1, 2] *= scale_y

        adjusted_intrinsics[i] = K

    if not is_batch:
        adjusted_intrinsics = adjusted_intrinsics.squeeze(0)

    return adjusted_intrinsics
