import torch
from jaxtyping import Float
from torch import Tensor


def opengl_projection_mtx(
    intrinsics: Float[Tensor, "b 3 3"],  # B x 3 x 3
    height: int,  # B
    width: int,  # B
    znear: float = 0.01,
    zfar: float = 10000.0,
) -> Float[Tensor, "b 4 4"]:
    """
    Computes the OpenGL projection matrix from camera intrinsics.
    Args:
        intrinsics (Float[Tensor, "b 3 3"]): Camera intrinsics matrix.
        height (int): Height of the image.
        width (int): Width of the image.
        znear (float): Near clipping plane distance.
        zfar (float): Far clipping plane distance.
    Returns:
        Float[Tensor, "b 4 4"]: OpenGL projection matrix.
    """
    fx, fy = intrinsics[..., 0, 0], intrinsics[..., 1, 1]
    cx, cy = intrinsics[..., 0, 2], intrinsics[..., 1, 2]

    tanHalfFovy = height / (2 * fy)
    tanHalfFovx = width / (2 * fx)

    top = znear * tanHalfFovy
    bottom = -top
    right = znear * tanHalfFovx
    left = -right
    z_sign = 1.0

    projection = torch.zeros((*intrinsics.shape[:-2], 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    projection[..., 0, 0] = 2 * znear / (right - left)
    projection[..., 1, 1] = 2 * znear / (top - bottom)
    projection[..., 0, 2] = 2 * (cx - width / 2) / width
    projection[..., 1, 2] = 2 * (cy - height / 2) / height
    projection[..., 2, 2] = z_sign * zfar / (zfar - znear)
    projection[..., 2, 3] = -(zfar * znear) / (zfar - znear)
    projection[..., 3, 2] = z_sign

    return projection
