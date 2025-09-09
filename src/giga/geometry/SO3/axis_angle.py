import torch
from jaxtyping import Float
from torch import Tensor

from .quaternions import matrix_to_quaternion, quaternion_to_axis_angle, quaternion_to_matrix


def axisangle_to_quaternion(axis_angle: Float[Tensor, "... 3"]) -> Float[Tensor, "... 4"]:
    # This function comes from pytorch3d, licensed under BSD 3-Clause License.
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
    return torch.cat([torch.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], dim=-1)


def axisangle_to_matrix(axisangle: Float[Tensor, "... 3"]) -> Float[Tensor, "... 3 3"]:
    return quaternion_to_matrix(axisangle_to_quaternion(axisangle))


def matrix_to_axisangle(R: Float[Tensor, "... 3 3"]) -> Float[Tensor, "... 3"]:
    return quaternion_to_axis_angle(matrix_to_quaternion(R))
