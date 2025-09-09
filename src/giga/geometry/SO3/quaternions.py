# This file is derived from PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
# Licensed under the BSD License:
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided the conditions of the BSD License are met.
#
# Modifications in this file:
# - Removed docstrings
# - Added proper type hints with shape information
#
# In addition, the Quaternion->Axis-Angle conversion has been adapted from Kornia (https://github.com/kornia/kornia)
# Which is licensed under the Apache License 2.0

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


def quaternion_multiply(q1: Float[Tensor, "... 4"], q2: Float[Tensor, "... 4"]) -> Float[Tensor, "... 4"]:
    r1, i1, j1, k1 = q1.unbind(-1)
    r2, i2, j2, k2 = q2.unbind(-1)

    r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

    return torch.stack([r, i, j, k], dim=-1)


def quaternion_inverse(q: Float[Tensor, "... 4"]) -> Float[Tensor, "... 4"]:
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


def quaternion_log(q: Float[Tensor, "... 4"]) -> Float[Tensor, "... 3"]:
    # Input: unit quaternion (w, x, y, z); Output: 3D vector
    w = q[..., 0]
    v = q[..., 1:]
    norm_v = v.norm(dim=-1, keepdim=True)
    theta = torch.acos(torch.clamp(w, -1 + 1e-7, 1 - 1e-7)).unsqueeze(-1)
    factor = torch.where(norm_v > 1e-6, theta / norm_v, torch.zeros_like(norm_v))
    return factor * v


def quaternion_exp(v: Float[Tensor, "... 3"]) -> Float[Tensor, "... 4"]:
    theta = v.norm(dim=-1, keepdim=True)
    w = torch.cos(theta)
    factor = torch.where(theta > 1e-6, torch.sin(theta) / theta, torch.ones_like(theta))
    xyz = factor * v
    return torch.cat([w, xyz], dim=-1)


def quaternion_distance(q1: Float[Tensor, "... 4"], q2: Float[Tensor, "... 4"]) -> Float[Tensor, "..."]:
    dot = torch.abs((q1 * q2).sum(dim=-1))
    return torch.acos(torch.clamp(2 * dot**2 - 1, -1, 1))


def quaternion_to_matrix(quaternion: Float[Tensor, "... 4"]) -> Float[Tensor, "... 3 3"]:
    r, i, j, k = torch.unbind(quaternion, dim=-1)
    two_s = 2.0 / (quaternion * quaternion).sum(dim=-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternion.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: Float[Tensor, "... 3 3"]) -> Float[Tensor, "... 4"]:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: Float[Tensor, "... 4"]) -> Float[Tensor, "... 3"]:
    cos_theta = quaternions[..., 0]
    q1, q2, q3 = quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    sin_sq_theta = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta = torch.sqrt(sin_sq_theta)

    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_sq_theta > 0.0, k_pos, k_neg)

    axis_angle = torch.zeros_like(quaternions)[..., :3]
    axis_angle[..., 0] = q1 * k
    axis_angle[..., 1] = q2 * k
    axis_angle[..., 2] = q3 * k
    return axis_angle


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def project_to_quat_hemisphere(quaternion: Float[Tensor, "... 4"]) -> Float[Tensor, "... 4"]:
    quaternion = F.normalize(quaternion, dim=-1)
    return standardize_quaternion(quaternion)
