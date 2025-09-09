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
# - Extended to support intrinsics (xyz) and extrinsics (XYZ) Euler angles

import torch
from jaxtyping import Float
from torch import Tensor


def euler_to_matrix(euler: Float[Tensor, "... 3"], convention: str = "XYZ") -> Float[Tensor, "... 3 3"]:
    # Create base rotation matrix and reshape it to have the same number of dims as euler
    R = torch.eye(3, device=euler.device, dtype=euler.dtype)
    R = R.view(*([1] * (euler.dim() - 1)), 3, 3)

    # Expand to match the input shape
    output_shape = euler.shape[:-1] + (3, 3)
    R = R.expand(output_shape)

    extrinsic = convention.islower()
    convention = convention.lower()

    for i, axis in enumerate(convention):
        angle = euler[..., i]  # Extract the angle for the current axis
        R_axis = rotation_matrix(angle, axis)  # Compute the rotation matrix for this axis
        R = torch.matmul(R, R_axis) if not extrinsic else torch.matmul(R_axis, R)

    return R


def matrix_to_euler(matrix: Float[Tensor, "... 3 3"], convention: str = "XYZ") -> Float[Tensor, "... 3"]:
    is_extrinsic = convention.islower()

    if is_extrinsic:
        convention = convention.upper()
        convention = convention[::-1]

    eulers = matrix_to_euler_angles(matrix, convention)

    if is_extrinsic:
        a, b, c = eulers.unbind(-1)
        eulers = torch.stack([c, b, a], -1)

    return eulers


def rotation_matrix(angle: Float[Tensor, "..."], axis: str) -> Float[Tensor, "... 3 3"]:
    cos, sin = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(cos, device=cos.device, dtype=cos.dtype)
    ones = torch.ones_like(cos, device=cos.device, dtype=cos.dtype)

    if axis == "x":
        mat = torch.stack(
            [
                torch.stack([ones, zero, zero], dim=-1),
                torch.stack([zero, cos, -sin], dim=-1),
                torch.stack([zero, sin, cos], dim=-1),
            ],
            dim=-2,
        )
    elif axis == "y":
        mat = torch.stack(
            [
                torch.stack([cos, zero, sin], dim=-1),
                torch.stack([zero, ones, zero], dim=-1),
                torch.stack([-sin, zero, cos], dim=-1),
            ],
            dim=-2,
        )
    elif axis == "z":
        mat = torch.stack(
            [
                torch.stack([cos, -sin, zero], dim=-1),
                torch.stack([sin, cos, zero], dim=-1),
                torch.stack([zero, zero, ones], dim=-1),
            ],
            dim=-2,
        )

    return mat


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        x = matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        eps = torch.finfo(x.dtype).eps
        central_angle = torch.asin(torch.clamp(x, -1.0 + eps, 1.0 - eps))
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")
