# mostly borrowed and adapted to work with torch.Tensors instead of np.ndarrays from
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/cameras/camera_utils.py

from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from giga.geometry import SO3

EPSILON = torch.finfo(torch.float32).eps * 4.0

# here everywhere we need poses, not extrinsics! Do not forget to invert extrinsics before using these functions.


def quaternion_slerp(
    start: Float[Tensor, " 4 "],
    end: Float[Tensor, " 4 "],
    fraction: float,
    spin: int = 0,
    shortestpath: bool = True,
) -> Float[Tensor, " 4 "]:
    # start and end are already unit-length
    if fraction == 0.0:
        return start
    elif fraction == 1.0:
        return end

    d = torch.sum(start * end, dim=-1, keepdim=True)
    if torch.abs(torch.abs(d) - 1.0) < EPSILON:
        d = -d
        start = torch.neg(start)
    angle = torch.acos(d) + spin * torch.pi
    if torch.abs(angle) < EPSILON:
        return start

    isin = 1.0 / torch.sin(angle)
    start *= torch.sin((1.0 - fraction) * angle) * isin
    end *= torch.sin(fraction * angle) * isin
    return start + end


def interpolate_poses(
    pose_start: Float[Tensor, " 4 4 "],
    pose_end: Float[Tensor, " 4 4 "],
    steps: int = 10,
) -> Float[Tensor, "steps 4 4"]:
    q_start = SO3.matrix_to_quaternion(pose_start[:3, :3])
    q_end = SO3.matrix_to_quaternion(pose_end[:3, :3])

    ts = torch.linspace(0, 1, steps, device=pose_start.device)
    interp = SO3.slerp(q_start, q_end, ts.unsqueeze(-1)).squeeze(-2)
    poses = SO3.quaternion_to_matrix(interp)
    tvecs = torch.lerp(pose_start[:3, 3], pose_end[:3, 3], ts.unsqueeze(-1))

    t4x4 = torch.eye(4, device=pose_start.device).unsqueeze(0).repeat(steps, 1, 1)
    t4x4[:, :3, :3] = poses
    t4x4[:, :3, 3] = tvecs

    return t4x4


def interpolate_intrinsics(
    intrinsics_start: Float[Tensor, " 3 3 "],
    intrinsics_end: Float[Tensor, " 3 3 "],
    steps: int = 10,
) -> Float[Tensor, "steps 3 3"]:
    ts = torch.linspace(0, 1, steps, device=intrinsics_start.device)
    intrinsics = torch.lerp(intrinsics_start, intrinsics_end, ts.unsqueeze(-1).unsqueeze(-1))
    return intrinsics


def get_ordered_cameras(
    poses: Float[Tensor, "N 4 4"],
    intrinsics: Float[Tensor, "N 3 3"],
) -> tuple[Float[Tensor, "N 4 4"], Float[Tensor, "N 3 3"]]:
    """
    Sort cameras based on euclidian distance between poses.
    """

    ordered_poses = poses[0, None]
    ordered_intrinsics = intrinsics[0, None]

    poses = poses[1:]
    intrinsics = intrinsics[1:]
    num_steps = poses.shape[0]

    for _ in range(num_steps):
        distances = torch.norm(ordered_poses[-1][:, 3] - poses[:, :, 3], dim=-1)
        idx = torch.argmin(distances)
        ordered_poses = torch.cat([ordered_poses, poses[idx][None]], dim=0)
        ordered_intrinsics = torch.cat([ordered_intrinsics, intrinsics[idx][None]], dim=0)
        poses = torch.cat([poses[:idx], poses[idx + 1 :]], dim=0)
        intrinsics = torch.cat([intrinsics[:idx], intrinsics[idx + 1 :]], dim=0)

    return ordered_poses, ordered_intrinsics


def get_many_interpolated_poses(
    poses: Float[Tensor, "N 4 4"],
    intrinsics: Float[Tensor, "N 3 3"],
    steps: int = 10,
    order_poses: bool = False,
) -> tuple[Float[Tensor, "steps N 4 4"], Float[Tensor, "steps N 3 3"]]:
    """
    Interpolate between all pairs of poses and intrinsics.
    """
    trajectory = []
    intrinsics_trajectory = []

    if order_poses:
        poses, intrinsics = get_ordered_cameras(poses, intrinsics)

    for idx in range(poses.shape[0] - 1):
        pose_start = poses[idx]
        pose_end = poses[idx + 1]
        intrinsics_start = intrinsics[idx]
        intrinsics_end = intrinsics[idx + 1]

        interpolated_poses = interpolate_poses(pose_start, pose_end, steps)
        interpolated_intrinsics = interpolate_intrinsics(intrinsics_start, intrinsics_end, steps)

        trajectory.append(interpolated_poses)
        intrinsics_trajectory.append(interpolated_intrinsics)

    trajectory = torch.cat(trajectory, dim=0)
    intrinsics_trajectory = torch.cat(intrinsics_trajectory, dim=0)

    return trajectory, intrinsics_trajectory


def rotmat_between(
    start: Float[Tensor, "3 3"],
    end: Float[Tensor, "3 3"],
) -> Float[Tensor, "3 3"]:
    start = start / torch.norm(start, dim=-1, keepdim=True)
    end = end / torch.norm(end, dim=-1, keepdim=True)

    v = torch.cross(start, end, dim=-1)
    eps = 1e-6

    if torch.sum(torch.abs(v)) < eps:
        x = (
            torch.tensor([1.0, 0.0, 0.0], device=start.device)
            if torch.abs(start[0]) < eps
            else torch.tensor([0.0, 1.0, 0.0], device=start.device)
        )
        v = torch.cross(start, x, dim=-1)

    v = v / torch.norm(v, dim=-1, keepdim=True)
    skewsym = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=start.device)
    theta = torch.acos(torch.clip(torch.dot(start, end), -1.0, 1.0))

    return torch.eye(3, device=start.device) + torch.sin(theta) * skewsym + (1 - torch.cos(theta)) * torch.dot(skewsym, skewsym)


def focus_of_attention(
    poses: Float[Tensor, "N 4 4"],
    initial_focus: Float[Tensor, " 3 "],
) -> Float[Tensor, " 3 "]:
    active_directions = -poses[..., :3, 2:3]
    active_origins = poses[..., :3, 3:4]

    focus_pt = initial_focus
    active = torch.sum(active_directions[..., 0] * (focus_pt - active_origins[..., 0]), dim=-1) > 0
    done = False
    while torch.sum(active.to(torch.uint8)) > 1 and not done:
        active_origins = active_origins[active]
        active_directions = active_directions[active]

        m = torch.eye(3) - active_directions * active_directions.transpose(-1, -2)
        mt_m = torch.matmul(m, m.transpose(-1, -2))
        focus_pt = torch.inverse(mt_m.mean(dim=0)) @ (mt_m @ active_origins).mean(dim=0)[:, 0]
        active = torch.sum(active_directions[..., 0] * (focus_pt - active_origins[..., 0]), dim=-1) > 0
        if active.all():
            done = True
    return focus_pt


def auto_orient_and_center_poses(
    poses: Float[Tensor, "N 4 4"],
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "manual", "none"] = "poses",
    initial_focus: Float[Tensor, " 3 "] | None = None,
) -> tuple[Float[Tensor, "N 4 4"], Float[Tensor, "4 4"]]:
    origins = poses[..., :3, 3]
    mean_origin = origins.mean(dim=0, keepdim=True)
    tvec_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "manual":
        assert initial_focus is not None, "Initial focus must be provided for manual centering."
        translation = initial_focus
    else:
        translation = torch.zeros(3, device=poses.device)

    if method == "pca":
        eigvals, eigvecs = torch.linalg.eig(tvec_diff.T @ tvec_diff)
        eigvecs = torch.flip(eigvecs, dims=(-1,))

        if torch.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        transform = torch.cat([eigvecs, eigvecs @ -translation.unsqueeze(-1)], dim=-1)
        oriented_poses = transform @ poses
        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[..., :3, 1], dim=0)
        up = up / torch.norm(up)
        if method == "vertical":
            x_axis = poses[..., :3, 0]
            _, S, Vh = torch.linalg.svd(x_axis, full_matrices=True)
            if S[1] > 0.17 * torch.sqrt(poses.shape[0]):
                up_vertical = Vh[2, :]
                up = up_vertical * torch.sign(torch.dot(up_vertical, up))
            else:
                up = up - Vh[0, :] * torch.dot(up, Vh[0, :])
                up = up / torch.norm(up)
        rotation = rotmat_between(up, torch.tensor([0.0, 0.0, 1.0], device=poses.device))
        transform = torch.cat((rotation, rotation @ -translation.unsqueeze(-1)), dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4, device=poses.device)
        transform[:3, 3] = -translation
        oriented_poses = poses.clone()
    else:
        raise ValueError(f"Unknown orientation method: {method}")

    return oriented_poses, transform


def orbit_around_moving_actor(
    initial_camera_pose: Float[Tensor, "4 4"],
    initial_intrinsics: Float[Tensor, "3 3"],
    actor_centers: Float[Tensor, "N 3"],
    orbit_radius: float | None = None,
    fixed_y: float | None = None,
    orbit_angle_per_step: float = 0.1,
) -> tuple[Float[Tensor, "N 4 4"], Float[Tensor, "N 3 3"]]:
    """
    Generate an orbiting camera trajectory around a moving actor.

    Args:
        initial_camera_pose: Initial camera pose (4x4 transformation matrix)
        initial_intrinsics: Initial camera intrinsics (3x3 matrix)
        actor_centers: 3D center coordinates of the actor for each timestep (N x 3)
        orbit_radius: Fixed distance from actor center. If None, computed from initial camera position
        fixed_y: Fixed Y coordinate for camera. If None, uses initial camera Y coordinate
        orbit_angle_per_step: Angle increment per step for orbiting (in radians)

    Returns:
        Tuple of (camera poses, intrinsics) for each timestep
    """
    device = initial_camera_pose.device
    num_steps = actor_centers.shape[0]

    initial_camera_pos = initial_camera_pose[:3, 3]
    initial_camera_rotation = initial_camera_pose[:3, :3]

    initial_up_vector = initial_camera_rotation[:, 1]
    initial_up_vector = initial_up_vector / torch.norm(initial_up_vector)

    # Compute orbit radius from initial position if not provided
    if orbit_radius is None:
        initial_actor_center = actor_centers[0]
        orbit_radius = torch.norm(initial_camera_pos - initial_actor_center).item()

    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)

    # Use initial camera Y coordinate if not provided
    if fixed_y is None:
        fixed_y = initial_camera_pos[1].item()

    # Compute initial angle in ZX plane relative to first actor position
    initial_actor_center = actor_centers[0]
    initial_offset = initial_camera_pos - initial_actor_center
    initial_angle = torch.atan2(initial_offset[0], initial_offset[2]).item()

    camera_poses = []
    intrinsics_list = []

    for i in range(num_steps):
        actor_center = actor_centers[i]
        current_angle = torch.tensor(initial_angle + i * orbit_angle_per_step)

        camera_x = actor_center[0] + orbit_radius * torch.sin(current_angle)
        camera_z = actor_center[2] + orbit_radius * torch.cos(current_angle)
        camera_y = torch.tensor(fixed_y, device=device)

        camera_pos = torch.stack([camera_x, camera_y, camera_z])

        # Compute camera orientation (look at actor center)
        # Forward direction (negative Z in camera space)
        forward = camera_pos - actor_center
        forward = forward / torch.norm(forward)

        # Use the initial camera's up vector to maintain consistent orientation
        camera_up = world_up

        right = torch.cross(forward, camera_up)
        right = right / torch.norm(right)

        up_corrected = torch.cross(right, forward)
        up_corrected = up_corrected / torch.norm(up_corrected)

        # Rotation matrix (camera-to-world)
        rotation = torch.stack([right, up_corrected, -forward], dim=1)

        pose = torch.eye(4, device=device)
        pose[:3, :3] = rotation
        pose[:3, 3] = camera_pos

        camera_poses.append(pose)
        intrinsics_list.append(initial_intrinsics.clone())

    camera_poses = torch.stack(camera_poses, dim=0)
    intrinsics_trajectory = torch.stack(intrinsics_list, dim=0)

    return camera_poses, intrinsics_trajectory


def circle_around_rig(
    poses: Float[Tensor, "N 4 4"],
    intrinsics: Float[Tensor, "N 3 3"],
    key_cameras: list[int],
    steps_per_transition: int = 5,
    order_poses: bool = False,
    loop: bool = True,
) -> tuple[Float[Tensor, "steps 4 4"], Float[Tensor, "steps 3 3"]]:
    """
    Generate a circular trajectory around the rig based on key camera poses.
    """

    if len(key_cameras) < 2:
        raise ValueError("At least two key cameras are required for circular trajectory.")

    key_poses = poses[key_cameras]
    key_intrinsics = intrinsics[key_cameras]

    if order_poses:
        key_poses, key_intrinsics = get_ordered_cameras(key_poses, key_intrinsics)

    if loop:
        key_poses = torch.cat([key_poses, key_poses[:1]], dim=0)
        key_intrinsics = torch.cat([key_intrinsics, key_intrinsics[:1]], dim=0)

    trajectory, intrinsics_trajectory = get_many_interpolated_poses(
        key_poses, key_intrinsics, steps=steps_per_transition, order_poses=order_poses
    )

    if loop:
        trajectory = torch.cat([trajectory, trajectory[:1]], dim=0)
        intrinsics_trajectory = torch.cat([intrinsics_trajectory, intrinsics_trajectory[:1]], dim=0)

    return trajectory, intrinsics_trajectory
