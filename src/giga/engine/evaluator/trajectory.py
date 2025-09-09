from typing import Literal

import torch
from jaxtyping import Float
from lightning import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils import base as mng_base
from ..utils import camera as mng_camera
from .utils import adjust_intrinsics_for_crop_and_resize


def up_vector_to_world_transform(up: str) -> Float[Tensor, "4 4"]:
    if up == "y" or up == "+y":
        transform = torch.eye(4, device="cpu", dtype=torch.float32)
    elif up == "z" or up == "+z":
        transform = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], device="cpu", dtype=torch.float32)
    elif up == "-z":
        transform = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], device="cpu", dtype=torch.float32)
    else:
        transform = torch.eye(4, device="cpu", dtype=torch.float32)
    return transform


@torch.inference_mode()
def find_actor_center(
    model: LightningModule,
    dataloader: DataLoader,
    world_transform: Float[Tensor, "4 4"],
    max_steps: int | None = 50,
) -> Float[Tensor, "N 3"]:
    actor_centers = []

    data_len = len(dataloader.dataset)
    for idx, batch in enumerate(dataloader):
        if max_steps is not None and idx >= max_steps:
            break

        smplx_params = mng_base.to_device(batch["motion"][0], model.device)  # pick the first item in the batch
        smplx_outputs = model.mesh_lbs_layer(**smplx_params)
        vertices = smplx_outputs["vertices"].squeeze().cpu()
        actor_center = vertices.mean(dim=0)
        actor_centers.append(actor_center)

    actor_centers = torch.stack(actor_centers, dim=0)
    actor_centers = actor_centers.mean(dim=0, keepdim=True).expand(data_len, -1)
    actor_centers = torch.einsum("ij,bj->bi", world_transform[:3, :3], actor_centers)

    return actor_centers


def resolve_cycles(
    rig_poses: Float[Tensor, "N 4 4"],
    rig_intrinsics: Float[Tensor, "N 3 3"],
    key_ids: list[int],
    num_steps: int,
    data_len: int,
    num_steps_per_transition: int | None = None,
    loop: bool = False,
) -> tuple[Float[Tensor, "N 4 4"], Float[Tensor, "N 3 3"]]:
    num_cycles = data_len // num_steps
    remaining_steps = data_len % num_steps

    if num_cycles > 1:
        steps_per_transition = num_steps // len(key_ids) if num_steps_per_transition is None else num_steps_per_transition

        pose_trajectory_cycle, intrinsics_trajectory_cycle = mng_camera.circle_around_rig(
            rig_poses, rig_intrinsics, key_ids, steps_per_transition=steps_per_transition, loop=loop, order_poses=False
        )

        pose_trajectory = [pose_trajectory_cycle] * num_cycles
        intrinsics_trajectory = [intrinsics_trajectory_cycle] * num_cycles

        if remaining_steps > 0:
            pose_trajectory.append(pose_trajectory_cycle[:remaining_steps])
            intrinsics_trajectory.append(intrinsics_trajectory_cycle[:remaining_steps])

        pose_trajectory = torch.cat(pose_trajectory, dim=0)
        intrinsics_trajectory = torch.cat(intrinsics_trajectory, dim=0)
    else:
        steps_per_transition = num_steps // len(key_ids) if num_steps_per_transition is None else num_steps_per_transition

        pose_trajectory, intrinsics_trajectory = mng_camera.circle_around_rig(
            rig_poses, rig_intrinsics, key_ids, steps_per_transition=steps_per_transition, loop=loop, order_poses=False
        )

        trajectory_len = pose_trajectory.shape[0]
        if trajectory_len > data_len:
            pose_trajectory = pose_trajectory[:data_len]
            intrinsics_trajectory = intrinsics_trajectory[:data_len]
        elif trajectory_len < data_len:
            extra_steps = data_len - trajectory_len
            pose_trajectory = torch.cat([pose_trajectory, pose_trajectory[:extra_steps]], dim=0)
            intrinsics_trajectory = torch.cat([intrinsics_trajectory, intrinsics_trajectory[:extra_steps]], dim=0)

    return pose_trajectory, intrinsics_trajectory


def create_trajectory(
    model: LightningModule,
    dataloader: DataLoader,
    mode: Literal["orbit", "interpolate"],
    up_vector: str = "y",
    num_steps: int = 100,
    num_steps_per_transition: int | None = None,
    loop: bool = False,
    resolution: tuple[int, int] | None = None,
    orbit_radius: float | None = None,
    fixed_y: float | None = None,
    angle_step: float | None = None,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    data_len = len(dataloader.dataset)
    camera_rig = dataloader.dataset.cameras[0]
    rig_extrinsics = camera_rig["extrinsics"]
    rig_intrinsics = camera_rig["intrinsics"]
    rig_resolution = camera_rig.get("resolution", None)
    rig_size = rig_extrinsics.shape[0]

    if rig_intrinsics.ndim == 2:
        rig_intrinsics = rig_intrinsics.unsqueeze(0).expand(rig_size, -1, -1)
    elif rig_intrinsics.shape[0] != rig_size:
        rig_intrinsics = rig_intrinsics[0].unsqueeze(0).expand(rig_size, -1, -1)

    if resolution is None:
        resolution = dataloader.dataset.config.image_size

    rig_intrinsics = adjust_intrinsics_for_crop_and_resize(rig_intrinsics, rig_resolution, resolution)

    if "cam_names" in camera_rig.keys():
        cam_names = camera_rig["cam_names"]
        key_cam_names = dataloader.dataset.config.num_target_cameras
        key_ids = [cam_names.index(name) for name in key_cam_names]
    else:
        cam_names = [f"cam_{idx:03d}" for idx in range(rig_size)]
        key_ids = list(range(rig_size))

    rig_poses = torch.linalg.inv(rig_extrinsics)
    world_transform = up_vector_to_world_transform(up_vector)

    if mode == "orbit":
        centers = find_actor_center(model, dataloader, world_transform)
        start_cam_idx = key_ids[0]
        start_extr = rig_extrinsics[start_cam_idx] @ world_transform.T
        start_pose = torch.linalg.inv(start_extr)
        start_intr = rig_intrinsics[start_cam_idx]

        pose_trajectory, intrinsics_trajectory = mng_camera.orbit_around_moving_actor(
            initial_camera_pose=start_pose,
            initial_intrinsics=start_intr,
            actor_centers=centers,
            orbit_radius=orbit_radius,
            fixed_y=fixed_y,
            orbit_angle_per_step=angle_step,
        )
    elif mode == "interpolate" and num_steps > data_len:
        num_steps = data_len
        steps_per_transition = num_steps // len(key_ids) if num_steps_per_transition is None else num_steps_per_transition

        pose_trajectory, intrinsics_trajectory = mng_camera.circle_around_rig(
            rig_poses, rig_intrinsics, key_ids, steps_per_transition=steps_per_transition, loop=loop, order_poses=False
        )
    elif mode == "interpolate" and num_steps <= data_len:
        pose_trajectory, intrinsics_trajectory = resolve_cycles(
            rig_poses,
            rig_intrinsics,
            key_ids,
            num_steps=num_steps,
            data_len=data_len,
            num_steps_per_transition=num_steps_per_transition,
            loop=loop,
        )

    rig_center = torch.mean(rig_poses[:, :3, 3], dim=0)
    rig_focus = mng_camera.focus_of_attention(rig_poses, rig_center)
    pose_trajectory, _ = mng_camera.auto_orient_and_center_poses(
        pose_trajectory, method="none", center_method="manual", initial_focus=rig_focus
    )

    extr_trajectory = torch.linalg.inv(pose_trajectory)
    extr_trajectory = torch.einsum("bij,jk->bik", extr_trajectory, world_transform)

    return extr_trajectory, intrinsics_trajectory, resolution
