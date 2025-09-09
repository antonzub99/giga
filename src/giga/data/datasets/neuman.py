import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
import torchvision.transforms.v2 as tvtransforms
from jaxtyping import Float
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

from giga.geometry import SO3

from ..extimage import ExtendedImageBatch
from .config import DatasetConfig, register_dataset


@dataclass
class NeumanDatasetConfig(DatasetConfig):
    """Configuration for Neuman dataset."""

    _target_: str = "giga.data.datasets.neuman.NeumanDataset"
    rig_size: int = 1
    crop_size: tuple[int, int] = (700, 700)

    @classmethod
    def load(cls: Type["NeumanDatasetConfig"], config_path: str | Path) -> "NeumanDatasetConfig":
        """Load configuration from YAML file."""
        config = OmegaConf.load(config_path)
        config.data_dir = Path(config.data_dir)
        if "texture_dir" in config and config.texture_dir is not None:
            config.texture_dir = Path(config.texture_dir)
        return cls(**config)


@register_dataset("neuman")
class NeumanDataset(Dataset):
    __config_class__ = NeumanDatasetConfig

    def __init__(
        self,
        config: NeumanDatasetConfig,
        characters: list[str],
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.characters = characters  # always a list with a single character
        self.split = split

        self.annotations = self._load_smplx_annotations()
        self.cameras = self._load_camera_calibrations()

    def _load_smplx_annotations(self) -> dict[str, Any]:
        """
        We use Neural Localizer Fields to produce SMPL-X annotations for the Neuman dataset.
        """
        timesteps_list = []
        shape_list = []
        expression_list = []
        body_pose_list = []
        hand_pose_list = []
        head_pose_list = []
        pelvis_rotation_list = []
        global_translation_list = []
        for char_name in self.characters:
            smplx_annots = np.load(self.config.data_dir / char_name / "smplx_nlf" / "smplx.npz", allow_pickle=True)
            frame_indices = np.loadtxt(self.config.data_dir / char_name / "smplx_nlf" / "frame_ids.txt", dtype=int)

            if self.config.timesteps[0] == -1:
                sampled_timesteps = frame_indices
            else:
                sampled_timesteps = frame_indices[self.config.timesteps]

            timesteps_list.append(torch.from_numpy(sampled_timesteps))
            shape = torch.from_numpy(smplx_annots["shape"][sampled_timesteps])
            shape_list.append(shape)
            expression_list.append(torch.zeros_like(shape))
            pose = torch.from_numpy(smplx_annots["pose"][sampled_timesteps])
            pelvis_rotation_list.append(pose[:, :3])
            body_pose_list.append(pose[:, 3:66])
            head_pose_list.append(pose[:, 66:75])
            hand_pose_list.append(pose[:, 75:])
            global_translation_list.append(torch.from_numpy(smplx_annots["global_translation"][sampled_timesteps]))

        return {
            "timesteps": torch.stack(timesteps_list),
            "shape": torch.stack(shape_list),
            "expression": torch.stack(expression_list),
            "body_pose": torch.stack(body_pose_list),
            "head_pose": torch.stack(head_pose_list),
            "hand_pose": torch.stack(hand_pose_list),
            "pelvis_rotation": torch.stack(pelvis_rotation_list),
            "global_translation": torch.stack(global_translation_list),
        }

    def _load_camera_calibrations(self) -> list[dict[str, Any]]:
        calibrations = []
        for char_name in self.characters:
            _intrinsics_data = (self.config.data_dir / char_name / "sparse" / "cameras.txt").read_text().strip().splitlines()
            intrinsics_data = _intrinsics_data[3].split(" ")
            cam_h, cam_w = int(intrinsics_data[2]), int(intrinsics_data[3])
            focal_x, focal_y = float(intrinsics_data[4]), float(intrinsics_data[5])
            principal_x, principal_y = float(intrinsics_data[6]), float(intrinsics_data[7])
            intrinsics = torch.tensor([[focal_x, 0, principal_x], [0, focal_y, principal_y], [0, 0, 1]], dtype=torch.float32)
            resolution = torch.tensor([cam_w, cam_h], dtype=torch.float32)

            _image_data = (self.config.data_dir / char_name / "sparse" / "images.txt").read_text().strip().splitlines()
            num_frames, mean_pts_per_image = re.findall(r"[-+]?\d*\.\d+|\d+", _image_data[3])
            num_frames = int(num_frames)

            extrinsics_list = []

            for idx in range(num_frames):
                data = _image_data[4 + idx * 2].split()
                qw, qx, qy, qz, tx, ty, tz = map(float, data[1:8])

                quaternion = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
                w2v_rotation = SO3.quaternion_to_matrix(quaternion)
                w2v_translation = torch.tensor([tx, ty, tz], dtype=torch.float32) / 1000.0  # Convert mm to m
                extrinsics = torch.eye(4, dtype=torch.float32)
                extrinsics[:3, :3] = w2v_rotation
                extrinsics[:3, 3] = w2v_translation

                extrinsics_list.append(extrinsics)

            extrinsics_list = torch.stack(extrinsics_list, dim=0)  # (num_frames, 4, 4)
            intrinsics_list = intrinsics.unsqueeze(0).expand(num_frames, -1, -1)  # (num_frames, 3, 3)
            resolution_list = resolution.unsqueeze(0).expand(num_frames, -1)  # (num_frames, 2)

            calibrations.append(
                {
                    "intrinsics": intrinsics_list,
                    "extrinsics": extrinsics_list,
                    "resolution": resolution_list,
                }
            )

        return calibrations

    def _get_character_images(self, char_idx: int, frame_idx: int, crop: bool) -> ExtendedImageBatch:
        image = self.load_image(char_idx, frame_idx)
        mask = self.load_mask(char_idx, frame_idx)
        bbox_xyxy = tvops.masks_to_boxes((mask * 255).long()).long()

        # Ensure bbox is at least crop_size but doesn't exceed image boundaries
        _, h, w = image.shape
        crop_w, crop_h = self.config.crop_size

        x1, y1, x2, y2 = bbox_xyxy[0]
        current_w = x2 - x1
        current_h = y2 - y1

        # Expand bbox to meet minimum crop size requirements
        if current_w < crop_w:
            expand_w = crop_w - current_w
            x1 = max(0, x1 - expand_w // 2)
            x2 = min(w, x1 + crop_w)
            x1 = max(0, x2 - crop_w)

        if current_h < crop_h:
            expand_h = crop_h - current_h
            y1 = max(0, y1 - expand_h // 2)
            y2 = min(h, y1 + crop_h)
            y1 = max(0, y2 - crop_h)

        bbox_xyxy[0] = torch.tensor([x1, y1, x2, y2], dtype=bbox_xyxy.dtype)

        bbox_yxhw = bbox_xyxy.clone()
        bbox_yxhw[:, 0] = bbox_xyxy[:, 1]
        bbox_yxhw[:, 1] = bbox_xyxy[:, 0]
        bbox_yxhw[:, 2] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]  # Convert to xyhw format
        bbox_yxhw[:, 3] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]  # Convert to xyhw format

        rgba_image = torch.cat([image * mask, mask], dim=0)  # (4, h, w)
        extrinsics = self.cameras[char_idx]["extrinsics"][frame_idx]
        intrinsics = self.cameras[char_idx]["intrinsics"][frame_idx]
        extimages = ExtendedImageBatch(
            image=rgba_image.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0),
        )
        if crop:
            extimages = extimages.crop(bbox_yxhw.squeeze(0))

        return extimages

    def load_image(
        self,
        char_idx: int,
        frame_idx: int,
    ) -> Float[Tensor, "3 h w"]:
        image = Image.open(self.config.data_dir / self.characters[char_idx] / "images" / f"{frame_idx:05d}.png").convert("RGB")
        tv_image = tvtransforms.functional.to_image(image)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_mask(
        self,
        char_idx: int,
        frame_idx: int,
    ) -> Float[Tensor, "1 h w"]:
        mask = Image.open(self.config.data_dir / self.characters[char_idx] / "segmentations" / f"{frame_idx:05d}.png").convert("L")
        mask = (np.array(ImageOps.invert(mask), dtype=np.uint8) > 0).astype(np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel_erode), kernel_dilate)

        tv_mask = tvtransforms.functional.to_image(mask[..., None] * 255)
        return tvtransforms.functional.to_dtype(tv_mask, dtype=torch.float32, scale=True)

    def load_texture(self, char_idx: int) -> Float[Tensor, "3 h w"]:
        texture_path = self.config.data_dir / f"{self.characters[char_idx]}" / "texture_soft.jpg"
        if texture_path.exists():
            texture = Image.open(texture_path).convert("RGB").resize(self.config.texture_resolution, Image.Resampling.BILINEAR)
            tv_texture = tvtransforms.functional.to_image(texture)
            tv_texture = tvtransforms.functional.to_dtype(tv_texture, dtype=torch.float32, scale=True)
        else:
            tv_texture = torch.ones(3, *self.config.texture_resolution, dtype=torch.float32) * 0.5
        return tv_texture

    def __len__(self) -> int:
        return len(self.characters) * self.annotations["timesteps"].shape[1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        char_idx = 0  # always a single character
        source_views = self._get_character_images(char_idx, idx, crop=False)
        target_views = self._get_character_images(char_idx, idx, crop=False)

        motion = {
            "gender": "neutral",
            "flat_hand": False,
            "hand_pose_dim": 45,
            "pelvis_translation": torch.zeros(1, 3, dtype=torch.float32),
            "global_rotation": torch.zeros(1, 3, dtype=torch.float32),
        }

        for k, v in self.annotations.items():
            if k != "timesteps":
                motion[k] = v[char_idx, idx].unsqueeze(0)
            else:
                continue

        patch_data = {
            "center": None,
            "radius": None,
            "size": None,
        }

        texture = self.load_texture(char_idx)

        return {"input_views": source_views, "target_views": target_views, "motion": motion, "patch_data": patch_data, "texture": texture}
