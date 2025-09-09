import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as tvtransforms
from jaxtyping import Float
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..extimage import ExtendedImage, ExtendedImageBatch
from .config import DatasetConfig, register_dataset
from .utils import center_crop_bbox, sample_character_patches


@dataclass
class MVHDatasetConfig(DatasetConfig):
    """Configuration for MVH dataset."""

    _target_: str = "giga.data.datasets.mvh.MVHDataset"
    rig_size: int = 16


@register_dataset("mvh")
class MVHDataset(Dataset):
    """Multi-View Human Dataset with cleaner implementation."""

    __config_class__ = MVHDatasetConfig

    TVEC_NORMALIZATION = 12 / 6500  # Camera translation normalization factor
    RESCALE_FACTOR = 0.5  # Image scale factor

    def __init__(
        self,
        config: MVHDatasetConfig,
        characters: list[str],
        split: str = "train",
    ):
        """Initialize dataset.

        Args:
            config: Dataset configuration
            characters: List of character names (pre-filtered)
            split: Dataset split ("train", "val", "test")
        """
        super().__init__()
        self.config = config
        self.characters = characters
        self.split = split

        # Load annotations once at init
        self.annotations = self._load_smplx_annotations()
        self.cameras = self._load_camera_calibrations()

    @classmethod
    def select_characters(cls, config: MVHDatasetConfig) -> list[str]:
        if config.exclusions_file is not None:
            with open(config.exclusions_file, "r") as f:
                excluded_characters = set(f.read().splitlines())
        else:
            excluded_characters = set()

        available_characters = sorted([d.name for d in config.data_dir.iterdir() if d.is_dir() if d not in excluded_characters])
        selection = config.character_selection

        if isinstance(selection, str):
            assert selection in available_characters, f"Character {selection} not found in dataset."
            return [selection]
        elif len(selection) == 1:
            return available_characters[: selection[0]]
        elif len(selection) == 2:
            return available_characters[selection[0] : selection[1] + 1]
        elif len(selection) == 3:
            return available_characters[selection[0] : selection[1] + 1 : selection[2]]
        else:
            return [available_characters[i] for i in selection]

    def _load_camera_calibrations(self) -> list[dict[str, Any]]:
        """Load camera calibration data for all characters.

        Returns:
            list[dict]: containing camera parameters for each character:
                - intrinsics: Camera intrinsic matrices (N, 3, 3)
                - extrinsics: Camera extrinsic matrices (N, 4, 4)
                - resolution: Image resolutions (N, 2)
                - cam_names: List of camera names
                - camera_scale: Camera scale factor
        """
        calibrations = []

        for char_name in self.characters:
            calib_path = self.config.data_dir / char_name / "camera_extrinsics.json"
            with open(calib_path, "r") as f:
                calibration = json.load(f)

            # Load camera scale
            with open(calib_path.parent / "camera_scale.pkl", "rb") as f:
                camera_scale = torch.as_tensor(np.load(f, allow_pickle=True))

            # Get valid camera names from image directory
            valid_cam_names = [item.name for item in (calib_path.parent / "images_lr").iterdir() if item.is_dir()]
            num_valid_cams = len(valid_cam_names)
            if num_valid_cams < 48:
                logger.info(f"Found {num_valid_cams} cameras for {char_name} (expected 48)")

            # Extract camera names from calibration
            cam_ids = calibration.keys()
            cam_names = [
                cam_id.split(".")[0].split("_")[-1] for cam_id in cam_ids if cam_id.split(".")[0].split("_")[-1] in valid_cam_names
            ]

            num_cams = len(cam_names)
            intrinsics = torch.zeros(num_cams, 3, 3)
            extrinsics = torch.zeros(num_cams, 4, 4)
            extrinsics[:] = torch.eye(4)
            resolution = torch.zeros(num_cams, 2)

            for cam_idx, cam_name in enumerate(cam_names):
                cam_data = calibration[f"1_{cam_name}.png"]

                # Extrinsics
                extrinsics[cam_idx, :3, :3] = torch.tensor(cam_data["rotation"], dtype=torch.float32)
                extrinsics[cam_idx, :3, 3] = torch.tensor(cam_data["translation"], dtype=torch.float32) * self.TVEC_NORMALIZATION

                # Intrinsics
                focal_length = torch.tensor(cam_data["focal_length"]) * self.RESCALE_FACTOR
                image_center = cam_data["image_center"]
                if len(image_center) < 4:
                    image_center += [image_center[0] * 2 + 1, image_center[1] * 2 + 1]
                image_center = torch.tensor(image_center) * self.RESCALE_FACTOR

                intrinsics[cam_idx] = torch.tensor(
                    [[focal_length[0], 0, image_center[0]], [0, focal_length[1], image_center[1]], [0, 0, 1]], dtype=torch.float32
                )

                resolution[cam_idx] = image_center[-2:]

            calibrations.append(
                {
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "resolution": resolution,
                    "cam_names": cam_names,
                    "camera_scale": camera_scale,
                }
            )

        return calibrations

    def _load_smplx_annotations(self) -> dict[str, Tensor]:
        """Load SMPL-X annotations for all characters.

        Returns:
            dict: containing stacked tensors for all characters:
                - timesteps: Frame indices (num_chars, num_frames)
                - shapes: Shape parameters (num_chars, 10)
                - expression: Expression parameters (num_chars, num_frames, 10)
                - full_pose: Pose parameters (num_chars, num_frames, 96)
                - global_rotation: Global rotation (num_chars, num_frames, 3)
                - global_translation: Global translation (num_chars, num_frames, 3)
        """
        timesteps_list = []
        betas_list = []
        expression_list = []
        full_pose_list = []
        global_rot_list = []
        global_trans_list = []

        for char_name in self.characters:
            annots_dir = self.config.data_dir / char_name / "smplx" / "smpl"
            annots_files = sorted(annots_dir.glob("*.json"))
            max_timesteps = len(annots_files)

            if max_timesteps == 0:
                logger.warning(f"No annotation files found for character {char_name}")
                continue

            if isinstance(self.config.timesteps[0], int) and len(self.config.timesteps) == 1:
                if self.config.timesteps[0] == -1:
                    frame_indices = list(range(max_timesteps))
                else:
                    frame_idx = self.config.timesteps[0] % max_timesteps
                    frame_indices = [frame_idx]
                    if frame_idx != self.config.timesteps[0]:
                        logger.info(
                            f"Adjusted timestep {self.config.timesteps[0]} -> {frame_idx} for character {char_name} (max: {max_timesteps})"
                        )
            else:
                frame_indices = [t % max_timesteps for t in self.config.timesteps]
                # Log if any adjustments were made
                adjusted = [(orig, adj) for orig, adj in zip(self.config.timesteps, frame_indices) if orig != adj]
                if adjusted:
                    logger.info(f"Adjusted {len(adjusted)} timesteps for character {char_name} (max: {max_timesteps}): {adjusted}")
            # Load annotations for each frame
            betas = []
            expressions = []
            full_poses = []
            global_rots = []
            global_trans = []

            for frame_idx in frame_indices:
                with open(annots_files[frame_idx], "r") as f:
                    data = json.load(f)[0]

                if not betas:  # Load shape only once per character
                    betas.append(torch.tensor(data["shapes"][0], dtype=torch.float32))

                expressions.append(torch.tensor(data["expression"][0], dtype=torch.float32))
                full_poses.append(torch.tensor(data["poses"][0], dtype=torch.float32))
                global_rots.append(torch.tensor(data["Rh"][0], dtype=torch.float32))
                global_trans.append(torch.tensor(data["Th"][0], dtype=torch.float32))

            timesteps_list.append(torch.tensor(frame_indices))
            betas_list.append(torch.stack(betas).squeeze())
            expression_list.append(torch.stack(expressions))
            full_pose_list.append(torch.stack(full_poses))
            global_rot_list.append(torch.stack(global_rots))
            global_trans_list.append(torch.stack(global_trans))

        return {
            "timesteps": torch.stack(timesteps_list),
            "shape": torch.stack(betas_list),
            "expression": torch.stack(expression_list),
            "full_pose": torch.stack(full_pose_list),
            "global_rotation": torch.stack(global_rot_list),
            "global_translation": torch.stack(global_trans_list),
        }

    def _get_character_images(self, char_idx: int, frame_idx: int, cam_ids: list[str] | list[int]) -> ExtendedImageBatch:
        """Load and process images for a character.

        Returns:
            ExtendedImageBatch: Batch of images with camera parameters
        """
        cam_names = self.cameras[char_idx]["cam_names"]
        strided_cam_ids = np.linspace(0, len(cam_names) - 1, self.config.rig_size, dtype=int)
        if not isinstance(cam_ids, list):
            cam_ids = OmegaConf.to_container(cam_ids)

        if isinstance(cam_ids[0], str):
            cam_ids = np.array([cam_names.index(cam_id) for cam_id in cam_ids if cam_id in cam_names])
        else:
            sampled_num_cams = np.random.choice(cam_ids, 1, replace=False)[0]
            max_num_cams = max(cam_ids)
            cam_ids = np.random.choice(strided_cam_ids, sampled_num_cams, replace=False)
            cam_ids = np.pad(cam_ids, (0, max_num_cams - len(cam_ids)), mode="constant", constant_values=cam_ids[0])

        extimages = []
        for cam_id in cam_ids:
            image = self.load_image(char_idx, frame_idx, cam_id)
            mask = self.load_mask(char_idx, frame_idx, cam_id)
            if image.shape[1:] != mask.shape[1:]:
                image = tvtransforms.functional.resize(image, mask.shape[1:])
            rgba_image = torch.cat([image * mask, mask], dim=0)
            extrinsics = self.cameras[char_idx]["extrinsics"][cam_id]
            intrinsics = self.cameras[char_idx]["intrinsics"][cam_id]

            extimages.append(
                ExtendedImage(
                    image=rgba_image,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                )
            )

        extimages = ExtendedImageBatch.stack(extimages)  # a stack of extended images with associated cameras

        # MVHumanNet images can be safely cropped in the center
        center_bbox = center_crop_bbox(extimages.resolution)
        extimages = extimages.crop(center_bbox)
        extimages = extimages.resize(self.config.image_size)

        return extimages

    def load_image(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Tensor:
        image = Image.open(
            self.config.data_dir
            / self.characters[char_idx]
            / "images_lr"
            / self.cameras[char_idx]["cam_names"][cam_idx]
            / f"{frame_idx:04d}_img.jpg"
        ).convert("RGB")
        tv_image = tvtransforms.functional.to_image(image)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_mask(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Tensor:
        mask = Image.open(
            self.config.data_dir
            / self.characters[char_idx]
            / "fmask_lr"
            / self.cameras[char_idx]["cam_names"][cam_idx]
            / f"{frame_idx:04d}_img_fmask.png"
        ).convert("L")
        mask = (np.array(mask, dtype=np.uint8) > 200).astype(np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel_erode), kernel_dilate)
        tv_image = tvtransforms.functional.to_image(mask[..., None] * 255)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_texture(self, char_idx: int) -> Float[Tensor, "3 h w"]:
        dummy_texture = torch.ones(3, *self.config.texture_resolution, dtype=torch.float32) * 0.5
        if self.config.texture_dir is None:
            tv_texture = dummy_texture
        else:
            texture_path = self.config.texture_dir / f"{self.characters[char_idx]}_apose_texture.jpg"
            if not texture_path.exists():
                tv_texture = dummy_texture
            else:
                texture = Image.open(texture_path).convert("RGB").resize(self.config.texture_resolution, Image.Resampling.BILINEAR)
                tv_texture = tvtransforms.functional.to_image(texture)
                tv_texture = tvtransforms.functional.to_dtype(tv_texture, dtype=torch.float32, scale=True)
        return tv_texture

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.characters) * self.annotations["timesteps"].shape[1]

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns:
            dict: containing:
            - input_views: ExtendedImageBatch of source views
            - target_views: ExtendedImageBatch of target views
            - motion: SMPL-X parameters
        """
        char_idx = idx % len(self.characters)
        frame_idx = idx % self.annotations["timesteps"].shape[1]

        # Get source and target views
        true_timestep = self.annotations["timesteps"][char_idx, frame_idx].item()  # map dataset frame_idx to the actual frame_idx on disk
        source_views = self._get_character_images(char_idx, (true_timestep + 1) * 5, self.config.num_input_cameras)
        target_views = self._get_character_images(char_idx, (true_timestep + 1) * 5, self.config.num_target_cameras)

        # Get motion parameters and split full_pose into components
        motion = {
            "gender": "neutral",  # default for MVHumanNet
            "flat_hand": True,
            "hand_pose_dim": 12,  # default for MVHumanNet
            "pelvis_rotation": torch.zeros(1, 3, dtype=torch.float32),  # default for MVHumanNet
            "pelvis_translation": torch.zeros(1, 3, dtype=torch.float32),  # default for MVHumanNet
        }
        for k, v in self.annotations.items():
            if k == "full_pose":
                # Split full_pose into components
                pose = v[char_idx, frame_idx].unsqueeze(0)
                motion["body_pose"] = pose[:, 3 : 3 * 21 + 3]  # body pose without pelvis
                hand_pose = pose[:, 3 * 21 + 3 : 3 * 21 + 15]  # hand pose
                motion["hand_pose"] = torch.cat([hand_pose, torch.zeros(1, 78, dtype=torch.float32)], dim=1)  # pad to 90 channels
                motion["head_pose"] = pose[:, 3 * 21 + 15 :]  # head pose
            elif k == "shape":
                motion[k] = v[char_idx].unsqueeze(0)
            elif k == "timesteps":
                continue
            else:
                motion[k] = v[char_idx, frame_idx].unsqueeze(0)

        patch_centers, patch_dims = sample_character_patches(
            target_views.image[:, 3:],
            self.config.patch_config["num_patches"],
            self.config.patch_config["patch_size"],
            self.config.patch_config["patch_scale"],
        )
        patch_data = {
            "center": patch_centers,
            "radius": patch_dims,
            "size": self.config.patch_config["patch_size"],
        }
        texture = self.load_texture(char_idx)

        return {
            "name": self.characters[char_idx],
            "input_views": source_views,
            "target_views": target_views,
            "motion": motion,
            "texture": texture,
            "patch_data": patch_data,
        }


@register_dataset("mvhapose")
class MVHAposeDataset(MVHDataset):
    __config_class__ = MVHDatasetConfig

    def __init__(self, config: MVHDatasetConfig, characters: list[str], split: str = "train"):
        """Initialize MVH Apose dataset."""
        super().__init__(config, characters, split)

        self.apose_frame = -5  # Usually A-pose is near the end of the sequence, not always though
        self.apose_annotations = self._load_smplx_apose_annotations()

    def _load_smplx_apose_annotations(self) -> dict[str, Tensor]:
        "Usually A-pose is somewhere at the very end of the sequence - let's use this."

        timesteps_list = []
        betas_list = []
        expression_list = []
        full_pose_list = []
        global_rot_list = []
        global_trans_list = []

        for char_name in self.characters:
            annots_dir = self.config.data_dir / char_name / "smplx" / "smpl"
            annots_files = sorted(annots_dir.glob("*.json"))
            ttl_len = len(annots_files)
            with open(annots_files[self.apose_frame], "r") as f:
                data = json.load(f)[0]

            timesteps_list.append(torch.tensor(ttl_len + self.apose_frame, dtype=torch.int32))
            betas_list.append(torch.tensor(data["shapes"][0], dtype=torch.float32))
            expression_list.append(torch.tensor(data["expression"][0], dtype=torch.float32))
            full_pose_list.append(torch.tensor(data["poses"][0], dtype=torch.float32))
            global_rot_list.append(torch.tensor(data["Rh"][0], dtype=torch.float32))
            global_trans_list.append(torch.tensor(data["Th"][0], dtype=torch.float32))

        return {
            "timesteps": torch.stack(timesteps_list),
            "betas": torch.stack(betas_list),
            "expression": torch.stack(expression_list),
            "full_pose": torch.stack(full_pose_list),
            "global_rot": torch.stack(global_rot_list),
            "global_trans": torch.stack(global_trans_list),
        }

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)

        char_idx = idx % len(self.characters)

        result["apose_motion"] = {
            "gender": "neutral",
            "flat_hand": True,
            "hand_pose_dim": 12,
            "pelvis_rotation": torch.zeros(1, 3, dtype=torch.float32),
            "pelvis_translation": torch.zeros(1, 3, dtype=torch.float32),
            "shape": self.apose_annotations["betas"][char_idx].unsqueeze(0),
            "expression": self.apose_annotations["expression"][char_idx].unsqueeze(0),
            "body_pose": self.apose_annotations["full_pose"][char_idx, 3:66].unsqueeze(0),
            "hand_pose": torch.cat(
                [self.apose_annotations["full_pose"][char_idx, 66:78], torch.zeros(78, dtype=torch.float32)], dim=0
            ).unsqueeze(0),
            "head_pose": self.apose_annotations["full_pose"][char_idx, 78:].unsqueeze(0),
            "global_rotation": self.apose_annotations["global_rot"][char_idx].unsqueeze(0),
            "global_translation": self.apose_annotations["global_trans"][char_idx].unsqueeze(0),
        }

        apose_idx = self.apose_annotations["timesteps"][char_idx]
        apose_frame = (apose_idx + 1) * 5  # map to the actual frame_idx on disk
        result["apose_views"] = self._get_character_images(char_idx, apose_frame, self.config.num_input_cameras)

        return result
