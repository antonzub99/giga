import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as tvtransforms
from jaxtyping import Float, Int
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..extimage import ExtendedImage, ExtendedImageBatch
from .config import DatasetConfig, register_dataset
from .utils import sample_character_patches


@dataclass
class MVHPPDatasetConfig(DatasetConfig):
    """Configuration for MVH++ dataset."""

    _target_: str = "giga.data.datasets.mvhpp.MVHPPDataset"
    rig_size: int = 16
    depth_dir: Optional[Path] = None  # practically never used
    normals_dir: Optional[Path] = None  # practically never used

    @classmethod
    def load(cls: Type["MVHPPDatasetConfig"], config_path: str | Path) -> "MVHPPDatasetConfig":
        """
        Load configuration from YAML file.
        We redefine it here because depth and normals might be available.
        """
        config = OmegaConf.load(config_path)
        config.data_dir = Path(config.data_dir)
        if "exclusions_file" in config and config.exclusions_file is not None:
            config.exclusions_file = Path(config.exclusions_file)
        if "texture_dir" in config and config.texture_dir is not None:
            config.texture_dir = Path(config.texture_dir)
        if "depth_dir" in config and config.depth_dir is not None:
            config.depth_dir = Path(config.depth_dir)
        if "normals_dir" in config and config.normals_dir is not None:
            config.normals_dir = Path(config.normals_dir)
        return cls(**config)


@register_dataset("mvhpp")
class MVHPPDataset(Dataset):
    __config_class__ = MVHPPDatasetConfig
    MAX_NUM_FRAMES = 60  # Maximum number of frames per character

    def __init__(
        self,
        config: MVHPPDatasetConfig,
        characters: list[str],
        split: str = "train",
    ):
        super().__init__()
        self.config = config
        self.characters = characters
        self.split = split

        self.annotations = self._load_smplx_annotations()
        self.cameras = self._load_camera_calibrations()

    @classmethod
    def select_characters(cls, config: MVHPPDatasetConfig) -> list[str]:
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
        """

        calibrations = []

        for char_name in self.characters:
            calib_dir = self.config.data_dir / char_name / "cameras"

            cam_iter = list(calib_dir.iterdir())
            num_cams = len(cam_iter)

            intrinsics = torch.zeros((num_cams, 3, 3), dtype=torch.float32)
            extrinsics = torch.zeros((num_cams, 4, 4), dtype=torch.float32)
            extrinsics[:] = torch.eye(4)
            resolution = torch.zeros((num_cams, 2), dtype=torch.int32)

            cam_names = []
            for cam_idx, cam_dir in enumerate(cam_iter):
                if not cam_dir.is_dir():
                    continue

                cam_names.append(cam_dir.name)
                cam_data = np.load(cam_dir / "camera.npz")
                intrinsics[cam_idx] = torch.tensor(cam_data["intrinsic"], dtype=torch.float32)
                extrinsics[cam_idx] = torch.cat(
                    [torch.tensor(cam_data["extrinsic"], dtype=torch.float32), torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0
                )
                resolution[cam_idx] = torch.tensor([1024, 1024], dtype=torch.int32)

            calibrations.append({"intrinsics": intrinsics, "extrinsics": extrinsics, "resolution": resolution, "cam_names": cam_names})

        return calibrations

    def _load_smplx_annotations(self) -> dict[str, Float[Tensor, "..."] | Int[Tensor, "..."]]:
        num_characters = len(self.characters)

        config_timesteps = self.config.timesteps
        assert all(isinstance(t, int) for t in config_timesteps), (
            f"Expected timesteps to be a list of integers, got {type(config_timesteps)} with value {config_timesteps}"
        )
        if config_timesteps[0] == -1:
            config_timesteps = list(range(self.MAX_NUM_FRAMES))

        num_timesteps = len(config_timesteps)
        timesteps = torch.zeros((num_characters, num_timesteps), dtype=torch.int32)
        shape = torch.zeros((num_characters, 10), dtype=torch.float32)
        expression = torch.zeros((num_characters, num_timesteps, 10), dtype=torch.float32)
        body_pose = torch.zeros((num_characters, num_timesteps, 63), dtype=torch.float32)
        hand_pose = torch.zeros((num_characters, num_timesteps, 12), dtype=torch.float32)
        head_pose = torch.zeros((num_characters, num_timesteps, 9), dtype=torch.float32)
        pelvis_rotation = torch.zeros((num_characters, num_timesteps, 3), dtype=torch.float32)
        global_rotation = torch.zeros((num_characters, num_timesteps, 3), dtype=torch.float32)
        global_translation = torch.zeros((num_characters, num_timesteps, 3), dtype=torch.float32)

        for char_idx, char_name in enumerate(self.characters):
            annots_dir = self.config.data_dir / char_name / "smplx_params_new"
            annots_files = sorted([f for f in annots_dir.glob("*.npz") if not f.name.startswith("A-")])  # Exclude A-pose file
            max_timesteps = min(len(annots_files), self.MAX_NUM_FRAMES)

            all_timesteps = list(range(max_timesteps))
            filtered_timesteps = [t for t in all_timesteps if t in config_timesteps]
            num_filtered_timesteps = len(filtered_timesteps)
            if num_filtered_timesteps < num_timesteps:
                do_padding = True
            else:
                do_padding = False

            for idx, t in enumerate(filtered_timesteps):
                timesteps[char_idx, idx] = torch.tensor(int(annots_files[t].stem), dtype=torch.int32)
                smplx_labels = np.load(annots_files[t])
                if idx == 0:  # only load once
                    shape[char_idx] = torch.tensor(smplx_labels["betas"], dtype=torch.float32).squeeze()
                expression[char_idx, idx] = torch.tensor(smplx_labels["expression"], dtype=torch.float32).squeeze()

                body_pose[char_idx, idx] = torch.tensor(smplx_labels["body_pose"], dtype=torch.float32).squeeze()
                hand_pose[char_idx, idx] = torch.cat(
                    [
                        torch.tensor(smplx_labels["left_hand_pose"], dtype=torch.float32),
                        torch.tensor(smplx_labels["right_hand_pose"], dtype=torch.float32),
                    ],
                    dim=1,
                ).squeeze()
                head_pose[char_idx, idx] = torch.cat(
                    [
                        torch.tensor(smplx_labels["jaw_pose"], dtype=torch.float32),
                        torch.tensor(smplx_labels["leye_pose"], dtype=torch.float32),
                        torch.tensor(smplx_labels["reye_pose"], dtype=torch.float32),
                    ],
                    dim=1,
                ).squeeze()

                pelvis_rotation[char_idx, idx] = torch.tensor(smplx_labels["global_orient"], dtype=torch.float32).squeeze()
                global_translation[char_idx, idx] = torch.tensor(smplx_labels["transl"], dtype=torch.float32).squeeze()

            if do_padding:
                logger.info(f"Padding enabled for character '{char_name}'.")
                difference = num_timesteps - num_filtered_timesteps
                if difference > num_filtered_timesteps:
                    high = 1
                else:
                    high = difference
                timesteps[char_idx, num_filtered_timesteps:] = timesteps[char_idx, 0:high]
                expression[char_idx, num_filtered_timesteps:] = expression[char_idx, 0:high]
                body_pose[char_idx, num_filtered_timesteps:] = body_pose[char_idx, 0:high]
                hand_pose[char_idx, num_filtered_timesteps:] = hand_pose[char_idx, 0:high]
                head_pose[char_idx, num_filtered_timesteps:] = head_pose[char_idx, 0:high]
                pelvis_rotation[char_idx, num_filtered_timesteps:] = pelvis_rotation[char_idx, 0:high]
                global_translation[char_idx, num_filtered_timesteps:] = global_translation[char_idx, 0:high]

        # After populating, check for NaNs and fix them
        for char_idx in range(num_characters):
            for t_idx in range(1, num_timesteps):  # Start from the second timestep
                # Check for NaNs in any of the pose/motion tensors for the current timestep
                is_nan = (
                    torch.isnan(expression[char_idx, t_idx]).any()
                    or torch.isnan(body_pose[char_idx, t_idx]).any()
                    or torch.isnan(hand_pose[char_idx, t_idx]).any()
                    or torch.isnan(head_pose[char_idx, t_idx]).any()
                    or torch.isnan(pelvis_rotation[char_idx, t_idx]).any()
                    or torch.isnan(global_rotation[char_idx, t_idx]).any()
                    or torch.isnan(global_translation[char_idx, t_idx]).any()
                )

                if is_nan:
                    char_name = self.characters[char_idx]
                    logger.warning(f"NaN found for character '{char_name}' at timestep {t_idx}. Replacing with previous frame's data.")
                    # If NaN is found, copy data from the previous timestep
                    timesteps[char_idx, t_idx] = timesteps[char_idx, t_idx - 1]
                    expression[char_idx, t_idx] = expression[char_idx, t_idx - 1]
                    body_pose[char_idx, t_idx] = body_pose[char_idx, t_idx - 1]
                    hand_pose[char_idx, t_idx] = hand_pose[char_idx, t_idx - 1]
                    head_pose[char_idx, t_idx] = head_pose[char_idx, t_idx - 1]
                    pelvis_rotation[char_idx, t_idx] = pelvis_rotation[char_idx, t_idx - 1]
                    global_rotation[char_idx, t_idx] = global_rotation[char_idx, t_idx - 1]
                    global_translation[char_idx, t_idx] = global_translation[char_idx, t_idx - 1]

        return {
            "timesteps": timesteps,
            "shape": shape,
            "expression": expression,
            "body_pose": body_pose,
            "hand_pose": hand_pose,
            "head_pose": head_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": global_rotation,
            "global_translation": global_translation,
        }

    def _get_character_images(self, char_idx: int, frame_idx: int, cam_ids: list[str] | list[int]) -> ExtendedImageBatch:
        cam_names = self.cameras[char_idx]["cam_names"]
        strided_cam_ids = np.linspace(0, len(cam_names) - 1, self.config.rig_size, dtype=int)
        cam_ids = OmegaConf.to_container(cam_ids)

        if isinstance(cam_ids[0], str):
            cam_ids = np.array([cam_names.index(cam_id) for cam_id in cam_ids])
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
                mask = tvtransforms.functional.resize(mask, image.shape[1:])
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

        extimages = ExtendedImageBatch.stack(extimages)
        return extimages

    def load_image(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
        prefix: bool = False,
    ) -> Float[Tensor, "3 h w"]:
        filename = f"{frame_idx:04d}.jpg"
        if prefix:
            filename = "A-" + filename  # For A-pose images, prefix with 'A-'
        image = Image.open(
            self.config.data_dir / self.characters[char_idx] / "images" / self.cameras[char_idx]["cam_names"][cam_idx] / filename
        ).convert("RGB")
        tv_image = tvtransforms.functional.to_image(image)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_mask(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
        prefix: bool = False,
    ) -> Float[Tensor, "1 h w"]:
        filename = f"{frame_idx:04d}.png"
        if prefix:
            filename = "A-" + filename
        mask = Image.open(
            self.config.data_dir / self.characters[char_idx] / "masks" / self.cameras[char_idx]["cam_names"][cam_idx] / filename
        ).convert("L")
        mask = (np.array(mask, dtype=np.uint8) > 200).astype(np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel_erode), kernel_dilate)
        tv_mask = tvtransforms.functional.to_image(mask[..., None] * 255)
        return tvtransforms.functional.to_dtype(tv_mask, dtype=torch.float32, scale=True)

    def load_depth(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Float[Tensor, "1 h w"]:
        depth = cv2.imread(
            str(
                self.config.depth_dir
                / self.characters[char_idx]
                / self.cameras[char_idx]["cam_names"][cam_idx]
                / f"{(frame_idx + 1) * 25:04d}.exr"
            ),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )  # presumably this is absolute metric depth, depth == 0 means background
        return torch.from_numpy(depth[None, ...]).to(dtype=torch.float32)

    def load_normal(
        self,
        char_idx: int,
        frame_idx: int,
        cam_idx: int,
    ) -> Float[Tensor, "3 h w"]:
        normals = Image.open(
            self.config.normals_dir
            / self.characters[char_idx]
            / self.cameras[char_idx]["cam_names"][cam_idx]
            / f"{(frame_idx + 1) * 25:04d}.jpg"
        ).convert("RGB")
        tv_normals = tvtransforms.functional.to_image(normals)
        return tvtransforms.functional.to_dtype(tv_normals, dtype=torch.float32, scale=True)  # keep in [0, 1] region

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
        """Return the total number of frames across all characters."""
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

        # frame_idx is in the range [0, MAX_NUM_FRAMES-1] for array indexing
        frame_number = self.annotations["timesteps"][char_idx, frame_idx]  # This is the actual frame name on disk
        source_views = self._get_character_images(char_idx, frame_number, self.config.num_input_cameras)
        target_views = self._get_character_images(char_idx, frame_number, self.config.num_target_cameras)

        # load depth and normals here if needed

        motion = {
            "gender": "neutral",
            "flat_hand": True,
            "hand_pose_dim": 12,
            "pelvis_translation": torch.zeros((1, 3), dtype=torch.float32),
        }

        for k, v in self.annotations.items():
            if k == "shape":
                motion[k] = v[char_idx].unsqueeze(0)
            elif k == "timesteps":
                continue
            elif k == "hand_pose":
                motion[k] = torch.cat([v[char_idx, frame_idx], torch.zeros(78, dtype=torch.float32)], dim=0).unsqueeze(
                    0
                )  # hand pose is 90D max, we pad to it
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
            "texture": texture,
            "motion": motion,
            "patch_data": patch_data,
        }


@register_dataset("mvhppapose")
class MVHPPAposeDataset(MVHPPDataset):
    __config_class__ = MVHPPDatasetConfig

    def __init__(
        self,
        config: MVHPPDatasetConfig,
        characters: list[str],
        split: str = "train",
    ):
        super().__init__(config, characters, split)

        self.apose_annotations = self._load_smplx_apose_annotations()

    def _load_smplx_apose_annotations(self) -> dict[str, Float[Tensor, "..."] | Int[Tensor, "..."]]:
        num_characters = len(self.characters)

        timesteps = torch.zeros(num_characters, dtype=torch.int32)  # A-pose has only one frame per character
        shape = torch.zeros((num_characters, 10), dtype=torch.float32)
        expression = torch.zeros((num_characters, 10), dtype=torch.float32)
        body_pose = torch.zeros((num_characters, 63), dtype=torch.float32)
        hand_pose = torch.zeros((num_characters, 12), dtype=torch.float32)
        head_pose = torch.zeros((num_characters, 9), dtype=torch.float32)
        pelvis_rotation = torch.zeros((num_characters, 3), dtype=torch.float32)
        global_rotation = torch.zeros((num_characters, 3), dtype=torch.float32)
        global_translation = torch.zeros((num_characters, 3), dtype=torch.float32)

        for char_idx, char_name in enumerate(self.characters):
            annots_dir = self.config.data_dir / char_name / "smplx_params_new"

            apose_file = next(annots_dir.glob("A-*.npz"))
            apose_idx = int(apose_file.stem.split("-")[1])  # Extract the index from the filename
            timesteps[char_idx] = torch.tensor(apose_idx, dtype=torch.int32)

            apose_labels = np.load(apose_file)
            shape[char_idx] = torch.tensor(apose_labels["betas"], dtype=torch.float32).squeeze()
            expression[char_idx] = torch.tensor(apose_labels["expression"], dtype=torch.float32).squeeze()
            body_pose[char_idx] = torch.tensor(apose_labels["body_pose"], dtype=torch.float32).squeeze()
            hand_pose[char_idx] = torch.cat(
                [
                    torch.tensor(apose_labels["left_hand_pose"], dtype=torch.float32),
                    torch.tensor(apose_labels["right_hand_pose"], dtype=torch.float32),
                ],
                dim=1,
            ).squeeze()
            head_pose[char_idx] = torch.cat(
                [
                    torch.tensor(apose_labels["jaw_pose"], dtype=torch.float32),
                    torch.tensor(apose_labels["leye_pose"], dtype=torch.float32),
                    torch.tensor(apose_labels["reye_pose"], dtype=torch.float32),
                ],
                dim=1,
            ).squeeze()
            pelvis_rotation[char_idx] = torch.tensor(apose_labels["global_orient"], dtype=torch.float32).squeeze()
            global_translation[char_idx] = torch.tensor(apose_labels["transl"], dtype=torch.float32).squeeze()

        return {
            "timesteps": timesteps,
            "shape": shape,
            "expression": expression,
            "body_pose": body_pose,
            "hand_pose": hand_pose,
            "head_pose": head_pose,
            "pelvis_rotation": pelvis_rotation,
            "global_rotation": global_rotation,
            "global_translation": global_translation,
        }

    def _get_character_apose_images(self, char_idx: int, frame_idx: int, cam_ids: list[str] | list[int]) -> ExtendedImageBatch:
        cam_names = self.cameras[char_idx]["cam_names"]
        strided_cam_ids = np.linspace(0, len(cam_names) - 1, self.config.rig_size, dtype=int)
        if not isinstance(cam_ids, list):
            cam_ids = OmegaConf.to_container(cam_ids)

        if isinstance(cam_ids[0], str):
            cam_ids = np.array([cam_names.index(cam_id) for cam_id in cam_ids])
        else:
            sampled_num_cams = np.random.choice(cam_ids, 1, replace=False)[0]
            max_num_cams = max(cam_ids)
            cam_ids = np.random.choice(strided_cam_ids, sampled_num_cams, replace=False)
            cam_ids = np.pad(cam_ids, (0, max_num_cams - len(cam_ids)), mode="constant", constant_values=cam_ids[0])

        extimages = []
        for cam_id in cam_ids:
            image = self.load_image(char_idx, frame_idx, cam_id, prefix=True)
            mask = self.load_mask(char_idx, frame_idx, cam_id, prefix=True)
            if image.shape[1:] != mask.shape[1:]:
                mask = tvtransforms.functional.resize(mask, image.shape[1:])
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

        extimages = ExtendedImageBatch.stack(extimages)
        return extimages

    def __getitem__(self, idx: int) -> dict:
        result = super().__getitem__(idx)

        char_idx = idx % len(self.characters)

        result["apose_motion"] = {
            "gender": "neutral",
            "flat_hand": True,
            "hand_pose_dim": 12,
            "pelvis_translation": torch.zeros((1, 3), dtype=torch.float32),
            "shape": self.apose_annotations["shape"][char_idx].unsqueeze(0),
            "expression": self.apose_annotations["expression"][char_idx].unsqueeze(0),
            "body_pose": self.apose_annotations["body_pose"][char_idx].unsqueeze(0),
            "hand_pose": torch.cat([self.apose_annotations["hand_pose"][char_idx], torch.zeros(78, dtype=torch.float32)], dim=0).unsqueeze(
                0
            ),
            "head_pose": self.apose_annotations["head_pose"][char_idx].unsqueeze(0),
            "pelvis_rotation": self.apose_annotations["pelvis_rotation"][char_idx].unsqueeze(0),
            "global_rotation": self.apose_annotations["global_rotation"][char_idx].unsqueeze(0),
            "global_translation": self.apose_annotations["global_translation"][char_idx].unsqueeze(0),
        }

        apose_idx = self.apose_annotations["timesteps"][char_idx].item()
        result["apose_views"] = self._get_character_apose_images(char_idx, apose_idx, self.config.num_input_cameras)

        return result
