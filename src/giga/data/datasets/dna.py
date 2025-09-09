import io
from dataclasses import dataclass
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.v2 as tvtransforms
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ..extimage import ExtendedImage, ExtendedImageBatch
from .config import DatasetConfig, register_dataset
from .utils import center_crop_bbox, sample_character_patches


@dataclass
class DNADatasetConfig(DatasetConfig):
    """Configuration for DNA dataset."""

    _target_: str = "giga.data.datasets.dna.DNADataset"
    main_dir: str = "main"
    annots_dir: str = "annotations"
    rig_size: int = 16


@register_dataset("dna")
class DNADataset(Dataset):
    """DNA Dataset with clean implementation matching MVH structure."""

    __config_class__ = DNADatasetConfig

    DATA_MIN_SIZE = 2048  # Minimum size for initial image loading

    def __init__(
        self,
        config: DNADatasetConfig,
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

        self.main_data_dir = config.data_dir / config.main_dir
        self.annots_data_dir = config.data_dir / config.annots_dir

        self.annotations = self._load_smplx_annotations()
        self.cameras = self._load_camera_calibrations()
        self.main_files = self._open_data_files()

    @classmethod
    def select_characters(cls, config: DNADatasetConfig) -> list[str]:
        if config.exclusions_file is not None:
            with open(config.exclusions_file, "r") as f:
                excluded_characters = set(f.read().splitlines())
        else:
            excluded_characters = set()

        available_characters = sorted([d.stem for d in (config.data_dir / config.main_dir).iterdir() if d not in excluded_characters])
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

    def _open_data_files(self) -> list[tuple[h5py.File, h5py.File]]:
        """Open HDF5 files for all characters."""
        files = []
        for char_name in self.characters:
            main_file = h5py.File(self.main_data_dir / f"{char_name}.smc", "r")
            annots_file = h5py.File(self.annots_data_dir / f"{char_name}_annots.smc", "r")
            files.append((main_file, annots_file))
        return files

    def _load_camera_calibrations(self) -> list[dict[str, Any]]:
        """Load camera calibration data for all characters."""
        calibrations = []
        for char_name in self.characters:
            main_file = h5py.File(self.main_data_dir / f"{char_name}.smc", "r")
            annots_file = h5py.File(self.annots_data_dir / f"{char_name}_annots.smc", "r")

            meta = {
                "5mp": main_file["Camera_5mp"].attrs["resolution"][::-1].tolist(),
                "12mp": main_file["Camera_12mp"].attrs["resolution"][::-1].tolist(),
            }

            cam_data = annots_file["Camera_Parameter"]
            num_cams = min(60, len(cam_data))

            intrinsics = torch.zeros((num_cams, 3, 3))
            distortion = torch.zeros((num_cams, 5))
            distorted_intrinsics = torch.zeros((num_cams, 3, 3))
            extrinsics = torch.zeros((num_cams, 4, 4))
            resolution = torch.zeros((num_cams, 2), dtype=torch.int32)
            cam_names = []

            for idx, (cam_key, cam_val) in enumerate(cam_data.items()):
                if idx < num_cams:
                    meta_info = meta["5mp"]
                else:
                    meta_info = meta["12mp"]

                cam_name = f"{int(cam_key):02d}"
                cam_names.append(cam_name)

                # Load camera parameters
                _K = cam_val["K"][()]
                _D = cam_val["D"][()]
                # We do not really undistort images, but in case we will do it - load it here
                new_K, roi = cv2.getOptimalNewCameraMatrix(_K, _D, meta_info, 0, meta_info)
                K = torch.tensor(new_K, dtype=torch.float32).reshape(3, 3)
                distorted_K = torch.tensor(_K, dtype=torch.float32).reshape(3, 3)
                D = torch.tensor(_D, dtype=torch.float32)
                RT = torch.tensor(cam_val["RT"][()], dtype=torch.float32)

                intrinsics[idx] = K
                extrinsics[idx] = torch.inverse(RT)
                resolution[idx] = torch.tensor(meta_info, dtype=torch.int32)
                distortion[idx] = D
                distorted_intrinsics[idx] = distorted_K

            calibrations.append(
                {
                    "intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "resolution": resolution,
                    "cam_names": cam_names,
                    "distortion": distortion,
                    "distorted_intrinsics": distorted_intrinsics,
                }
            )

        return calibrations

    def _load_smplx_annotations(self) -> dict[str, Tensor]:
        """Load SMPL-X annotations for all characters."""
        timesteps_list = []
        betas_list = []
        expression_list = []
        full_pose_list = []
        pelvis_trans_list = []

        for char_name in self.characters:
            annots_file = h5py.File(self.annots_data_dir / f"{char_name}_annots.smc", "r")
            max_timesteps = len(annots_file["SMPLx"]["fullpose"][()])
            # Handle timesteps based on config
            if isinstance(self.config.timesteps[0], int) and len(self.config.timesteps) == 1:
                if self.config.timesteps[0] == -1:
                    frame_indices = list(range(max_timesteps))
                else:
                    # Single timestep case
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

            betas = torch.tensor(annots_file["SMPLx"]["betas"][()][0], dtype=torch.float32).reshape(-1)
            betas_list.append(betas)

            expressions = []
            full_poses = []
            pelvis_trans = []

            for frame_idx in frame_indices:
                expressions.append(torch.tensor(annots_file["SMPLx"]["expression"][()][frame_idx], dtype=torch.float32).reshape(-1))
                full_poses.append(torch.tensor(annots_file["SMPLx"]["fullpose"][()][frame_idx], dtype=torch.float32).reshape(-1))
                pelvis_trans.append(torch.tensor(annots_file["SMPLx"]["transl"][()][frame_idx], dtype=torch.float32).reshape(-1))

            timesteps_list.append(torch.tensor(frame_indices))
            expression_list.append(torch.stack(expressions))
            full_pose_list.append(torch.stack(full_poses))
            pelvis_trans_list.append(torch.stack(pelvis_trans))

        return {
            "timesteps": torch.stack(timesteps_list),
            "shape": torch.stack(betas_list),
            "expression": torch.stack(expression_list),
            "full_pose": torch.stack(full_pose_list),
            "pelvis_translation": torch.stack(pelvis_trans_list),
        }

    def _get_character_images(self, char_idx: int, frame_idx: int, cam_ids: list[str] | list[int]) -> ExtendedImageBatch:
        """
        Load and process images for a character.
        IMPORTANT: first 48 cameras are in 5mp, and the last 12 in 12 mp - they have different resolution.
        Randomly sampled cameras will always be picked from 5 mp.
        Manually selected cameras should all belong to the same type: either provide indices for 5 mp, or 12 mp cameras.
        Otherwise ExtendedImageBatch class will crash because of the collation issues across image resolution dimensions.
        """
        main_file, annots_file = self.main_files[char_idx]
        cam_names = self.cameras[char_idx]["cam_names"]
        if not isinstance(cam_ids, list):
            cam_ids = OmegaConf.to_container(cam_ids)
        if isinstance(cam_ids[0], str):
            cam_indices = [cam_names.index(cam_id) for cam_id in cam_ids]
        else:
            strided_indices = np.linspace(0, len(cam_names[:48]) - 1, self.config.rig_size, dtype=int)
            num_cams = np.random.choice(cam_ids)
            cam_indices = np.random.choice(strided_indices, num_cams, replace=False)

        extimages = []
        for cam_idx in cam_indices:
            # Load image and mask
            image = self.load_image(main_file, cam_idx, frame_idx)
            mask = self.load_mask(annots_file, cam_idx, frame_idx)
            rgba_image = torch.cat([image * mask, mask], dim=0)

            extimages.append(
                ExtendedImage(
                    image=rgba_image,
                    intrinsics=self.cameras[char_idx]["intrinsics"][cam_idx],
                    extrinsics=self.cameras[char_idx]["extrinsics"][cam_idx],
                )
            )

        extimages = ExtendedImageBatch.stack(extimages)
        center_bbox = center_crop_bbox(extimages.resolution)
        extimages = extimages.crop(center_bbox)
        extimages = extimages.resize(self.config.image_size)

        return extimages

    def load_image(self, main_file: h5py.File, cam_idx: int, frame_idx: int) -> Tensor:
        """Load image from HDF5 file."""
        img_bytes = main_file["Camera_5mp"][str(cam_idx)]["color"][str(frame_idx)][()]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tv_image = tvtransforms.functional.to_image(image)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_mask(self, annots_file: h5py.File, cam_idx: int, frame_idx: int) -> Tensor:
        """Load mask from HDF5 file."""
        mask_bytes = annots_file["Mask"][str(cam_idx)]["mask"][str(frame_idx)][()]
        mask = (np.array(Image.open(io.BytesIO(mask_bytes)).convert("L")) > 200).astype(np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(cv2.erode(mask, kernel_erode), kernel_dilate)
        tv_image = tvtransforms.functional.to_image(mask[..., None] * 255)
        return tvtransforms.functional.to_dtype(tv_image, dtype=torch.float32, scale=True)

    def load_texture(self, char_idx: int) -> Tensor:
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
        """Get a single sample."""
        char_idx = idx % len(self.characters)
        frame_idx = idx % self.annotations["timesteps"].shape[1]

        # Get source and target views
        true_timestep = self.annotations["timesteps"][char_idx, frame_idx].item()
        source_views = self._get_character_images(char_idx, true_timestep, self.config.num_input_cameras)
        target_views = self._get_character_images(char_idx, true_timestep, self.config.num_target_cameras)

        # Get motion parameters
        motion = {
            "gender": self.main_files[char_idx][0].attrs["gender"],
            "flat_hand": False,
            "hand_pose_dim": 90,  # 45 per hand
            "global_rotation": torch.zeros(1, 3, dtype=torch.float32),  # default for DNA-Rendering
            "pelvis_translation": torch.zeros(1, 3, dtype=torch.float32),  # default for DNA-Rendering
        }
        for k, v in self.annotations.items():
            if k == "full_pose":
                pose = v[char_idx, frame_idx].unsqueeze(0)
                motion["pelvis_rotation"] = pose[:, :3]
                motion["body_pose"] = pose[:, 3:66]
                motion["hand_pose"] = pose[:, 75:]  # 45 per hand, 90 total
                motion["head_pose"] = pose[:, 66:75]
            elif k == "shape":
                motion[k] = v[char_idx].unsqueeze(0)
            elif k == "pelvis_translation":  # replace global translation with pelvis translation, they behave similarly
                motion["global_translation"] = v[char_idx, frame_idx].unsqueeze(0)
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

        texture = self.load_texture(char_idx)

        return {
            "name": self.characters[char_idx],
            "input_views": source_views,
            "target_views": target_views,
            "motion": motion,
            "texture": texture,
            "patch_data": {"center": patch_centers, "radius": patch_dims, "size": self.config.patch_config["patch_size"]},
        }
