import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision
from jaxtyping import Float
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from giga.engine.utils import base as mng_utils
from giga.engine.utils import testing as mng_testing


class Evaluator:
    def __init__(
        self,
        logging_dir: Path,
        device: str | torch.device,
        save_rgb: bool,
        save_prims: bool,
        render_save_format: Literal["jpg", "png"] = "png",
        render_grid_format: Literal["vertical", "horizontal", "grid"] = "vertical",
        video_save_framerate: int = 30,
    ):
        self.device = device
        logging_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = logging_dir
        self.render_save_format = render_save_format
        self.render_grid_format = render_grid_format

        self.save_rgb = save_rgb
        self.save_prims = save_prims
        self.video_save_framerate = video_save_framerate

        if self.save_rgb:
            self.rgb_dir = logging_dir / "rgb"
            self.rgb_dir.mkdir(parents=True, exist_ok=True)

            self.input_texture_dir = logging_dir / "input_textures"
            self.input_texture_dir.mkdir(parents=True, exist_ok=True)

            self.output_texture_dir = logging_dir / "output_textures"
            self.output_texture_dir.mkdir(parents=True, exist_ok=True)

        if self.save_prims:
            self.prims_dir = logging_dir / "prims"
            self.prims_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image: Float[Tensor, "b c h w"],
        name: str | Path,
        nrow: int | None = None,
    ):
        if self.render_grid_format == "vertical":
            nrow = 1
        elif self.render_grid_format == "horizontal":
            nrow = image.shape[0]
        else:
            pass  # whatever nrow is passed

        torchvision.utils.save_image(image, name, format=self.render_save_format, nrow=nrow)

    def save_metrics(
        self,
        psnr: Tensor,
        ssim: Tensor,
        lpips: Tensor,
        name: str | Path,
    ):
        metrics = {
            "psnr": psnr.numpy(),
            "ssim": ssim.numpy(),
            "lpips": lpips.numpy(),
        }

        np.savez(name, **metrics)

        summary_filename = name.with_suffix(".summary.json")
        with open(summary_filename, "w") as f:
            json.dump(
                {
                    "psnr": float(psnr.mean().item()),
                    "ssim": float(ssim.mean().item()),
                    "lpips": float(lpips.mean().item()),
                },
                f,
            )

    @torch.no_grad()
    def evaluate(
        self,
        model: LightningModule,
        dataloader: DataLoader,
    ):
        model = model.to(self.device)
        model.eval()

        precision = torch.bfloat16
        dataloader_len = len(dataloader)

        psnr_scores, ssim_scores, lpips_scores = [], [], []

        for batch_idx, batch in tqdm(enumerate(dataloader), total=dataloader_len):
            batch = mng_utils.to_device(batch, self.device)

            with torch.autocast(device_type="cuda", dtype=precision):
                prediction, renders = model.prediction_step(batch)

            batch_size, num_views = renders.shape[:2]
            renders = mng_utils.tensor_to_4d(renders)
            pred_rgb, pred_mask = renders.split([3, 1], dim=1)

            gt_images = torch.cat([b.image for b in batch["target_views"]], dim=0)
            gt_rgb, gt_mask = gt_images.split([3, 1], dim=1)

            # given masks, will do bbox cropping
            metrics = model.eval_metrics(pred_rgb.float(), gt_rgb, gt_mask)

            psnr_scores.append(torch.atleast_1d(metrics["psnr"].cpu()))
            ssim_scores.append(torch.atleast_1d(metrics["ssim"].cpu()))
            lpips_scores.append(torch.atleast_1d(metrics["lpips"].cpu()))
            if self.save_rgb:
                rgb_name = self.rgb_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(renders.float(), rgb_name, nrow=batch_size)

            if self.save_prims:
                raise RuntimeWarning("Saving prims is not implemented yet.")
                # render gaussians as solid ellipsoids here
                prims_name = self.prims_dir / f"batch_{batch_idx:04d}"
                self.save_image(pred_mask, prims_name, nrow=batch_size)

        psnr_scores = torch.cat(psnr_scores, dim=0)
        ssim_scores = torch.cat(ssim_scores, dim=0)
        lpips_scores = torch.cat(lpips_scores, dim=0)
        self.save_metrics(psnr_scores, ssim_scores, lpips_scores, self.logging_dir / "metrics.npz")

        if self.save_rgb:
            mng_testing.make_video(
                self.rgb_dir,
                self.logging_dir / "rgb_video.mp4",
                file_pattern=r"batch_%04d",
                image_format=self.render_save_format,
                framerate=self.video_save_framerate,
            )

        if self.save_prims:
            raise RuntimeWarning("Saving prims is not implemented yet.")
            mng_testing.make_video(
                self.prims_dir,
                self.logging_dir / "prims_video.mp4",
                image_format=self.render_save_format,
                framerate=self.video_save_framerate,
            )

    @torch.inference_mode()
    def render_only(
        self,
        model: LightningModule,
        dataloader: DataLoader,
    ):
        model = model.to(self.device)
        model.eval()

        dataloader_len = len(dataloader)
        precision = torch.bfloat16

        for batch_idx, batch in tqdm(enumerate(dataloader), total=dataloader_len):
            batch = mng_utils.to_device(batch, self.device)

            with torch.autocast(device_type="cuda", dtype=precision):
                prediction, renders = model.prediction_step(batch)

            batch_size, num_views = renders.shape[:2]
            renders = mng_utils.tensor_to_4d(renders)
            pred_rgb, pred_mask = renders.split([3, 1], dim=1)

            gt_images = torch.cat([b.image for b in batch["target_views"]], dim=0)
            gt_rgb, gt_mask = gt_images.split([3, 1], dim=1)

            if self.save_rgb:
                rgb_name = self.rgb_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(renders.float(), rgb_name, nrow=batch_size)

                input_texture = (prediction["texture"][:, :3] + 1) * 0.5
                input_texture_name = self.input_texture_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(input_texture.float(), input_texture_name, nrow=batch_size)

                output_texture = prediction["colors"]
                output_texture_name = self.output_texture_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(output_texture.float(), output_texture_name, nrow=batch_size)

            if self.save_prims:
                raise RuntimeWarning("Saving prims is not implemented yet.")
                # render gaussians as solid ellipsoids here
                prims_name = self.prims_dir / f"batch_{batch_idx:04d}"
                self.save_image(pred_mask, prims_name, nrow=batch_size)

        if self.save_rgb:
            mng_testing.make_video(
                self.rgb_dir,
                self.logging_dir / "rgb_video.mp4",
                file_pattern=r"batch_%04d",
                image_format=self.render_save_format,
                framerate=self.video_save_framerate,
            )

        if self.save_prims:
            raise RuntimeWarning("Saving prims is not implemented yet.")
            mng_testing.make_video(
                self.prims_dir,
                self.logging_dir / "prims_video.mp4",
                image_format=self.render_save_format,
                framerate=self.video_save_framerate,
            )

    @torch.inference_mode()
    def freeview(
        self,
        model: LightningModule,
        dataloader: DataLoader,
        extrinsics_trajectory: Tensor,
        intrinsics_trajectory: Tensor,
        resolution: tuple[int, int],
    ):
        model = model.to(self.device)
        model.eval()

        dataloader_len = len(dataloader)
        precision = torch.bfloat16  # if "bf16" in model.precision else torch.float32

        trajectory_len = extrinsics_trajectory.shape[0]

        for batch_idx, batch in tqdm(enumerate(dataloader), total=dataloader_len):
            batch = mng_utils.to_device(batch, self.device)

            camera = {
                "extrinsics": extrinsics_trajectory[batch_idx % trajectory_len],
                "intrinsics": intrinsics_trajectory[batch_idx % trajectory_len],
                "resolution": resolution,
            }
            with torch.autocast(device_type="cuda", dtype=precision):
                prediction, renders = model.freeview_step(batch, camera)

            batch_size, num_views = renders.shape[:2]
            renders = mng_utils.tensor_to_4d(renders)
            pred_rgb, pred_mask = renders.split([3, 1], dim=1)

            if self.save_rgb:
                rgb_name = self.rgb_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(renders.float(), rgb_name, nrow=batch_size)

                input_texture = (prediction["texture"][:, :3] + 1) * 0.5
                input_texture_name = self.input_texture_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(input_texture.float(), input_texture_name, nrow=batch_size)

                output_texture = prediction["colors"]
                output_texture_name = self.output_texture_dir / f"batch_{batch_idx:04d}.{self.render_save_format}"
                self.save_image(output_texture.float(), output_texture_name, nrow=batch_size)

            if self.save_prims:
                raise RuntimeWarning("Saving prims is not implemented yet.")
                # render gaussians as solid ellipsoids here
                prims_name = self.prims_dir / f"batch_{batch_idx:04d}"
                self.save_image(pred_mask, prims_name, nrow=batch_size)

        if self.save_rgb:
            mng_testing.make_video(
                self.rgb_dir,
                self.logging_dir / "rgb_video.mp4",
                file_pattern=r"batch_%04d",
                image_format=self.render_save_format,
                framerate=self.video_save_framerate,
            )
