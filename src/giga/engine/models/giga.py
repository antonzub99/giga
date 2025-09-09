import os
from enum import Enum
from functools import partial
from typing import Callable

import nvdiffrast.torch as dr
import torch
import torchvision.transforms.v2.functional as tvtransforms
import wandb
from einops import rearrange
from lightning import LightningModule
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from giga.data import ExtendedImageBatch

from ..loss import LossComputer
from ..projector import collect_uv_texture
from ..renderer import gaussians as gaussian_rasterizer
from ..trainer.config import LoggingConfig, TrainerConfig
from ..trainer.scheduler import NoOpScheduler
from ..utils import base as mng_utils
from ..utils import geometry as mng_geometry
from ..utils import metrics as mng_metrics


class BG_COLORS(Enum):
    BLACK = torch.zeros(3, dtype=torch.float32)
    WHITE = torch.ones(3, dtype=torch.float32)


class GIGA(LightningModule):
    """
    This is the base version of GIGA, that needs sparse images and SMPLX parameters as inputs.
    """

    def __init__(
        self,
        model: Module,
        conditioner: Module,
        optimizer: Callable[..., Optimizer],
        loss_criterion: LossComputer,
        mesh_lbs_layer: Module,
        scheduler: Callable[..., LRScheduler] | None,
        trainer_config: TrainerConfig,
        logging_config: LoggingConfig,
        texel_scales: Tensor | None = None,
    ):
        super().__init__()

        self.trainer_config = trainer_config
        self.logging_config = logging_config

        self.compile_flag = False
        if self.trainer_config.compile:
            logger.info("Compiling model for single-GPU training...")
            model = torch.compile(model, mode="max-autotune-no-cudagraphs", fullgraph=True)
            conditioner = torch.compile(conditioner, mode="max-autotune-no-cudagraphs", fullgraph=True)
            self.compile_flag = True

        DEBUG = os.environ.get("GIGA_DEBUG", "0") == "1"
        if DEBUG:
            logger.info("Debug mode is enabled, registering hooks for activation and gradient inspection.")
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Module):
                    module.register_forward_hook(partial(self.activation_inspector_hook, module_name=name))
                    module.register_full_backward_hook(partial(self.gradient_inspector_hook, module_name=name))

        self.model = model
        self.mesh_lbs_layer = mesh_lbs_layer
        self.conditioner = conditioner

        self.texel_scales = torch.nn.ParameterDict()
        if texel_scales is not None:
            self._add_texel_scales_for_resolution(self.trainer_config.texture_resolution, texel_scales)
        else:
            logger.warning("No precomputed texel scales provided, will use default scales when needed.")

        # set fake fields to be used in the configure_optimizer
        self._optimizer = optimizer
        self._scheduler = scheduler

        self.loss_criterion: LossComputer = loss_criterion.to(self.device)

        # utility fields
        self.phase = None
        self.lr_schedule_scale = 1.0
        self.base_lr = 1.0
        self.current_batch = None
        self.current_batch_idx = None

    def _inspect_tensor(self, tensor: Tensor, name: str, module_name: str) -> None:
        if torch.isnan(tensor).any():
            batch_info = ""
            if self.current_batch_idx is not None:
                batch_info += f" (batch_idx: {self.current_batch_idx})"
            if self.current_batch is not None and "name" in self.current_batch:
                batch_info += f" (batch_names: {self.current_batch['name']})"
            logger.error(f"NaN {name} found in {module_name}{batch_info}")
        if not tensor.abs().sum() > 0:
            logger.warning(f"Zero {name} found in {module_name}")

    def activation_inspector_hook(self, module, inputs, outputs, module_name: str):
        if isinstance(outputs, Tensor):
            self._inspect_tensor(outputs, "activation", module_name)
        elif isinstance(outputs, (tuple, list)):
            for item in outputs:
                if isinstance(item, Tensor):
                    self._inspect_tensor(item, "activation", module_name)

    def gradient_inspector_hook(self, module, grad_input, grad_output, module_name: str):
        for grad in grad_input:
            if isinstance(grad, Tensor):
                self._inspect_tensor(grad, "gradient input", module_name)
        for grad in grad_output:
            if isinstance(grad, Tensor):
                self._inspect_tensor(grad, "gradient output", module_name)

    def forward(self, inputs: dict) -> Tensor:
        """Forward pass through the model. A Lightning wrapper around the model forward pass."""
        condition = self.conditioner(inputs["motion"])  # (b l f)
        main_input = torch.cat([inputs["appearance"], inputs["geometry"]], dim=1)  # (b c h w) -> (b 6 h w)
        return self.model(main_input, condition)

    def training_step(self, batch: dict, batch_idx: int | None = None):
        """Perform a single training step."""
        self.phase = "train"
        self.current_batch = batch
        self.current_batch_idx = batch_idx

        inputs, lbs_kwargs = self.prepare_inputs(batch, eval=False)
        targets = batch["target_views"]  # should contain ExtendedImageBatch'es
        target_images = self.prepare_targets(batch)

        prediction = self(inputs)
        prediction = self.apply_activation_and_pose(prediction, inputs["geometry"], lbs_kwargs)
        targets = self.normalize_targets(targets, lbs_kwargs)
        outputs = self.prepare_outputs(prediction, targets)

        loss_dict = self.loss_criterion(mng_utils.tensor_to_4d(outputs), mng_utils.tensor_to_4d(target_images), batch["patch_data"][0])
        loss = sum(loss_dict.values())
        scales_threshold = torch.mean(
            self.get_texel_scales(self.trainer_config.texture_resolution) * self.trainer_config.scale_multiplier
        )  # use average scales as threshold
        regularizer_dict = self.loss_criterion.regularize(prediction, opacity_threshold=0.1, scales_threshold=scales_threshold)
        penalty = sum(regularizer_dict.values())
        loss += penalty
        loss_dict.update(regularizer_dict)

        self.log_step(batch, batch_idx, inputs, outputs, prediction, targets, loss_dict, lbs_kwargs)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict, batch_idx: int | None = None):
        """Perform a single validation step."""
        self.phase = "val"
        self.current_batch = batch
        self.current_batch_idx = batch_idx

        inputs, lbs_kwargs = self.prepare_inputs(batch, eval=True)
        targets = batch["target_views"]  # should contain ExtendedImageBatch'es
        target_images = self.prepare_targets(batch)

        prediction = self(inputs)
        prediction = self.apply_activation_and_pose(prediction, inputs["geometry"], lbs_kwargs)
        targets = self.normalize_targets(targets, lbs_kwargs)

        outputs = self.prepare_outputs(prediction, targets)

        loss_dict = self.loss_criterion(mng_utils.tensor_to_4d(outputs), mng_utils.tensor_to_4d(target_images), batch["patch_data"][0])
        loss = sum(loss_dict.values())
        scales_threshold = torch.mean(self.get_texel_scales(self.trainer_config.texture_resolution) * self.trainer_config.scale_multiplier)
        regularizer_dict = self.loss_criterion.regularize(prediction, opacity_threshold=0.1, scales_threshold=scales_threshold)
        penalty = sum(regularizer_dict.values())
        loss += penalty
        loss_dict.update(regularizer_dict)

        self.log_step(batch, batch_idx, inputs, outputs, prediction, targets, loss_dict, lbs_kwargs)

        return loss

    @torch.no_grad()
    def prediction_step(self, batch: dict, batch_idx: int | None = None):
        """Perform a single prediction step."""
        self.phase = "predict"
        self.current_batch = batch
        self.current_batch_idx = batch_idx

        inputs, lbs_kwargs = self.prepare_inputs(batch, eval=True)
        targets = batch["target_views"]  # should contain ExtendedImageBatch'es

        prediction = self(inputs)
        prediction = self.apply_activation_and_pose(prediction, inputs["geometry"], lbs_kwargs)
        targets = self.normalize_targets(targets, lbs_kwargs)

        outputs = self.prepare_outputs(prediction, targets)
        prediction["texture"] = inputs["appearance"]

        return prediction, outputs

    @torch.inference_mode()
    def freeview_step(self, batch: dict, camera: dict, batch_idx: int | None = None):
        """Perform a single freeview rendering step."""
        self.phase = "predict"
        self.current_batch = batch
        self.current_batch_idx = batch_idx

        inputs, lbs_kwargs = self.prepare_inputs(batch, eval=True)
        camera_res = camera["resolution"]

        empty_target_image = torch.empty(1, 4, *camera_res, device=self.device)
        target_camera = [
            ExtendedImageBatch(
                image=empty_target_image,
                extrinsics=camera["extrinsics"].unsqueeze(0),
                intrinsics=camera["intrinsics"].unsqueeze(0),
            ).to(self.device)
        ]

        prediction = self(inputs)
        prediction = self.apply_activation_and_pose(prediction, inputs["geometry"], lbs_kwargs)
        targets = self.normalize_targets(target_camera, lbs_kwargs)

        outputs = self.prepare_outputs(prediction, targets)
        prediction["texture"] = inputs["appearance"]

        return prediction, outputs

    @torch.inference_mode()
    def freeview_step_debug(self, batch: dict, camera: dict, batch_idx: int | None = None):
        """Perform a single freeview rendering step."""
        self.phase = "predict"
        self.current_batch = batch
        self.current_batch_idx = batch_idx

        # Timing setup
        start_network = torch.cuda.Event(enable_timing=True)
        end_network = torch.cuda.Event(enable_timing=True)
        start_input = torch.cuda.Event(enable_timing=True)
        end_input = torch.cuda.Event(enable_timing=True)
        start_render = torch.cuda.Event(enable_timing=True)
        end_render = torch.cuda.Event(enable_timing=True)

        # Start timing network computation

        start_input.record()
        inputs, lbs_kwargs = self.prepare_inputs(batch, eval=True)
        camera_res = camera["resolution"]

        empty_target_image = torch.empty(1, 4, *camera_res, device=self.device)
        target_camera = [
            ExtendedImageBatch(
                image=empty_target_image,
                extrinsics=camera["extrinsics"].unsqueeze(0),
                intrinsics=camera["intrinsics"].unsqueeze(0),
            ).to(self.device)
        ]
        end_input.record()

        start_network.record()
        prediction = self(inputs)
        prediction = self.apply_activation_and_pose(prediction, inputs["geometry"], lbs_kwargs)
        targets = self.normalize_targets(target_camera, lbs_kwargs)

        end_network.record()
        start_render.record()

        outputs = self.prepare_outputs(prediction, targets)

        end_render.record()

        prediction["texture"] = inputs["appearance"]

        torch.cuda.synchronize()
        input_time = start_input.elapsed_time(end_input)
        network_time = start_network.elapsed_time(end_network)
        render_time = start_render.elapsed_time(end_render)
        total_time = network_time + render_time

        logger.info(
            f"TIMING - Input preparation: {input_time:.4f}ms, Network computation: {network_time:.4f}ms, Rendering: {render_time:.4f}ms, Total: {total_time:.4f}ms"
        )

        return prediction, outputs

    @torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True)
    def prepare_mesh(self, body_outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare the body template mesh for input texture estimation."""
        vertices = body_outputs["vertices"].squeeze(0)
        normals = mng_geometry.compute_vertex_normals(vertices, self.mesh_lbs_layer.faces)
        attrs = self.mesh_lbs_layer.uv_vertices
        ids_w_boundary = self.mesh_lbs_layer.template_with_boundary

        body_mesh = {
            "vertices": vertices[ids_w_boundary],
            "faces": self.mesh_lbs_layer.uv_faces,
            "normals": normals[ids_w_boundary],
            "attrs": attrs[ids_w_boundary],
        }

        return body_mesh

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True)
    def prepare_inputs(self, batch: dict, eval: bool) -> tuple[dict, dict]:
        """Input for the model are UV-aligned RGB and position textures. We compute them here."""
        batch_size = len(batch["input_views"])  # ExtendedImageBatch is collated as a list of objects
        motion = batch["motion"]  # a list of dictionaries, each dictionary contains motion data for a single character instance
        lbs_data = []

        rgb_texture = torch.zeros(batch_size, 3, *self.trainer_config.texture_resolution, device=self.device)
        pos_texture = torch.zeros(batch_size, 3, *self.trainer_config.texture_resolution, device=self.device)

        # nvdiffrast can either handle rasterization of multiple mesh instances from 1 view
        # or rasterization of 1 mesh instance from multiple views
        # hence we have to loop over items in the batch
        for idx in range(batch_size):
            cameras: ExtendedImageBatch = batch["input_views"][idx]
            body_outputs = self.mesh_lbs_layer(**motion[idx])  # regular call on the usual mesh
            body_mesh = self.prepare_mesh(body_outputs)

            # now we go to the texel space
            vertices_rest, T_world_joint, joint_translation = self.mesh_lbs_layer.get_joint_world_transforms(**motion[idx])
            _pos_texture, texel_lbs_weights, uv_barycentrics = self.mesh_lbs_layer.texelize_template(
                vertices_rest, motion[idx]["gender"], self.nvdiffrast_ctx, self.trainer_config.texture_resolution
            )
            _pos_texture = rearrange(_pos_texture.squeeze(0), "h w c -> c h w")
            pos_mean = _pos_texture.mean()
            pos_max = _pos_texture.abs().max()
            pos_texture[idx] = _pos_texture / pos_max
            _rgb_texture = collect_uv_texture(
                body_mesh,
                cameras,
                self.nvdiffrast_ctx,
                uv_barycentrics,
                self.trainer_config.texture_resolution,
                **self.trainer_config.projection_settings,
            )
            rgb_texture[idx] = _rgb_texture.squeeze(0)[:3]  # first 3 channels are RGB colors
            uv_mask = uv_barycentrics[..., 3] > 0  # (1 h w) tensor, mask for valid texels
            # this will be required for the observed pose transformation

            # now let's normalize joints
            T_world_joint[..., :3, 3] = T_world_joint[..., :3, 3] / pos_max
            joint_translation = joint_translation / pos_max
            global_translation = motion[idx]["global_translation"] / pos_max

            _lbs_data = {
                "T_world_joint": T_world_joint.squeeze(0),
                "joint_translation": joint_translation.squeeze(0),
                "texel_lbs_weights": texel_lbs_weights.squeeze(0),
                "global_rotation": motion[idx]["global_rotation"].squeeze(0),
                "global_translation": global_translation.squeeze(0),
                "center_scale": torch.tensor([pos_mean, pos_max], device=self.device, dtype=_pos_texture.dtype),
                "uv_mask": uv_mask,
            }
            lbs_data.append(_lbs_data)

        lbs_data = mng_utils.batch_dict(lbs_data)  # convert list of dictionaries to a single dictionary with batched values
        repacked_motion = mng_utils.batch_dict(motion)

        # pad hand pose tensor if it is not 90-channels long
        hand_pose = repacked_motion["hand_pose"]
        hand_channels_pad = 90 - hand_pose.shape[-1]  # >= 0
        hand_pose = torch.cat([hand_pose, torch.zeros(batch_size, 1, hand_channels_pad, device=self.device, dtype=hand_pose.dtype)], dim=-1)
        local_motion = torch.cat([repacked_motion["body_pose"], repacked_motion["head_pose"], hand_pose], dim=-1)  # (b 1 (num_joints x 3))

        return {
            "appearance": rgb_texture * 2.0 - 1.0,  # normalize to [-1, 1] range
            "geometry": pos_texture,  # already zero-centered in [-1, 1] range
            "motion": local_motion,
        }, lbs_data

    def prepare_targets(self, batch: dict) -> Tensor:
        """Prepare targets for the model."""
        images = torch.stack([item.image for item in batch["target_views"]], dim=0)  # (b n c h w) -> (b c h w n)
        return images

    @torch.no_grad()
    def normalize_targets(self, targets: list[ExtendedImageBatch], lbs_kwargs: dict[str, Tensor]) -> list[ExtendedImageBatch]:
        pos_mean, pos_max = lbs_kwargs["center_scale"].chunk(2, dim=1)
        for idx, target in enumerate(targets):
            extrinsics = target.extrinsics.clone()  # (n 4 4) tensor
            extrinsics[:, :3, 3] = extrinsics[:, :3, 3] / pos_max[idx]  # normalize extrinsics translations
            target.extrinsics = extrinsics
        return targets

    @torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True)
    def prepare_outputs(self, gaussian_model: dict[str, Tensor], model_targets: list[ExtendedImageBatch]) -> Tensor:
        """
        The actual splatting of 3D Gaussians happens here.
        Note: This has to be performed in float32 precision, because rasterization is not supported in mixed precision.
        """
        # Rendering is happening here
        batch_size, *_ = gaussian_model["means"].shape
        gaussian_model = {k: v.view(v.shape[0], v.shape[1], -1).permute(0, 2, 1).contiguous() for k, v in gaussian_model.items()}

        extrinsics = torch.stack([item.extrinsics for item in model_targets], dim=0)  # (b n 4 4)
        intrinsics = torch.stack([item.intrinsics for item in model_targets], dim=0)
        resolutions = [item.resolution for item in model_targets]
        assert len(set(resolutions)) == 1, "All target images should have the same resolution"
        resolution = resolutions[0]
        batch_size, num_views, _, _ = extrinsics.shape
        if self.phase == "train":
            bg_color = torch.randn(
                batch_size, num_views, 3, device=gaussian_model["means"].device, dtype=gaussian_model["means"].dtype
            ).clip(0, 1)
        else:
            bg_color = (
                getattr(BG_COLORS, self.trainer_config.bg_color.upper(), BG_COLORS.BLACK)
                .value.to(gaussian_model["means"].device, dtype=gaussian_model["means"].dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, num_views, -1)
            )

        rgb, alpha = gaussian_rasterizer.rasterize(
            gaussian_model,
            extrinsics,
            intrinsics,
            resolution,
            bg_color,
            absgrad=True,
            rasterize_mode="antialiased",
        )

        renders = torch.cat([rgb, alpha], dim=2).clip(0, 1)  # (b n (3+1) h w) -> (b n 4 h w)
        return renders

    def apply_activation_and_pose(
        self,
        prediction: Tensor,
        pos_texture: Tensor,
        lbs_kwargs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """
        Raw network predictions have to be mapped to correct ranges for parameters of 3D Gaussians.
        LBS is also applied here.
        """
        gaussian_model = mng_utils.tensor_as_gaussians(prediction)  # key: (b c h w) tensor, batch dim is squeezed because batch dim = 1

        # we manually cast some tensors to the target precision because they are not handled by the autocast
        dtype = gaussian_model["means"].dtype
        uv_mask = lbs_kwargs["uv_mask"].to(dtype)
        offsets = self.offset_annealing * gaussian_model["means"] * uv_mask
        # apply annealed offsets in the canonical space
        shaped_pos_texture = pos_texture.to(dtype) + offsets

        # local LBS pass for points
        local_posed_pos_texture = self.mesh_lbs_layer.points_apply_lbs(
            rearrange(shaped_pos_texture, "b c h w -> b h w c"),
            lbs_kwargs["texel_lbs_weights"].to(dtype),
            lbs_kwargs["joint_translation"].to(dtype),
            lbs_kwargs["T_world_joint"].to(dtype),
        )

        # global pose warp
        global_posed_pos_texture, _ = self.mesh_lbs_layer.global_transform(
            local_posed_pos_texture,
            lbs_kwargs["T_world_joint"][..., :3, 3].to(dtype),
            lbs_kwargs["global_rotation"].to(dtype),
            lbs_kwargs["global_translation"].to(dtype),
        )

        gaussian_model["means"] = rearrange(global_posed_pos_texture, "b h w c -> b c h w")

        # local LBS pass for quaternions
        quat_texture = torch.tanh(gaussian_model["quats"])
        quats = self.mesh_lbs_layer.quaternions_apply_lbs(
            rearrange(quat_texture, "b c h w -> b h w c"),
            lbs_kwargs["texel_lbs_weights"].to(dtype),
            lbs_kwargs["T_world_joint"].to(dtype),  # translations are irrelevant for quaternions
        )
        gaussian_model["quats"] = rearrange(quats, "b h w c -> b c h w")

        gaussian_model["colors"] = torch.sigmoid(gaussian_model["colors"])

        base_scales = torch.sigmoid(gaussian_model["scales"]).clip(1e-5, 1) * uv_mask
        reference_scales = self.get_texel_scales(self.trainer_config.texture_resolution) * self.trainer_config.scale_multiplier
        gaussian_model["scales"] = base_scales * reference_scales.expand_as(base_scales)  # let them be a bit larger

        gaussian_model["opacities"] = torch.sigmoid(gaussian_model["opacities"]) * uv_mask
        gaussian_model["offsets"] = offsets  # raw per-Gaussian offsets, used for regularization
        gaussian_model["uv_mask"] = uv_mask  # mask for valid texels

        return gaussian_model

    def configure_optimizers(self):
        logger.info("Setting up the optimizer...")
        # Handle both full optimizers and partial ones
        assert callable(self._optimizer), "Optimizer must be partially instantiated from the config"
        assert callable(self._scheduler) or self._scheduler is None, "Scheduler must be partially instantiated from the config"
        optimizer: Optimizer = self._optimizer(self.model.parameters())
        scheduler: LRScheduler = self._scheduler(optimizer) if self._scheduler else NoOpScheduler(optimizer)

        self.base_lr = optimizer.param_groups[0]["lr"]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _add_texel_scales_for_resolution(self, texture_resolution: tuple[int, int], scales: torch.Tensor) -> None:
        """Add texel scales for a specific resolution."""
        resolution_key = f"{texture_resolution[0]}"

        if resolution_key in self.texel_scales:
            logger.info(f"Overwriting existing texel scales for resolution {resolution_key}")

        self.texel_scales[resolution_key] = torch.nn.Parameter(
            scales.to(self.device, dtype=torch.float32),
            requires_grad=False,
        )
        logger.info(f"Added texel scales for resolution {resolution_key}: {scales.shape}")

    def get_texel_scales(self, texture_resolution: tuple[int, int]) -> torch.Tensor:
        """
        Get texel scales for a given texture resolution.
        If not available, create default scales.
        """
        resolution_key = f"{texture_resolution[0]}"

        if resolution_key in self.texel_scales:
            return self.texel_scales[resolution_key].to(self.device)
        else:
            logger.warning(
                f"No precomputed texel scales found for resolution {resolution_key}, "
                f"creating default scaling from the config. Consider pre-computing scales for better results."
            )
            # Create default scales and add them to the ParameterDict for future use
            default_scales = torch.ones(1, 1, *texture_resolution, device=self.device, dtype=torch.float32)
            self._add_texel_scales_for_resolution(texture_resolution, default_scales)
            return default_scales

    def on_save_checkpoint(self, checkpoint):
        """Save texel scales with the checkpoint."""
        super().on_save_checkpoint(checkpoint)

        # The ParameterDict will be automatically saved, but we can add metadata
        checkpoint["texel_scales_metadata"] = {"resolutions": list(self.texel_scales.keys()), "num_resolutions": len(self.texel_scales)}

        if self.texel_scales:
            logger.info(f"Saving texel scales for {len(self.texel_scales)} resolutions: {list(self.texel_scales.keys())}")

    def on_load_checkpoint(self, checkpoint):
        # if model is compiled and checkpoint was not (and vice versa), we need to rename module keys in the checkpoint
        ckpt_compiled = any("_orig_mod" in key for key in checkpoint["state_dict"].keys())

        if self.compile_flag != ckpt_compiled:
            logger.info("Loaded checkpoint and current model have different compilation status, attempting to rename keys in state dict...")
            new_state_dict = {}
            skip_keys = ["loss_criterion", "deformer", "lbs_layer", "texel_scales"]
            for key, val in checkpoint["state_dict"].items():
                old_key_splits = key.split(".")
                first_key = old_key_splits[0]
                if first_key in skip_keys:
                    new_key = key
                elif ckpt_compiled:
                    new_key = ".".join([first_key, *old_key_splits[2:]])
                else:
                    new_key = ".".join([first_key, "._orig_mod", *old_key_splits[1:]])

                new_state_dict[new_key] = val
            checkpoint["state_dict"] = new_state_dict

        current_resolution = f"{self.trainer_config.texture_resolution[0]}"

        # Reconstruct ParameterDict structure based on checkpoint content
        if "texel_scales_metadata" in checkpoint:
            ckpt_resolutions = checkpoint["texel_scales_metadata"]["resolutions"]
            logger.info(f"Checkpoint contains texel scales for resolutions: {ckpt_resolutions}")

            # Pre-create parameters for all resolutions found in checkpoint
            for res_key in ckpt_resolutions:
                self.texel_scales[res_key] = checkpoint["state_dict"][f"texel_scales.{res_key}"]
                logger.info(
                    f"Loaded texel scales for resolution {res_key}, min/max: {self.texel_scales[res_key].min()}/{self.texel_scales[res_key].max()}"
                )

            if current_resolution not in ckpt_resolutions:
                logger.warning(
                    f"Current resolution {current_resolution} not found in checkpoint. "
                    f"Texel scales for this resolution will be created after loading."
                )
        else:
            # Also check for texel_scales directly in state_dict for older checkpoints
            texel_scale_keys = [k for k in checkpoint["state_dict"].keys() if k.startswith("texel_scales.")]
            if texel_scale_keys:
                ckpt_resolutions = [k.split("texel_scales.")[1] for k in texel_scale_keys]
                logger.info(f"Found texel scales in state_dict for resolutions: {ckpt_resolutions}")

                # Pre-create parameters for all resolutions found in state_dict
                for res_key in ckpt_resolutions:
                    self.texel_scales[res_key] = checkpoint["state_dict"][f"texel_scales.{res_key}"]
                    logger.info(
                        f"Loaded texel scales for resolution {res_key}, min/max: {self.texel_scales[res_key].min()}/{self.texel_scales[res_key].max()}"
                    )
            else:
                logger.info("No texel scales metadata found in checkpoint")

        super().on_load_checkpoint(checkpoint)

        # After loading, check if we need to add scales for current resolution
        current_resolution = f"{self.trainer_config.texture_resolution[0]}"
        if current_resolution not in self.texel_scales:
            logger.info(f"Creating default texel scales for current resolution {current_resolution}")
            default_scales = torch.ones(1, 1, *self.trainer_config.texture_resolution, device=self.device, dtype=torch.float32)
            self._add_texel_scales_for_resolution(self.trainer_config.texture_resolution, default_scales)

    def on_fit_start(self):
        logger.info(f"Preparing rasterization context on device {self.device}...")
        self.nvdiffrast_ctx = dr.RasterizeCudaContext(device=self.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        num_warmup_steps = self.trainer_config.num_warmup_steps
        if self.global_step < num_warmup_steps:
            lr_scale = min(1.0, float(self.global_step + 1) / num_warmup_steps)
            self.lr_schedule_scale = lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.base_lr
        else:
            self.lr_schedule_scale = 1.0

        if self.global_rank == 0:
            self.log("lr", optimizer.param_groups[0]["lr"], sync_dist=False)

    @property
    def offset_annealing(self) -> float:
        warming_up = self.global_step < self.trainer_config.num_warmup_steps if self.phase != "predict" else False
        warmup_multiplier = min(1.0, float(self.global_step + 1) / self.trainer_config.num_warmup_steps)
        return warmup_multiplier * warming_up + (1 - warming_up) * 1.0

    @torch.no_grad()
    def eval_metrics(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        psnr = mng_metrics.eval_psnr(pred, target, mask)
        ssim = mng_metrics.eval_ssim(pred, target, mask)
        lpips = mng_metrics.eval_lpips(pred, target, mask)
        metrics = {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
        }
        return metrics

    @torch.no_grad()
    def log_step(
        self,
        batch: dict,
        batch_idx: int,
        inputs: dict[str, Tensor],
        outputs: Tensor,
        prediction: dict[str, Tensor],
        targets: list[ExtendedImageBatch],
        loss_dict: dict[str, Tensor],
        lbs_kwargs: dict[str, Tensor] = None,
    ) -> None:
        logger_object = self.logger

        self.log(
            f"{self.phase}/loss",
            sum(loss_dict.values()),
            sync_dist=True,
        )
        for key, value in loss_dict.items():
            self.log(f"{self.phase}/{key}", value, sync_dist=True)

        # Log min/max values for scales and offsets
        if "scales" in prediction:
            self.log(f"{self.phase}/scales_min", prediction["scales"].min(), sync_dist=True)
            self.log(f"{self.phase}/scales_max", prediction["scales"].max(), sync_dist=True)

        if "offsets" in prediction:
            self.log(f"{self.phase}/offsets_min", prediction["offsets"].min(), sync_dist=True)
            self.log(f"{self.phase}/offsets_max", prediction["offsets"].max(), sync_dist=True)

        self.log("offset_annealing", self.offset_annealing, sync_dist=True)

        # during validation, global_step is not updated
        step_counter = self.global_step + batch_idx * (self.phase == "val")
        vis_rate = self.logging_config.log_media_freq * (1 if self.phase == "train" else 25)
        metric_rate = self.logging_config.log_stats_freq

        input_image = torch.cat([item.image[:, :3] for item in batch["input_views"]], dim=0)
        target_image = torch.cat([item.image for item in targets], dim=0)
        target_rgb, target_mask = target_image.split([3, 1], dim=1)

        rendered_image = mng_utils.tensor_to_4d(outputs)[:, :3]
        if step_counter % metric_rate == 0 and self.global_step != 0:
            metrics = self.eval_metrics(rendered_image, target_rgb, target_mask)
            for key, value in metrics.items():
                self.log(f"{self.phase}_metrics/{key}", value, sync_dist=True)

        if step_counter % metric_rate == 0:
            hist_data = {}
            uv_mask = prediction["uv_mask"].detach() > 0
            for k, v in prediction.items():
                if k != "uv_mask":
                    v_slice = (
                        v.detach()[uv_mask.repeat(1, v.shape[1], 1, 1)].cpu().flatten()[::4].float().numpy()
                    )  # downsample for histograms

                    hist_data[f"{self.phase}_hist/{k}"] = wandb.Histogram(v_slice)
            logger_object.log_metrics(hist_data, step=step_counter)

        if step_counter % vis_rate == 0:
            logger_object.log_image(
                f"{self.phase}_media/input",
                [input_image],
                step=step_counter,
                file_type=["jpg"],
            )
            logger_object.log_image(
                f"{self.phase}_media/target",
                [target_rgb],
                step=step_counter,
                file_type=["jpg"],
            )
            logger_object.log_image(
                f"{self.phase}_media/prediction",
                [rendered_image],
                step=step_counter,
                file_type=["jpg"],
            )

            logger_object.log_image(
                f"{self.phase}_media/texture/rgb_input",
                [mng_utils.tensor_to_4d(inputs["appearance"])[:, :3]],
                step=step_counter,
                file_type=["jpg"],
            )
            logger_object.log_image(
                f"{self.phase}_media/texture/rgb_prediction",
                [mng_utils.tensor_to_4d(prediction["colors"])],
                step=step_counter,
                file_type=["jpg"],
            )
            logger_object.log_image(
                f"{self.phase}_media/texture/opac_prediction",
                [mng_utils.tensor_to_4d(prediction["opacities"])],
                step=step_counter,
                file_type=["jpg"],
            )


class GIGAv2(GIGA):
    """
    This version of GIGA also takes a precomputed static texture as input.
    """

    @torch.autocast(device_type="cuda", dtype=torch.float32, enabled=True)
    def prepare_inputs(self, batch: dict, eval: bool) -> tuple[dict, dict]:
        inputs, lbs_kwargs = super().prepare_inputs(batch, eval)

        prob = torch.rand(inputs["appearance"].shape[0], device=self.device) < self.trainer_config.texture_dropout
        texture = tvtransforms.resize(
            batch["texture"], self.trainer_config.texture_resolution
        )  # (b 3 h w) tensor, RGB texture for the A-pose
        zero_texture = torch.ones_like(texture) * 0.5  # will be 0 after normalization

        texture = (
            texture if eval else torch.where(prob.view(-1, 1, 1, 1), texture, zero_texture)
        )  # blend in zero texture with the A-pose texture, randomly dropping it during training
        inputs["appearance"] = torch.cat(
            [
                inputs["appearance"],  # dynamic texture, already in [-1, 1] range
                texture * 2.0 - 1.0,  # normalize to [-1, 1] range
            ],
            dim=1,
        )

        return inputs, lbs_kwargs
