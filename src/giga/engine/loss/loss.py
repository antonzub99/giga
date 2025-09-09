from functools import partial
from typing import Callable

import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from torch.nn import Module
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .norm import NORMS
from .perceptual import NoOpLPIPS, perceptual_patch_loss
from .reconstruction import alpha_channel, rgb_channel
from .regularizer import beta_regularization, l2_regularization, transparent_offsets, transparent_scales, uv_opacity
from .ssim import masked_ssim_loss

# these will need one of NORMS as loss_fn input
LOSS_COLLECTION = {
    "rgb": rgb_channel,
    "alpha": alpha_channel,
    "ssim": masked_ssim_loss,
}

REGULARIZER_COLLECTION = {
    "opacities_beta": beta_regularization,
    "offsets_mean": l2_regularization,
    "transparent_offsets": transparent_offsets,
    "transparent_scales": transparent_scales,
    "uv_opacity": uv_opacity,
}


def list_supported_losses() -> list[str]:
    """
    List all supported losses and regularizers.
    Returns:
        list[str]: List of supported loss names.
    """
    return (
        list(LOSS_COLLECTION.keys())
        + [
            "perceptual",  # added in the LossComputer constructor if specified in loss_weights
        ]
        + list(REGULARIZER_COLLECTION.keys())
    )


def get_recon_function(query: str, loss_dict: dict[str, Callable]) -> tuple[Callable, str | None] | tuple[None, None]:
    for key in loss_dict.keys():
        if query.startswith(key):
            norm = query[len(key) :].lstrip("_")
            return loss_dict[key], norm if norm else None
    return None, None


class LossComputer(Module):
    """
    Loss computer for the model.
    This class computes both the image-space reconstruction losses and various regularizers.
    """

    def __init__(
        self,
        compile: bool,
        loss_weights: dict[str, float],
        regularizer_weights: dict[str, float] | None = None,
    ):
        super().__init__()

        # also specify regularizer weights here and parse them into a different dict
        self.loss_weights = loss_weights  # should be like {"rgb_l2": 0.5, "ssim": 0.5, "alpha_l2": 0.5, "perceptual": 0.1}
        self.regularizer_weights = regularizer_weights or {}

        loss_collection = {}

        if "perceptual" in self.loss_weights.keys():
            lpips_network = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            if compile:
                self.lpips_network = torch.compile(lpips_network, mode="reduce-overhead")
            else:
                self.lpips_network = lpips_network
            loss_collection["perceptual"] = partial(perceptual_patch_loss, network=self.lpips_network)
            logger.info("Added perceptual loss function to the loss collection.")
        else:
            self.lpips_network = NoOpLPIPS()

        for loss_key in set(loss_weights.keys()).difference({"perceptual"}):
            loss_func, norm = get_recon_function(loss_key, LOSS_COLLECTION)
            if loss_func is not None:
                if norm is not None:
                    assert norm in NORMS, f"Norm {norm} is not supported. Supported norms are: {list(NORMS.keys())}."
                    recon_loss = partial(loss_func, loss_fn=NORMS[norm])
                else:
                    # this is ssim for sure
                    if compile:
                        recon_loss = torch.compile(loss_func, mode="reduce-overhead")
                    else:
                        recon_loss = loss_func
                loss_collection[loss_key] = recon_loss
                logger.info(f"Added loss function {loss_key} with norm {norm} to the loss collection.")
            else:
                raise Warning(
                    f"Loss key {loss_key} does not match any supported loss function. Supported losses are: {list(LOSS_COLLECTION.keys())}."
                )

        regularizer_collection = {key: val for key, val in REGULARIZER_COLLECTION.items() if key in self.regularizer_weights.keys()}

        assert all(key in regularizer_collection.keys() for key in self.regularizer_weights.keys()), (
            f"Regularizer weights keys {self.regularizer_weights.keys()} are not in the supported regularizer collection {regularizer_collection.keys()}."
        )
        logger.info(
            f"Using regularizers: {', '.join(regularizer_collection.keys())} with weights: {', '.join(map(str, self.regularizer_weights.values()))}."
        )

        self.loss_collection = loss_collection
        self.regularizer_collection = regularizer_collection

    def forward(
        self,
        inputs: Float[Tensor, "b c h w"],
        targets: Float[Tensor, "b c h w"],
        patch_meta: dict[str, Tensor] | None,
        masking: bool = True,
    ) -> dict[str, Tensor]:
        mask = (
            torch.cat(
                [
                    inputs[:, 3:4, :, :].repeat(1, 3, 1, 1),  # alpha channel as mask for RGB
                    torch.ones(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3], device=inputs.device, dtype=inputs.dtype),
                ],
                dim=1,
            )
            if masking
            else torch.ones_like(inputs)
        )
        loss_dict = {}
        for loss_name, loss_func in self.loss_collection.items():
            loss_dict[loss_name] = self.loss_weights.get(loss_name, 0) * loss_func(inputs * mask, targets, **(patch_meta or {}))

        return loss_dict

    def regularize(
        self,
        inputs: dict[str, Tensor],  # should be a dict with gaussian parameters
        opacity_threshold: float | None = None,
        scales_threshold: float | Tensor | None = None,
    ) -> dict[str, Tensor]:
        reg_dict = {}
        for reg_name, reg_func in self.regularizer_collection.items():
            reg_dict[reg_name] = self.regularizer_weights.get(reg_name, 0) * reg_func(
                inputs, scales_threshold=scales_threshold, opacity_threshold=opacity_threshold
            )
        return reg_dict
