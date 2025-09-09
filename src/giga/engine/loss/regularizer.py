import torch
from jaxtyping import Float
from torch import Tensor


def l2_regularization(gaussian_model: dict[str, Float[Tensor, "b c h w"]], **kwargs) -> Tensor:
    """
    Apply L2 regularization to the Gaussian model parameters.
    Args:
        gaussian_model (dict[str, Float[Tensor, "b c h w"]]): Dictionary containing Gaussian model parameters.
    Returns:
        Tensor: Regularization loss value.
    """
    scores = gaussian_model["offsets"].square().sum(dim=1)
    return scores.mean()


def uv_opacity(gaussian_model: dict[str, Float[Tensor, "b c h w"]], **kwargs) -> Tensor:
    opacity = gaussian_model["opacities"]
    uv_mask = gaussian_model["uv_mask"]
    scores = (opacity - uv_mask).abs()
    return scores.mean()


def beta_regularization(gaussian_model: dict[str, Float[Tensor, "b c h w"]], **kwargs) -> Tensor:
    """
    Apply log-likelihood Beta distribution regularization to the opacities of the Gaussian model.
    The input is expected to be a tensor of values in the range [0, 1].
    https://github.com/facebookresearch/neuralvolumes/blob/main/models/neurvol1.py#L131
    Args:
        gaussian_model (dict[str, Float[Tensor, "b c h w"]]): Dictionary containing Gaussian model parameters.
    Returns:
        Tensor: Regularization loss value.
    """
    scores = (0.1 + gaussian_model["opacities"]).log() + (1.1 - gaussian_model["opacities"]).log() + 2.20727
    return scores.mean()


def transparent_offsets(gaussian_model: dict[str, Float[Tensor, "b c h w"]], opacity_threshold: float = 0.1, **kwargs) -> Tensor:
    """
    Apply transparent offsets regularization to the Gaussian model.
    Args:
        gaussian_model (dict[str, Float[Tensor, "b c h w"]]): Dictionary containing Gaussian model parameters.
    Returns:
        Tensor: Regularization loss value.
    """
    mask = (gaussian_model["opacities"] < opacity_threshold).detach()
    offsets = gaussian_model["offsets"]
    scores = torch.where(mask, offsets.abs(), torch.zeros_like(offsets))
    return scores.mean()


def transparent_scales(
    gaussian_model: dict[str, Float[Tensor, "b c h w"]],
    scales_threshold: Float[Tensor, " ... "] | float = 0.5,
    opacity_threshold: float = 0.1,
    **kwargs,
) -> Tensor:
    """
    Apply transparent scales regularization to the Gaussian model.
    Args:
        gaussian_model (dict[str, Float[Tensor, "b c h w"]]): Dictionary containing Gaussian model parameters.
        scales_threshold (Float[Tensor, " ... "] | float): Threshold for scales to consider transparency.
        opacity_threshold (float): Threshold for alpha channel to consider transparency.
    Returns:
        Tensor: Regularization loss value.
    """
    mask = (gaussian_model["opacities"] < opacity_threshold).detach()
    scales = gaussian_model["scales"]
    scores = torch.where(mask, (scales - scales_threshold).abs(), torch.zeros_like(scales))
    return scores.mean()
