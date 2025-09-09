from typing import Callable

from jaxtyping import Float
from torch import Tensor


def rgb_channel(
    pred: Float[Tensor, "b 4 h w"],
    target: Float[Tensor, "b 4 h w"],
    loss_fn: Callable,
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute loss for RGB channels. Both inputs should already be masked in the RGB channels.

    Args:
        loss_fn: Loss function to apply
        pred: Predicted tensor with shape (batch, 4, height, width)
        target: Target tensor with shape (batch, 4, height, width)

    Returns:
        Loss value reduced over the RGB channels
    """

    assert pred.shape[1] == 4, f"Expected 4 channels (RGBA format) in prediction, got {pred.shape[1]}"
    assert target.shape[1] == 4, f"Expected 4 channels (RGBA format) in target, got {target.shape[1]}"

    return loss_fn(pred[:, :3, :, :], target[:, :3, :, :])


def alpha_channel(
    pred: Float[Tensor, "b 4 h w"],
    target: Float[Tensor, "b 4 h w"],
    loss_fn: Callable,
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute loss for Alpha channel.

    Args:
        loss_fn: Loss function to apply
        pred: Predicted tensor with shape (batch, 4, height, width)
        target: Target tensor with shape (batch, 4, height, width)

    Returns:
        Loss value for the alpha channel
    """

    assert pred.shape[1] == 4, f"Expected 4 channels (RGBA format) in prediction, got {pred.shape[1]}"
    assert target.shape[1] == 4, f"Expected 4 channels (RGBA format) in target, got {target.shape[1]}"

    return loss_fn(pred[:, 3:4, :, :], target[:, 3:4, :, :])
