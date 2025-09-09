from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

dim_map = {
    "none": None,
    "first": 1,
    "last": -1,
}


def l1(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
    dim: Literal["none", "first", "last"] = "none",
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute the L1 loss between x and y.

    Args:
        x: First tensor
        y: Second tensor
        dim: Channel dimension to compute norm over ("none", "first", "last")

    Returns:
        L1 loss with shape [...] (norm computed over specified dimension)
    """
    dim = dim_map[dim]
    if dim is None:
        score = (x - y).abs()
    else:
        score = (x - y).abs().sum(dim=dim)
    return score.mean()


def l2(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
    dim: Literal["none", "first", "last"] = "none",
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute the squared L2 loss between x and y.

    Args:
        x: First tensor
        y: Second tensor
        dim: Channel dimension to compute norm over ("none", "first", "last")

    Returns:
        L2 loss with shape [...] (norm computed over specified dimension)
    """
    dim = dim_map[dim]
    if dim is None:
        score = (x - y).pow(2)
    else:
        score = (x - y).pow(2).sum(dim=dim)
    return score.mean()


def huber(
    x: Float[Tensor, "..."],
    y: Float[Tensor, "..."],
    delta: float = 1.0,
    dim: Literal["none", "first", "last"] = "none",
    **kwargs,
) -> Float[Tensor, "..."]:
    """Compute the Huber loss between x and y.

    Args:
        x: First tensor
        y: Second tensor
        delta: Threshold where loss transitions from L2 to L1 (default: 1.0)
        dim: Channel dimension to compute norm over ("none", "first", "last")

    Returns:
        Huber loss with shape [...] (norm computed over specified dimension)
    """
    dim = dim_map[dim]
    abs_diff = (x - y).abs()
    quadratic = 0.5 * (abs_diff**2)
    linear = delta * abs_diff - 0.5 * (delta**2)
    loss = torch.where(abs_diff <= delta, quadratic, linear)

    if dim is None:
        score = loss
    else:
        score = loss.sum(dim=dim)
    return score.mean()


NORMS = {
    "l1": l1,
    "l2": l2,
    "huber": huber,
}
