import torch
from einops import rearrange
from gsplat.rendering import rasterization
from jaxtyping import Float
from torch import Tensor


def rasterize(
    gaussians: dict[str, Float[Tensor, "... n c"]],
    extrinsics: Float[Tensor, "... n 4 4"],
    intrinsics: Float[Tensor, "... n 3 3"],
    resolution: tuple[int, int],
    bg_color: Tensor | None = None,
    **gsplat_kwargs,
) -> tuple[Float[Tensor, "... n c h w"], Float[Tensor, "... n 1 h w"]]:
    """
    Performs 3D Gaussian Splatting with nerfstudio gsplat rasterizer.
    Args:
        gaussians (dict[str, Float[Tensor, "... n c"]]): Dictionary of Gaussian parameters. Should be contiguous.
        extrinsics (Float[Tensor, "... n 4 4"]): Extrinsic camera parameters.
        intrinsics (Float[Tensor, "... n 3 3"]): Intrinsic camera parameters.
        bg_color (Tensor | None): Background color for rendering.
        **gsplat_kwargs: Additional arguments for the gsplat rasterizer.
    Returns:
        tuple (Float[Tensor, "... n c h w"], Float[Tensor, "... n 1 h w"]): Rendered image (RGB, D, or RGB+D), and alpha mask.
    """

    orig_dtype = gaussians["means"].dtype

    if bg_color is not None:
        bg_color = bg_color.to(intrinsics).to(torch.float32)

    image, alpha, _ = rasterization(
        means=gaussians["means"].to(torch.float32),
        quats=gaussians["quats"].to(torch.float32),
        scales=gaussians["scales"].to(torch.float32),
        opacities=gaussians["opacities"].squeeze(-1).to(torch.float32),  # in case there is 1D last dim
        colors=gaussians["colors"].to(torch.float32),
        viewmats=extrinsics.to(torch.float32),
        Ks=intrinsics.to(torch.float32),
        backgrounds=bg_color,
        width=resolution[1],
        height=resolution[0],
        packed=False,
        **gsplat_kwargs,
    )

    image = rearrange(image, "... h w c -> ... c h w").to(orig_dtype)
    alpha = rearrange(alpha, "... h w 1 -> ... 1 h w").to(orig_dtype)

    return image, alpha
