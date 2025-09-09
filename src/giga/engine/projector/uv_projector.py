from typing import Literal, Union

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float, Int
from torch import Tensor

from giga.data import ExtendedImageBatch

from ..renderer.mesh import clip_projection

NvRastContext = Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]
ANGLE_THRESHOLD = 0.33
DEPTH_THRESHOLD = 0.33


def compute_visibility(
    mesh: dict[str, Tensor],
    cameras: ExtendedImageBatch,
    rast_ctx: NvRastContext,
) -> tuple[Float[Tensor, "b v"], Int[Tensor, "b h w 1"]]:
    """
    Computes visibility mask for mesh vertices from a batch of cameras. All inputs must reside on GPU.
    Args:
        mesh (dict[str, Tensor]): The mesh data containing vertices and faces.
        cameras (ExtendedImageBatch): The camera parameters for rendering.
        rast_ctx (NvRastContext): The rasterization context to use.
    Returns:
        tuple (Float[Tensor, "b v"], Int[Tensor, "b h w 1"]): A tuple containing the vertex visibility mask and rasterized face IDs.
    """

    assert cameras.intrinsics.device == mesh["vertices"].device != "cpu", (
        "Camera intrinsics and mesh vertices must be on the same GPU device"
    )

    clip_vertices = clip_projection(
        mesh["vertices"],
        cameras.intrinsics,
        cameras.extrinsics,
        cameras.resolution[0],
        cameras.resolution[1],
    )

    rast_out, rast_out_db = dr.rasterize(
        rast_ctx,
        clip_vertices.contiguous(),
        mesh["faces"],
        resolution=cameras.resolution,
    )
    num_views = cameras.intrinsics.shape[0]
    face_ids = rast_out[..., -1:].long()
    vertex_visibility = torch.zeros((num_views, mesh["vertices"].shape[0]), device=clip_vertices.device, dtype=torch.bool)
    # we have to loop over views, because operations with unique cannot be naively vectorized
    for idx in range(num_views):
        buf = face_ids[idx]
        valid_faces = buf[buf > 0]
        visibile_faces = torch.unique(mesh["faces"][valid_faces - 1], dim=0).flatten()
        visibile_vertices = torch.unique(visibile_faces)
        vertex_visibility[idx, visibile_vertices] = True

    return vertex_visibility, face_ids


def unproject_texture(
    mesh: dict[str, Tensor],
    cameras: ExtendedImageBatch,
    rast_ctx: NvRastContext,
    uv_barycentrics: Float[Tensor, "b h w c"] | None = None,
    texture_resolution: tuple[int, int] = (1024, 1024),
) -> dict[str, Tensor]:
    """
    Computes texture by unprojecting mesh texels onto a batch of camera views and gathering the colors from the camera images.
    All inputs must reside on GPU.
    Args:
        mesh (dict[str, Tensor]): The mesh data containing vertices, faces, and attributes.
        cameras (ExtendedImageBatch): The camera parameters for rendering.
        rast_ctx (NvRastContext): The rasterization context to use.
        uv_barycentrics (Float[Tensor, "b h w c"] | None): Optional UV barycentric coordinates for the texture.
        texture_resolution (tuple[int, int]): The resolution of the output texture.

    Returns:
        dict ([str, Tensor]): A dictionary containing the sampled texture, view angle, depth visibility, and occupied texels.
    """
    assert cameras.intrinsics.device == mesh["vertices"].device != "cpu", (
        "Camera intrinsics and mesh vertices must be on the same GPU device"
    )

    vertices, faces = mesh["vertices"], mesh["faces"]
    normals = mesh["normals"]
    vertex_uv_coords = mesh["attrs"]

    assert vertices.ndim == 2 and vertices.shape[1] == 3, "Vertices must be a 2D tensor with shape (N, 3)"
    assert faces.ndim == 2 and faces.shape[1] == 3, "Faces must be a 2D tensor with shape (M, 3)"
    assert normals.ndim == 2 and normals.shape[1] == 3, "Normals must be a 2D tensor with shape (N, 3)"
    assert vertex_uv_coords.ndim == 2 and vertex_uv_coords.shape[0] == vertices.shape[0], (
        "Vertex UV coordinates (stored as vertex attributes) must be a 2D tensor with shape (N, 2) matching the number of vertices"
    )

    if uv_barycentrics is None:
        vertex_uv_coords_h = torch.cat(
            [vertex_uv_coords, torch.ones_like(vertex_uv_coords[..., :2])], dim=-1
        ).contiguous()  # Convert to homogeneous coordinates
        vertex_uv_coords_h = vertex_uv_coords_h * 2.0 - 1.0  # Normalize to [-1, 1] range
        uv_barycentrics, _ = dr.rasterize(rast_ctx, vertex_uv_coords_h.unsqueeze(0), faces, resolution=texture_resolution)

    # Compute texel positions and normals, each in a 1 x T_H x T_W x 3 tensor
    interp_pos, _ = dr.interpolate(vertices, uv_barycentrics, faces)
    interp_normals, _ = dr.interpolate(normals, uv_barycentrics, faces)

    num_views = cameras.intrinsics.shape[0]
    interp_pos = interp_pos.expand(num_views, -1, -1, -1)  # b x T_H x T_W x 3
    interp_normals = interp_normals.expand(num_views, -1, -1, -1)  # b x T_H x T_W x 3
    non_empty_texels = (interp_pos != 0).any(dim=-1).unsqueeze(-1)  # b x T_H x T_W x 1

    # Estimate texel visibilty based on the angle between the view direction and the normal
    cam_origin = torch.inverse(cameras.extrinsics)[..., :3, 3][:, None, None]  # b x 1 x 1 x 3
    view_dir = torch.nn.functional.normalize(cam_origin - interp_pos, dim=-1, eps=1e-6)
    view_angle = (
        torch.sum(view_dir * interp_normals, dim=-1, keepdim=True).clip(0, 1).permute(0, 3, 1, 2)
    )  # b x 1 x T_H x T_W, backward facing texels are clipped to 0
    # Now gather texel colors from the camera images
    clip_pos = clip_projection(
        interp_pos[0],  # remove batch dim from input points - it will be broadcasted inside the function
        cameras.intrinsics,
        cameras.extrinsics,
        *cameras.image.shape[2:],
    )

    ndc_pos = clip_pos / clip_pos[..., -1:]  # b x T_H x T_W x 3
    ndc_pix_coords = ndc_pos[..., :2]
    sampled_colors = torch.nn.functional.grid_sample(cameras.image, ndc_pix_coords, align_corners=False)

    # Estimate depth visibility based on which vertices are visible in each individual view
    vertex_visibility, face_ids = compute_visibility(mesh, cameras, rast_ctx)
    # vertex_visibility = torch.tile(vertex_visibility.unsqueeze(-1), (1, 1, 3)).contiguous().float()  # b x w x 3
    vertex_visibility = vertex_visibility.unsqueeze(-1).contiguous().float()  # b x w x 1
    batched_uv_barycentrics = uv_barycentrics.expand(num_views, -1, -1, -1).contiguous()  # b x T_H x T_W x 3
    depth_visibility, _ = dr.interpolate(vertex_visibility, batched_uv_barycentrics, faces)
    depth_visibility = (depth_visibility * non_empty_texels).permute(0, 3, 1, 2)

    return {
        "texture": sampled_colors,
        "angle": view_angle,  # b x 1 x T_H x T_W
        "depth_visibility": depth_visibility,  # b x 1 x T_H x T_W
        "occupied_texels": non_empty_texels[0].squeeze(),  # T_H x T_W
    }


def collect_uv_texture(
    mesh: dict[str, Tensor],
    cameras: ExtendedImageBatch,
    rast_ctx: NvRastContext,
    uv_barycentrics: Float[Tensor, "b h w c"] | None = None,
    texture_resolution: tuple[int, int] = (1024, 1024),
    strategy: Literal["best", "softmax"] = "softmax",
    top_k: int = 4,
) -> Float[Tensor, "1 c h w"]:
    """
    Assembles the texture from the unprojected texels using the specified partial texture merging strategy.
    Args:
        mesh (dict[str, Tensor]): The mesh data containing vertices and faces.
        cameras (ExtendedImageBatch): The batch of cameras for rendering.
        rast_ctx (NvRastContext): The rasterization context to use.
        uv_barycentrics (Float[Tensor, "b h w c"] | None): Optional UV barycentric coordinates for the texture.
        texture_resolution (tuple[int, int]): The resolution of the output texture.
        strategy (Literal["best", "softmax"]): The merging strategy to use ("best" or "softmax").
        top_k (int): The number of top textures to consider.

    Returns:
        texture (Float[Tensor, "1 c h w"]): The assembled texture.
    """

    assert set(["vertices", "faces", "normals", "attrs"]).issubset(mesh.keys()), (
        "Mesh must contain 'vertices', 'faces', 'normals', and 'attrs' keys"
    )
    assert isinstance(cameras, ExtendedImageBatch), f"Cameras must be of type ExtendedImageBatch, got {type(cameras)}"

    buffers = unproject_texture(
        mesh,
        cameras,
        rast_ctx,
        uv_barycentrics=uv_barycentrics,
        texture_resolution=texture_resolution,
    )

    visibility_mask = torch.logical_and(buffers["angle"] > ANGLE_THRESHOLD, buffers["depth_visibility"] > DEPTH_THRESHOLD)
    visibility_mask = torch.logical_and(visibility_mask, buffers["occupied_texels"][None, None, ...])

    visibility_weights = torch.where(visibility_mask, buffers["angle"], torch.zeros_like(buffers["angle"]))
    mean_visibility_weights = visibility_weights.mean(dim=(2, 3))

    top_k = min(top_k, buffers["texture"].shape[0])
    _, top_indices = torch.topk(mean_visibility_weights.squeeze(), top_k)

    top_textures = buffers["texture"][top_indices]
    top_weights = visibility_weights[top_indices]
    top_visibility_mask = visibility_mask[top_indices].float()

    if top_textures.ndim == 3:
        top_textures = top_textures.unsqueeze(0)
        top_weights = top_weights.unsqueeze(0)
        top_visibility_mask = top_visibility_mask.unsqueeze(0)

    if strategy == "best":
        best_angles = torch.argsort(top_weights, dim=0, descending=True)
        best_angles = best_angles.expand(-1, top_textures.shape[1] + 1, -1, -1)
        top_textures = torch.cat([top_textures * top_visibility_mask, top_visibility_mask], dim=1)
        best_texture = torch.gather(top_textures, 0, best_angles)[0:1]
    elif strategy == "softmax":
        weights = torch.nn.functional.softmax(top_weights * 5, dim=0)
        best_texture = torch.sum(top_textures * top_visibility_mask * weights, dim=0, keepdim=True)
        best_visibility = torch.sum(top_visibility_mask * weights, dim=0, keepdim=True)
        best_texture = torch.cat([best_texture, best_visibility], dim=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'best' or 'softmax'.")

    return best_texture
