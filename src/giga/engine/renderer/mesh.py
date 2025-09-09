from typing import Union

import nvdiffrast.torch as dr
import torch
from jaxtyping import Float
from torch import Tensor

from giga.data import ExtendedImageBatch

from .utils import opengl_projection_mtx

NvRastContext = Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]


def create_area_lights(cameras: ExtendedImageBatch, num_lights: int = 16) -> Float[Tensor, "n_lights 3"]:
    """
    Creates area lights in the upper hemisphere of the camera dome.
    Args:
        cameras (ExtendedImageBatch): Camera batch containing extrinsics.
        num_lights (int): Number of lights to create.
    Returns:
        Float[Tensor, "n_lights 3"]: Light positions.
    """
    device = cameras.extrinsics.device
    dtype = cameras.extrinsics.dtype

    # Get camera positions from extrinsics (inverse translation)
    camera_positions = -cameras.extrinsics[:, :3, 3]  # (n_views 3)

    # Compute scene center as mean of camera positions
    scene_center = camera_positions.mean(dim=0)  # (3,)

    # Compute average distance from cameras to scene center
    avg_distance = torch.norm(camera_positions - scene_center, dim=1).mean()

    # Create lights in upper hemisphere around scene center
    # Use spherical coordinates: theta (azimuth), phi (elevation)
    theta = torch.linspace(0, 2 * torch.pi, num_lights, device=device, dtype=dtype)
    phi = torch.linspace(0, torch.pi / 2, int(torch.sqrt(torch.tensor(num_lights, dtype=torch.float32))), device=device, dtype=dtype)

    # Create grid of angles
    theta_grid, phi_grid = torch.meshgrid(theta[: int(torch.sqrt(torch.tensor(num_lights, dtype=torch.float32)))], phi, indexing="ij")
    theta_flat = theta_grid.flatten()[:num_lights]
    phi_flat = phi_grid.flatten()[:num_lights]

    # Convert to Cartesian coordinates
    light_radius = avg_distance * 1.5  # Place lights further than cameras
    x = light_radius * torch.sin(phi_flat) * torch.cos(theta_flat)
    y = light_radius * torch.sin(phi_flat) * torch.sin(theta_flat)
    z = light_radius * torch.cos(phi_flat)

    light_positions = torch.stack([x, y, z], dim=1) + scene_center  # (n_lights 3)

    return light_positions


def simple_shading(
    vertices: Float[Tensor, "n_verts 3"],
    normals: Float[Tensor, "n_verts 3"],
    light_positions: Float[Tensor, "n_lights 3"],
    light_intensity: float = 1.0,
    ambient: float = 0.1,
) -> Float[Tensor, "n_verts 3"]:
    """
    Simple Lambertian shading for mesh vertices.
    Args:
        vertices (Float[Tensor, "n_verts 3"]): Vertex positions.
        normals (Float[Tensor, "n_verts 3"]): Vertex normals.
        light_positions (Float[Tensor, "n_lights 3"]): Light positions.
        light_intensity (float): Light intensity.
        ambient (float): Ambient light coefficient.
    Returns:
        Float[Tensor, "n_verts 3"]: Shaded vertex colors.
    """
    # Normalize normals
    normals = torch.nn.functional.normalize(normals, dim=1)

    # Compute light directions for each vertex-light pair
    light_dirs = light_positions.unsqueeze(0) - vertices.unsqueeze(1)  # (n_verts n_lights 3)
    light_dirs = torch.nn.functional.normalize(light_dirs, dim=2)

    # Compute Lambertian shading
    dot_products = torch.sum(normals.unsqueeze(1) * light_dirs, dim=2)  # (n_verts n_lights)
    dot_products = torch.clamp(dot_products, 0.0, 1.0)

    # Sum contributions from all lights
    diffuse = torch.sum(dot_products, dim=1, keepdim=True) * light_intensity / light_positions.shape[0]  # (n_verts 1)

    # Add ambient lighting
    shading = ambient + diffuse
    shading = torch.clamp(shading, 0.0, 1.0)

    # Convert to RGB (grayscale shading)
    vertex_colors = shading.expand(-1, 3)  # (n_verts 3)

    return vertex_colors


def clip_projection(
    points: Float[Tensor, "... 3"],
    intrinsics: Float[Tensor, "b 3 3"],
    extrinsics: Float[Tensor, "b 4 4"],
    height: int,
    width: int,
    znear: float = 0.01,
    zfar: float = 10000.0,
) -> Float[Tensor, "... 4"]:
    """
    Projects 3D points into 2D image space using OpenGL projection.
    Args:
        points (Float[Tensor, "... 3"]): 3D points to project.
        intrinsics (Float[Tensor, "b 3 3"]): Camera intrinsics matrix.
        extrinsics (Float[Tensor, "b 4 4"]): Camera extrinsics matrix.
        height (int): Height of the image.
        width (int): Width of the image.
        znear (float): Near clipping plane distance.
        zfar (float): Far clipping plane distance.
    Returns:
        Float[Tensor, "... 4"]: Projected points in homogeneous coordinates.
    """
    projection = opengl_projection_mtx(intrinsics, height, width, znear, zfar)
    full_projection = projection @ extrinsics
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # Convert to homogeneous coordinates
    return torch.einsum("...i,bki->b...k", points_h, full_projection)  # Project points


def rasterize(
    mesh: dict[str, Tensor],
    cameras: ExtendedImageBatch,
    bg_color: Tensor | None = None,
    rast_ctx: NvRastContext | None = None,
    antialias: bool = False,
    **raster_kwargs,
) -> tuple[Float[Tensor, "b c h w"], Float[Tensor, "b 1 h w"]]:
    """

    Rasterizes the given mesh using the specified cameras and returns the rendered image and alpha mask.
    Args:
        mesh (dict[str, Tensor]): The mesh data containing vertices, faces, and attributes.
        cameras (ExtendedImageBatch): The camera parameters for rendering.
        bg_color (Tensor | None): Background color to use if no texture is present.
        rast_ctx (NVRastContext | None): The rasterization context to use.
        antialias (bool): Whether to apply antialiasing.
        **raster_kwargs: Additional keyword arguments for rasterization.
    Returns:
        tuple (Float[Tensor, "b c h w"], Float[Tensor, "b 1 h w"]): The rendered image and alpha mask.
    """
    assert "vertices" in mesh and "faces" in mesh, "Mesh must contain 'vertices' and 'faces'."
    assert "attrs" in mesh or "texture" in mesh, "Mesh must contain 'attrs' or 'texture'."
    assert "faces_attrs" in mesh, "Mesh must contain 'faces_attrs' with per-face vertex id triplets in the UV space."
    if rast_ctx is None:
        rast_ctx = dr.RasterizeCudaContext(device=cameras.intrinsics.device)

    intrinsics = cameras.intrinsics
    extrinsics = cameras.extrinsics
    num_views = intrinsics.shape[0]

    clip_vertices = clip_projection(mesh["vertices"], intrinsics, extrinsics, cameras.resolution[0], cameras.resolution[1])
    clip_vertices = clip_vertices.contiguous()

    rast_out, rast_out_db = dr.rasterize(rast_ctx, clip_vertices, mesh["faces"], resolution=cameras.resolution)

    # interpolated attrs are any vertex attributes
    # which also includes UV coordintes for texture mapping, if texture is present in the mesh dict
    attrs, attr_db = dr.interpolate(mesh["attrs"], rast_out, mesh["faces_attrs"], rast_out_db)

    if "texture" in mesh:
        attrs = dr.texture(mesh["texture"], attrs, attr_db, filter_mode="linear")

    if antialias:
        attrs = dr.antialias(attrs, rast_out, clip_vertices, mesh["faces"])

    if bg_color is None:
        bg_color = torch.zeros((num_views, 1, 1, 3), device=intrinsics.device, dtype=intrinsics.dtype)

    alpha = torch.clamp(rast_out[..., -1:], 0, 1)
    image = attrs * alpha + bg_color * (1 - alpha)

    return image.permute(0, 3, 1, 2), alpha.permute(0, 3, 1, 2)  # (b h w c) -> (b c h w)
