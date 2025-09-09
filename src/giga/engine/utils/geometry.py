import torch
from jaxtyping import Float
from torch import Tensor


def compute_vertex_normals(
    vertices: Float[Tensor, "vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, "vertices 3"]:
    """
    Computes per-vertex unit normals for a mesh, with vertices and faces being
    torch tensors.
    """

    v_normals = torch.zeros_like(vertices)
    v_faces = vertices[faces]

    f_normals = torch.cross(
        v_faces[:, 2] - v_faces[:, 1],
        v_faces[:, 0] - v_faces[:, 1],
        dim=-1,
    )

    v_normals = v_normals.index_add(0, faces[:, 0], f_normals)
    v_normals = v_normals.index_add(0, faces[:, 1], f_normals)
    v_normals = v_normals.index_add(0, faces[:, 2], f_normals)

    v_normals = torch.nn.functional.normalize(v_normals, eps=1e-6, dim=-1)

    return v_normals


def batched_compute_vertex_normals(
    vertices: Float[Tensor, "b vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, "b vertices 3"]:
    """
    Computes per-vertex unit normals for a batch of meshes, with vertices and faces being
    torch tensors.
    NOTE: The topology of the mesh is fixed, so the same faces array is used for each mesh in the batch.
    """

    v_normals = torch.zeros_like(vertices)
    bs = vertices.shape[0]
    for idx in range(bs):
        v_normals[idx] = compute_vertex_normals(vertices[idx], faces)

    return v_normals


def triangle_area(
    vertices: Float[Tensor, "vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, " faces "]:
    """
    Computes the area of each triangle in 3D space.

    Args:
        vertices: Vertex positions in 3D space
        faces: Face indices (triangles)

    Returns:
        areas: Area of each triangle
    """
    v_faces = vertices[faces]  # (faces, 3, 3)

    edge1 = v_faces[:, 1] - v_faces[:, 0]  # (faces, 3)
    edge2 = v_faces[:, 2] - v_faces[:, 0]  # (faces, 3)

    cross_product = torch.cross(edge1, edge2, dim=-1)  # (faces, 3)
    areas = 0.5 * torch.norm(cross_product, dim=-1)  # (faces,)

    return areas


def batched_triangle_area(
    vertices: Float[Tensor, "b vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, "b faces"]:
    """
    Computes the area of each triangle in 3D space for a batch of meshes.

    Args:
        vertices: Batch of vertex positions in 3D space
        faces: Face indices (triangles) - same topology for all meshes

    Returns:
        areas: Area of each triangle for each mesh in the batch
    """
    v_faces = vertices[:, faces]  # (b, faces, 3, 3)
    edge1 = v_faces[:, :, 1] - v_faces[:, :, 0]  # (b, faces, 3)
    edge2 = v_faces[:, :, 2] - v_faces[:, :, 0]  # (b, faces, 3)

    cross_product = torch.cross(edge1, edge2, dim=-1)  # (b, faces, 3)
    areas = 0.5 * torch.norm(cross_product, dim=-1)  # (b, faces)

    return areas


def per_vertex_triangle_areas(
    vertices: Float[Tensor, "vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, " vertices "]:
    """
    Computes the average area of triangles adjacent to each vertex.

    Args:
        vertices: Vertex positions in 3D space
        faces: Face indices (triangles)

    Returns:
        vertex_areas: Average triangle area for each vertex
    """
    face_areas = triangle_area(vertices, faces)  # (faces,)

    vertex_areas = torch.zeros(vertices.shape[0], device=vertices.device, dtype=vertices.dtype)
    vertex_counts = torch.zeros(vertices.shape[0], device=vertices.device, dtype=vertices.dtype)

    vertex_areas = vertex_areas.index_add(0, faces[:, 0], face_areas)
    vertex_areas = vertex_areas.index_add(0, faces[:, 1], face_areas)
    vertex_areas = vertex_areas.index_add(0, faces[:, 2], face_areas)

    ones = torch.ones_like(face_areas)
    vertex_counts = vertex_counts.index_add(0, faces[:, 0], ones)
    vertex_counts = vertex_counts.index_add(0, faces[:, 1], ones)
    vertex_counts = vertex_counts.index_add(0, faces[:, 2], ones)

    vertex_areas = vertex_areas / (vertex_counts + 1e-8)

    return vertex_areas


def mesh_surface_area(
    vertices: Float[Tensor, "vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, ""]:
    """
    Computes the total surface area of a mesh.

    Args:
        vertices: Vertex positions in 3D space
        faces: Face indices (triangles)

    Returns:
        total_area: Total surface area of the mesh
    """
    face_areas = triangle_area(vertices, faces)
    return face_areas.sum()


def batched_mesh_surface_area(
    vertices: Float[Tensor, "b vertices 3"],
    faces: Float[Tensor, "faces 3"],
) -> Float[Tensor, " b "]:
    """
    Computes the total surface area of each mesh in a batch.

    Args:
        vertices: Batch of vertex positions in 3D space
        faces: Face indices (triangles) - same topology for all meshes

    Returns:
        total_areas: Total surface area of each mesh in the batch
    """
    face_areas = batched_triangle_area(vertices, faces)  # (b, faces)
    return face_areas.sum(dim=1)  # (b,)
