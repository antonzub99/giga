# This file is derived from Viser (https://github.com/project/viser)
# Licensed under the Apache License 2.0:
#
# Modifications in this file:
# - Expanded to support SMPLX with texels
# - Complete rewrite of constructor and forward pass signature
# - Omni-gendered: all gender models are loaded; also handles both PCA and non-PCA hand parameters as input

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from loguru import logger
from torch import Tensor

from giga.geometry import SO3

from ..utils import geometry as mng_geometry


class TexelSMPLX(nn.Module):
    NUM_BODY_JOINTS = 24
    NUM_HAND_JOINTS = 15
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + 1  # +1 for global root
    NUM_BETAS = 300
    NUM_EXPR_COEFFS = 100

    def __init__(
        self,
        model_path: Path,
        uvmap_path: Path,
        gender: Literal["male", "female", "neutral", "all"] = "neutral",
        flat_hand_mean: bool = False,
        use_hand_pca: bool = True,
    ):
        super().__init__()

        assert uvmap_path is not None, "UV map path must be provided."
        assert uvmap_path.exists(), f"UV map path {uvmap_path} does not exist! Please check the path."
        self.gender = gender
        self.flat_hand_mean = flat_hand_mean
        self.use_hand_pca = use_hand_pca

        self._loaded_generic_data = False
        if gender != "all":
            logger.info(f"Loading {gender} SMPL-X model...")
            self._load_gender(model_path, gender)
        else:
            logger.info("Loading all SMPL-X models...")
            for _gender in ["male", "female", "neutral"]:
                self._load_gender(model_path, _gender)

        # Load UV parameterization
        uv_data = np.load(uvmap_path, allow_pickle=True)
        vt = torch.as_tensor(uv_data["vt"], dtype=torch.float32)  # 11313 x 2
        ft = torch.as_tensor(uv_data["ft"], dtype=torch.int32)
        self.register_buffer("vt", vt)
        self.register_buffer("ft", ft)

    def _load_gender(self, model_path: Path, gender: str):
        model_path = find_model_path(model_path, gender)
        assert model_path.exists(), f"Model path {model_path} does not exist!"
        model_data = np.load(model_path, allow_pickle=True)

        # Initialize body model
        J_regressor = torch.as_tensor(model_data["J_regressor"], dtype=torch.float32)
        lbs_weights = torch.as_tensor(model_data["weights"], dtype=torch.float32)
        v_template = torch.as_tensor(model_data["v_template"], dtype=torch.float32)  # 10475 x 3

        # Initialize shape blend shapes
        shapedirs = torch.as_tensor(model_data["shapedirs"], dtype=torch.float32)
        exprdirs = shapedirs[:, :, self.NUM_BETAS : self.NUM_BETAS + self.NUM_EXPR_COEFFS]
        shapedirs = shapedirs[:, :, : self.NUM_BETAS]

        # Initialize pose blend shapes
        posedirs = torch.as_tensor(model_data["posedirs"], dtype=torch.float32)
        n_basis = posedirs.shape[-1]
        posedirs = posedirs.reshape(-1, n_basis).T

        # Register buffers
        self.register_buffer(f"{gender}_shapedirs", shapedirs)
        self.register_buffer(f"{gender}_exprdirs", exprdirs)
        self.register_buffer(f"{gender}_posedirs", posedirs)
        self.register_buffer(f"{gender}_v_template", v_template)
        self.register_buffer(f"{gender}_J_regressor", J_regressor)
        self.register_buffer(f"{gender}_lbs_weights", lbs_weights)

        if not self._loaded_generic_data:
            parents = torch.as_tensor(model_data["kintree_table"][0], dtype=torch.int32)
            faces = torch.as_tensor(model_data["f"], dtype=torch.int32)

            # Initialize hands
            if self.use_hand_pca:
                lh_comps = torch.as_tensor(model_data["hands_componentsl"], dtype=torch.float32)
                rh_comps = torch.as_tensor(model_data["hands_componentsr"], dtype=torch.float32)
            else:
                lh_comps = torch.eye(3 * self.NUM_HAND_JOINTS, dtype=torch.float32)
                rh_comps = torch.eye(3 * self.NUM_HAND_JOINTS, dtype=torch.float32)
            self.hands_dim = 3 * self.NUM_HAND_JOINTS * 2
            hand_comps = torch.stack([lh_comps, rh_comps], dim=0)

            lh_mean = torch.as_tensor(model_data["hands_meanl"], dtype=torch.float32)
            rh_mean = torch.as_tensor(model_data["hands_meanr"], dtype=torch.float32)
            hand_mean = torch.stack([lh_mean, rh_mean], dim=0)

            # This one is essentially doing nothing
            if self.flat_hand_mean:
                hand_mean.zero_()

            self.register_buffer("hand_comps", hand_comps)
            self.register_buffer("hand_mean", hand_mean)
            self.register_buffer("parents", parents)
            self.register_buffer("faces", faces)
            self._loaded_generic_data = True

    @property
    def device(self) -> torch.device:
        """A convenience property to get the device where the model is placed."""
        return self.faces.device

    @property
    def uv_vertices(self) -> Float[Tensor, "V 2"]:
        """UV space coordinates of the vertices, texture boundary included."""
        return self.vt

    @property
    def uv_faces(self) -> Int[Tensor, "F 3"]:
        """Indexes of the vertices that make the faces of the template in UV space (can differ due to boundaries)."""
        return self.ft

    @property
    def template_with_boundary(self) -> Int[Tensor, " N "]:
        """A mapping of indices from vertices in UV space to vertices in 3D space, accounting for texture boundaries."""
        vt_to_v = torch.zeros(self.vt.shape[0], dtype=torch.int32, device=self.vt.device)
        vt_to_v[self.ft.flatten()] = self.faces.flatten()
        return vt_to_v

    def points_apply_lbs(
        self,
        points: Float[Tensor, "B *V 3"],
        lbs_weights: Float[Tensor, "B *V J"],
        joint_pos: Float[Tensor, "B J 3"],
        joint_transforms: Float[Tensor, "B J 4 4"],
    ) -> Float[Tensor, "B *V 3"]:
        batch_size = points.shape[0]
        spatial_dims = points.shape[1:-1]

        p_delta = torch.ones((batch_size, *spatial_dims, self.NUM_JOINTS, 4), device=points.device)
        interdims = points.ndim - 2
        joint_pos = joint_pos.view(joint_pos.shape[0], *([1] * interdims), *joint_pos.shape[1:])
        p_delta[..., :3] = points[..., None, :] - joint_pos
        p_posed = torch.einsum("bjxy,b...j,b...jy->b...x", joint_transforms[..., :3, :], lbs_weights, p_delta)

        return p_posed

    def quaternions_apply_lbs(
        self,
        quats: Float[Tensor, "B *V 4"],
        lbs_weights: Float[Tensor, "B *V J"],
        joint_transforms: Float[Tensor, "B J 4 4"],
    ) -> Float[Tensor, "B *V 4"]:
        rotmats = SO3.quaternion_to_matrix(quats)
        rotmats_posed = torch.einsum("bjxy,b...j,b...yz->b...xz", joint_transforms[..., :3, :3], lbs_weights, rotmats)
        quats_posed = SO3.matrix_to_quaternion(rotmats_posed)

        return quats_posed

    def get_joint_world_transforms(
        self,
        gender: Literal["male", "female", "neutral"],
        flat_hand: bool,
        hand_pose_dim: int,
        shape: Float[Tensor, "_ num_betas"],
        expression: Float[Tensor, "B num_expr_coeffs"],
        body_pose: Float[Tensor, "B 63"],
        hand_pose: Float[Tensor, "B num_hand_dim"],
        head_pose: Float[Tensor, "B 9"],
        pelvis_rotation: Float[Tensor, "B 3"],
        **kwargs,  # unused arguments from the forward pass
    ) -> tuple[Float[Tensor, "B V 3"], Float[Tensor, "B J 4 4"], Float[Tensor, "B J 3"]]:
        # Prepare pose
        batch_size = body_pose.shape[0]
        if shape.shape[0] == 1 and batch_size > 1:
            shape = shape.expand(batch_size, -1)
        dtype, device = self.get_buffer(f"{gender}_J_regressor").dtype, self.get_buffer(f"{gender}_J_regressor").device

        lhand_pose, rhand_pose = hand_pose[:, :hand_pose_dim].chunk(2, axis=-1)
        hand_mean = self.hand_mean if not flat_hand else torch.zeros_like(self.hand_mean)
        if hand_pose_dim == 90:  # happens only if no PCA is used
            lhand_comps = torch.eye(3 * self.NUM_HAND_JOINTS, dtype=dtype, device=device)
            rhand_comps = torch.eye(3 * self.NUM_HAND_JOINTS, dtype=dtype, device=device)
        else:
            lhand_comps = self.hand_comps[0, : lhand_pose.shape[-1]]
            rhand_comps = self.hand_comps[1, : rhand_pose.shape[-1]]
        lhand_pose = lhand_pose @ lhand_comps + hand_mean[0]
        rhand_pose = rhand_pose @ rhand_comps + hand_mean[1]
        hand_pose = torch.cat([lhand_pose, rhand_pose], dim=-1)

        pose = torch.cat([pelvis_rotation, body_pose, head_pose, hand_pose], dim=-1)
        pose = pose.reshape(batch_size, -1, 3)

        # Convert to rotation matrices
        pose_matrices = SO3.axisangle_to_matrix(pose)
        shape_comps = torch.cat([shape, expression], dim=-1)
        shape_dim = shape.shape[-1]
        expression_dim = expression.shape[-1]

        sd = torch.cat(
            [self.get_buffer(f"{gender}_shapedirs")[:, :, :shape_dim], self.get_buffer(f"{gender}_exprdirs")[:, :, :expression_dim]], dim=-1
        )
        v_t = self.get_buffer(f"{gender}_v_template") + torch.einsum("bi,nki->bnk", shape_comps, sd)

        # Compute joint locations
        j_t = torch.einsum("bik,ji->bjk", v_t, self.get_buffer(f"{gender}_J_regressor"))

        # Kinematic transforms setup
        T_parent_joint = torch.eye(4, device=device, dtype=dtype).expand(batch_size, self.NUM_JOINTS, 4, 4).clone()
        T_parent_joint[..., :3, :3] = pose_matrices
        T_parent_joint[:, 0, :3, 3] = j_t[:, 0]
        T_parent_joint[:, 1:, :3, 3] = j_t[:, 1:] - j_t[:, self.parents[1:]]

        T_world_joint = torch.empty((batch_size, self.NUM_JOINTS, 4, 4), device=device, dtype=dtype)
        T_world_joint[:, 0] = T_parent_joint[:, 0]

        for idx in range(1, self.NUM_JOINTS):
            parent_idx = self.parents[idx]
            T_world_joint[:, idx] = T_world_joint[:, parent_idx].clone() @ T_parent_joint[:, idx]

        identity = torch.eye(3, device=device, dtype=dtype)
        pose_delta = (pose_matrices[:, 1:] - identity).reshape(batch_size, -1)
        v_shaped = v_t + (pose_delta @ self.get_buffer(f"{gender}_posedirs")).reshape(batch_size, -1, 3)

        return v_shaped, T_world_joint, j_t

    def texelize_template(
        self,
        vertices: Float[Tensor, "B V 3"],
        gender: str,
        rast_ctx: dr.RasterizeCudaContext,
        resolution: tuple[int, int] = (512, 512),
    ) -> tuple[Tensor, ...]:
        """
        Texelize the template vertices and weights using the UV parameterization.

        Args:
            vertices: The vertices of the SMPL-X model in 3D space.
            weights: The weights of the SMPL-X model.

        Returns:
            A tuple containing the texelized vertices and weights.
        """
        batch_size = vertices.shape[0]
        vt = self.vt.expand(batch_size, -1, -1)  # (B, 11313, 2) for smplx
        vidx_with_boundary = self.template_with_boundary
        v_with_boundary = vertices[:, vidx_with_boundary].contiguous()

        weights = self.get_buffer(f"{gender}_lbs_weights").reshape(1, vertices.shape[1], self.NUM_JOINTS)
        weights = weights[:, vidx_with_boundary].expand(batch_size, -1, -1).contiguous()  # (B, 11313, 24) for smplx

        vt_hom = torch.cat([vt, torch.ones_like(vt)], dim=-1).contiguous() * 2.0 - 1.0
        rast_out, rast_out_db = dr.rasterize(rast_ctx, vt_hom, self.ft, resolution)
        texel_pos, _ = dr.interpolate(v_with_boundary, rast_out, self.ft)
        texel_weights, _ = dr.interpolate(weights, rast_out, self.ft)
        return (texel_pos, texel_weights, rast_out)

    def global_transform(
        self,
        vertices: Float[Tensor, "B *V 3"],
        joints: Float[Tensor, "B J 3"],
        global_rotation: Float[Tensor, "B 3"],
        global_translation: Float[Tensor, "B 3"],
    ) -> tuple[Float[Tensor, "B *V 3"], Float[Tensor, "B J 3"]]:
        """
        Apply global transformation to the vertices and joints.
        Args:
            vertices: The vertices of the SMPL-X model in 3D space.
            global_rotation: Global rotation of the body. It should not be used with pelvis_rotation.
            global_translation: Global translation of the body.
        Returns:
            A tuple containing the transformed vertices and joints.
        """

        # Apply global transformation to vertices and joints
        R, t = SO3.axisangle_to_matrix(global_rotation), global_translation
        num_spatial_dims = vertices.shape[1:-1]
        t_broadcasted = t.view(-1, *([1] * len(num_spatial_dims)), 3)  # Ensure t has the right shape for broadcasting
        v_posed = torch.einsum("bij,b...j->b...i", R, vertices) + t_broadcasted
        joints = (R @ joints.mT).mT + t.unsqueeze(1)

        return v_posed, joints

    def forward(
        self,
        gender: Literal["male", "female", "neutral"],
        flat_hand: bool,
        hand_pose_dim: int,
        shape: Float[Tensor, "_ num_betas"],
        expression: Float[Tensor, "B num_expr_coeffs"],
        body_pose: Float[Tensor, "B 63"],
        hand_pose: Float[Tensor, "B num_hand_dim"],
        head_pose: Float[Tensor, "B 9"],
        pelvis_rotation: Optional[Float[Tensor, "B 3"]] = None,
        pelvis_translation: Optional[Float[Tensor, "B 3"]] = None,
        global_rotation: Optional[Float[Tensor, "B 3"]] = None,
        global_translation: Optional[Float[Tensor, "B 3"]] = None,
    ) -> dict[str, Tensor]:
        """
        Apply forward kinematics and linear blend skinning to the SMPL-X model.

        Args:
            gender: Gender of the SMPL-X model to use for the operation.
            flat_hand: If True, use flat hand pose (mean pose) instead of the actual hand pose.
            hand_pose_dim: Dimensionality of the hand pose. If use_hand_pca is True, it should be <=90, otherwise 90.
            shape: Shape parameters of the body model.
            expression: Expression parameters of the body model.
            body_pose: Pose parameters for the body.
            hand_pose: Pose parameters for the hands. If use_hand_pca is True, the shape is (B, 12), otherwise (B, 90). We stack left and right hand poses along the last dimension.
            head_pose: Pose parameters for the head. We stack  jaw, left eye and right eye poses along the last dimension.
            pelvis_rotation: Rotation of the pelvis. It should not be used with global_rotation.
            pelvis_translation: Translation of the pelvis.
            global_rotation: Global rotation of the body. It should not be used with pelvis_rotation.
            global_translation: Global translation of the body.

        Returns:
            A dictionary containing the following keys:
            - vertices: The vertices of the SMPL-X model. The shape is (B, N, 3), where N is the number of vertices.
            - joints: The joint locations of the SMPL-X model. The shape is (B, J, 3), where J is the number of joints.
            - faces: The faces of the SMPL-X model. The shape is (F, 3), where F is the number of faces.
        """

        batch_size = body_pose.shape[0]
        dtype, device = self.get_buffer(f"{gender}_J_regressor").dtype, self.get_buffer(f"{gender}_J_regressor").device

        if pelvis_rotation is None:
            pelvis_rotation = torch.zeros((batch_size, 3), dtype=dtype, device=device)
        if global_rotation is None:
            global_rotation = torch.zeros((batch_size, 3), dtype=dtype, device=device)

        translation = global_translation if global_translation is not None else pelvis_translation
        if translation is None:
            translation = torch.zeros((batch_size, 3), dtype=dtype, device=device)

        if shape.shape[0] == 1 and batch_size > 1:
            shape = shape.expand(batch_size, -1)

        v_shaped, T_world_joint, j_t = self.get_joint_world_transforms(
            gender, flat_hand, hand_pose_dim, shape, expression, body_pose, hand_pose, head_pose, pelvis_rotation
        )

        weights = self.get_buffer(f"{gender}_lbs_weights").unsqueeze(0).expand(batch_size, -1, -1)
        v_posed = self.points_apply_lbs(v_shaped, weights, j_t, T_world_joint)
        joints = T_world_joint[..., :3, 3]

        R, t = SO3.axisangle_to_matrix(global_rotation), translation[:, None]
        v_posed = (R @ v_posed.mT).mT + t
        joints = (R @ joints.mT).mT + t

        return {
            "vertices": v_posed,
            "joints": joints,
            "faces": self.faces,
            "transforms_world": T_world_joint,
        }


def find_model_path(model_path: Optional[Path] = None, gender: str = "neutral") -> Path:
    if model_path is not None:
        return model_path / f"SMPLX_{gender.upper()}.npz"
    else:
        # If no path is found, raise an error
        raise FileNotFoundError(
            f"Could not find SMPL-X model path for gender {gender}. \nYou can download the SMPL-X model from https://smpl-x.is.tue.mpg.de/."
        )


def compute_texel_scales(
    lbs_model: TexelSMPLX,
    texture_resolution: tuple[int, int] = (256, 256),
    gender: Literal["male", "female", "neutral"] = "neutral",
):
    vertices = lbs_model.get_buffer(f"{gender}_v_template")  # (1, 10475, 3)
    faces = lbs_model.get_buffer("faces")  # (F, 3)

    per_vertex_areas = mng_geometry.per_vertex_triangle_areas(vertices, faces)
    uv_coords = lbs_model.uv_vertices  # (11313, 2)
    ft = lbs_model.uv_faces  # (F, 3)
    vt_to_v = lbs_model.template_with_boundary  # (11313,)

    device = vertices.device
    if device.type != "cuda":
        raise RuntimeError("Texel scales computation requires a CUDA device.")

    areas_w_boundary = per_vertex_areas[None, vt_to_v, None].contiguous().to(device)
    vt_hom = torch.cat([uv_coords, torch.ones_like(uv_coords)], dim=-1)[None].contiguous() * 2.0 - 1.0

    ctx = dr.RasterizeCudaContext()
    rast_out, rast_out_db = dr.rasterize(ctx, vt_hom, ft, texture_resolution)

    texel_areas, _ = dr.interpolate(areas_w_boundary, rast_out, ft)
    texel_areas = dr.antialias(texel_areas, rast_out, vt_hom, ft)

    face_ids = rast_out[..., 3].long() - 1  # (1, H, W)
    valid_mask = face_ids >= 0
    num_faces = ft.shape[0]
    texels_per_triangle = torch.zeros(num_faces, device=device, dtype=torch.float32)

    if valid_mask.any():
        valid_face_ids = face_ids[valid_mask]
        face_counts = torch.bincount(valid_face_ids, minlength=num_faces).float()
        texels_per_triangle = face_counts

    texel_counts = torch.ones_like(face_ids, dtype=torch.float32, device=device)
    texel_counts[valid_mask] = texels_per_triangle[face_ids[valid_mask]]
    density_scale = 1.0 / torch.sqrt(texel_counts + 1e-8)
    density_scale = density_scale.unsqueeze(0)  # (1, 1, H, W)

    area_scale = torch.sqrt(texel_areas).permute(0, 3, 1, 2)  # B C H W
    scale_multipliers = area_scale * density_scale  # (1, 1, H, W)

    logger.info(
        f"Computed texel scales: mean={scale_multipliers.mean():.4f}, "
        f"std={scale_multipliers.std():.4f}, "
        f"range=[{scale_multipliers.min():.4f}, {scale_multipliers.max():.4f}]"
    )

    # Compute statistics for logging
    num_valid_texels = valid_mask.sum().item()
    total_texels = torch.prod(torch.tensor(texture_resolution)).item()
    triangle_stats = {
        "min_texels": texels_per_triangle[texels_per_triangle > 0].min().item() if (texels_per_triangle > 0).any() else 0,
        "max_texels": texels_per_triangle.max().item(),
        "mean_texels": texels_per_triangle[texels_per_triangle > 0].mean().item() if (texels_per_triangle > 0).any() else 0,
        "triangles_with_texels": (texels_per_triangle > 0).sum().item(),
        "total_triangles": num_faces,
    }
    logger.info(
        f"Texel distribution: {num_valid_texels}/{total_texels} texels cover geometry ({100 * num_valid_texels / total_texels:.1f}%)"
    )

    logger.info(
        f"Triangle coverage: {triangle_stats['triangles_with_texels']}/{triangle_stats['total_triangles']} "
        f"triangles have texels ({100 * triangle_stats['triangles_with_texels'] / triangle_stats['total_triangles']:.1f}%). "
        f"Texels per triangle: min={triangle_stats['min_texels']}, max={triangle_stats['max_texels']}, "
        f"mean={triangle_stats['mean_texels']:.1f}"
    )

    return scale_multipliers
