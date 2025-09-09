"""
Script to generate per-person projected textures for A-pose data from MVHPPAposeDataset.

This script:
1. Instantiates an (MVH/MVHPP/DNA)Dataset from a config file
2. Iterates over samples to get view data and motion data
3. Uses TexelSMPLX to generate mesh data from SMPLX parameters
4. Computes projected texture using collect_uv_texture
5. Saves textures as {char_name}_apose_texture.jpg
"""

import sys
from pathlib import Path

import nvdiffrast.torch as dr
import torch
import torchvision.transforms.functional as F
import typer
from jaxtyping import Float
from loguru import logger
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from giga.data import extimage_collate_fn
from giga.data.datasets.config import instantiate_dataset, load_dataset_config
from giga.engine.body_template.smplxtex import TexelSMPLX
from giga.engine.projector import collect_uv_texture
from giga.engine.utils import base as mng_base
from giga.engine.utils import geometry as mng_geometry


def setup_logger() -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


def prepare_mesh_from_smplx_output(
    smplx_output: dict[str, Tensor],
    smplx_model: TexelSMPLX,
    device: torch.device,
) -> dict[str, Tensor]:
    """
    Prepare mesh data for texture projection from SMPL-X output.

    Args:
        smplx_output: Output from TexelSMPLX forward pass
        smplx_model: The TexelSMPLX model instance
        device: Target device for computation

    Returns:
        Dictionary containing mesh data with vertices, faces, normals, and UV attributes
    """
    vertices = smplx_output["vertices"].squeeze(0)  # Remove batch dimension
    faces = smplx_model.faces

    normals = mng_geometry.compute_vertex_normals(vertices, faces)
    attrs = smplx_model.uv_vertices
    ids_w_boundary = smplx_model.template_with_boundary

    mesh = {
        "vertices": vertices[ids_w_boundary].to(device),
        "faces": smplx_model.uv_faces.to(device),
        "normals": normals[ids_w_boundary].to(device),
        "attrs": attrs.to(device),
    }

    return mesh


def tensor_to_pil_image(tensor: Float[Tensor, "c h w"]) -> Image.Image:
    """
    Convert a tensor to PIL Image.

    Args:
        tensor: Input tensor with shape (c, h, w) and values in [0, 1]

    Returns:
        PIL Image
    """
    if tensor.shape[0] > 3:
        tensor = tensor[:3]

    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()

    return F.to_pil_image(tensor)


def prepare_inputs(batch: dict, device: torch.device, dataset_type: str) -> tuple[dict, dict]:
    """
    Prepare inputs for texture generation from a batch, moving data to the correct device.

    Args:
        batch: A batch of data from the dataloader.
        device: The target device for tensors.

    Returns:
        A tuple containing the processed inputs and keyword arguments for SMPL-X.
    """
    batch = mng_base.to_device(batch, device=device)
    if dataset_type in ("mvh", "mvhpp"):
        views = batch["apose_views"][0]
        motion = batch["apose_motion"][0]
        inputs = {
            "name": batch["name"][0],
            "views": views,
        }
        lbs_kwargs = motion
    else:
        views = batch["input_views"][0]
        motion = batch["motion"][0]
        inputs = {
            "name": batch["name"][0],
            "views": views,
        }
        lbs_kwargs = motion

    return inputs, lbs_kwargs


def main(
    config_path: Path = typer.Argument(..., help="Path to the dataset config YAML file"),
    output_dir: Path = typer.Argument(..., help="Output directory for generated textures"),
    smplx_model_path: Path = typer.Option(..., "--smplx-path", help="Path to SMPL-X model file (optional, will use config default)"),
    texture_resolution: int = typer.Option(1024, help="Resolution for output textures"),
    strategy: str = typer.Option("softmax", help="Texture merging strategy: 'best' or 'softmax'"),
    top_k: int = typer.Option(8, help="Number of top views to use for texture merging"),
    num_workers: int = typer.Option(4, help="Number of workers for the dataloader"),
    gpu_id: int = typer.Option(0, "--gpu-id", help="GPU device ID to use for computation."),
):
    """Generate per-person projected textures for A-pose data."""

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    setup_logger()

    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")

    if strategy not in ["best", "softmax"]:
        raise typer.BadParameter(f"Invalid strategy: {strategy}. Must be 'best' or 'softmax'")

    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Loading dataset config from: {config_path}")

    # Override texture resolution if specified
    config = load_dataset_config(config_path)
    config.texture_resolution = (texture_resolution, texture_resolution)
    dataset = instantiate_dataset(config, split="train")

    dataset_cls_name = dataset.__class__.__name__
    if "MVHPP" in dataset_cls_name:
        dataset_type = "mvhpp"
    elif "MVH" in dataset_cls_name:
        dataset_type = "mvh"
    elif "DNA" in dataset_cls_name:
        dataset_type = "dna"
    else:
        dataset_type = "other"

    logger.info("Initializing dataset...")
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one character at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=extimage_collate_fn,
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples and {len(dataset.characters)} characters")

    logger.info("Initializing SMPL-X model...")
    smplx_model = TexelSMPLX(
        model_path=smplx_model_path,
        uvmap_path=smplx_model_path / "smplx_uv.npz",
        gender="all",  # Can be overridden per sample
        flat_hand_mean=True,  # irrelevant
        use_hand_pca=True,  # irrelevant
    ).to(device)

    rast_ctx = dr.RasterizeCudaContext(device=device)
    logger.info("Starting texture generation...")

    processed_chars = set()

    for batch in tqdm(dataloader, desc="Processing samples"):
        try:
            inputs, lbs_kwargs = prepare_inputs(batch, device, dataset_type)
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            continue
        char_name = inputs["name"]

        # Skip if we've already processed this character
        if char_name in processed_chars:
            continue

        processed_chars.add(char_name)

        with torch.inference_mode():
            smplx_output = smplx_model(**lbs_kwargs)
            mesh = prepare_mesh_from_smplx_output(smplx_output, smplx_model, device)
            texture = collect_uv_texture(
                mesh=mesh,
                cameras=inputs["views"],
                rast_ctx=rast_ctx,
                texture_resolution=(texture_resolution, texture_resolution),
                strategy=strategy,
                top_k=top_k,
            )

        texture_rgb = texture.squeeze(0)  # Remove batch dimension
        texture_pil = tensor_to_pil_image(texture_rgb)

        output_path = output_dir / f"{char_name}_apose_texture.jpg"
        texture_pil.save(output_path, quality=95)

    logger.info(f"Texture generation complete! Processed {len(processed_chars)} characters.")
    logger.info(f"Output saved to: {output_dir}")


app = typer.Typer(pretty_exceptions_enable=False)
app.command()(main)

if __name__ == "__main__":
    app()
