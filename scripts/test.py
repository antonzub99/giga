import ast
import sys
from pathlib import Path

import lightning as pl
import torch
import typer
from loguru import logger
from omegaconf import ListConfig, OmegaConf
from torch.utils.data import DataLoader

from giga.data import extimage_collate_fn
from giga.data.datasets import instantiate_dataset, load_dataset_config
from giga.engine.body_template import TexelSMPLX
from giga.engine.evaluator import Evaluator, create_trajectory
from giga.engine.trainer.config import InstantiableConfig, MainConfig

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


def setup_logger(log_file: Path) -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
    logger.add(log_file, rotation="100 MB")


def setup_globals():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)

    if torch.backends.cuda.is_flash_attention_available():
        logger.info("Flash attention is available; you might want to use it.")


def convert_wandb_config(config_path: Path) -> OmegaConf:
    """
    Load a wandb config file, clean it up, and return it as an OmegaConf object.
    It removes the _wandb key and unnests the 'value' field for other keys.
    It also parses string representations of dictionaries into actual dictionaries.
    """
    raw_config = OmegaConf.load(config_path)
    cleaned_config = {}
    for key, data in raw_config.items():
        if key == "_wandb":
            continue
        value = data["value"]
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        cleaned_config[key] = value

    return OmegaConf.create(cleaned_config)


def setup_pipeline(
    model_config: Path,
    data_config: Path,
    output_path: Path,
    smplx_path: Path,
    actor_id: str | None,
    background_color: str,
    compile: bool = False,
):
    """Common setup logic for both evaluation and render commands."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training without CUDA is not supported.")
    device = torch.device("cuda:0")
    setup_globals()

    logger.info(f"Loading run configuration from {model_config}")
    # Check if it's a wandb config format (has nested structure with 'value' keys)
    raw_config = OmegaConf.load(model_config)
    if isinstance(raw_config, dict) and any(isinstance(v, dict) and "value" in v for v in raw_config.values()):
        # It's a wandb config, convert it
        run_config = convert_wandb_config(model_config)
    else:
        # It's a regular config file
        run_config = raw_config
    run_config: MainConfig = MainConfig(**run_config)

    # Find checkpoint file in model_path
    model_path = model_config.parent
    checkpoint_files = list(model_path.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files (*.ckpt) found in {model_path}")
    elif len(checkpoint_files) > 1:
        logger.warning(f"Multiple checkpoint files found in {model_path}. Using the first one: {checkpoint_files[0].name}")
    checkpoint_path = checkpoint_files[0]
    logger.info(f"Found checkpoint: {checkpoint_path}")

    logger.info(f"Loading test dataset configuration from {data_config}")
    test_dataset_config = load_dataset_config(data_config)
    if actor_id is not None:
        test_dataset_config.character_selection = actor_id
    test_dataset = instantiate_dataset(test_dataset_config, split="test")

    run_config.trainer.bg_color = background_color
    run_id = model_path.name
    logger.info(f"Run ID: {run_id}")

    run_config.trainer.compile = compile

    actor_id = test_dataset_config.character_selection
    if isinstance(actor_id, (list, ListConfig)):
        actor_id = actor_id[0]
    output_path = output_path / f"{run_id}" / f"{actor_id}"
    logger.info(f"Results will be saved to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=extimage_collate_fn,
    )

    mesh_lbs_layer = TexelSMPLX(
        model_path=smplx_path,
        uvmap_path=smplx_path / "smplx_uv.npz",
        gender="all",
        flat_hand_mean=False,
    )

    model = InstantiableConfig.instantiate(run_config.model)
    conditioner = InstantiableConfig.instantiate(run_config.conditioner)
    loss_criterion = torch.nn.Module()

    pl_module_kwargs = {
        "model": model,
        "conditioner": conditioner,
        "optimizer": None,
        "loss_criterion": loss_criterion,
        "scheduler": None,
        "trainer_config": run_config.trainer,
        "logging_config": run_config.logging,
        "mesh_lbs_layer": mesh_lbs_layer,
    }

    if run_config.identity_conditioner is not None:
        identity_conditioner = InstantiableConfig.instantiate(run_config.identity_conditioner)
        logger.info(f"Using identity conditioner: {identity_conditioner.__class__.__name__}")
        pl_module_kwargs["identity_conditioner"] = identity_conditioner

    module_path, class_name = run_config.pl_model.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    pl_model = getattr(module, class_name)

    logger.info(f"Using model: {pl_model.__name__} from {module_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    lightning_module = pl_model.load_from_checkpoint(checkpoint_path, strict=False, **pl_module_kwargs)
    lightning_module.to(device)
    lightning_module.on_fit_start()
    lightning_module.eval()

    return lightning_module, test_dataloader, output_path, device


@app.command()
def eval(
    model_config: Path = typer.Option(..., "--model-config", help="Path to the model YAML config."),
    data_config: Path = typer.Option(..., "--dataset-config", help="Path to the test dataset configuration file."),
    output_path: Path = typer.Option(..., "--output-path", help="Path to save the results to."),
    smplx_path: Path = typer.Option(..., help="Path to SMPL-X files (.npz per gender)."),
    actor_id: str = typer.Option(None, help="Actor ID to use for the test dataset."),
    background_color: str = typer.Option("black", "--background-color", help="Background color for the renders."),
    render_save_format: str = typer.Option("png", "--render-save-format", help="Format to save rendered images."),
    render_grid_format: str = typer.Option("vertical", "--render-grid-format", help="Grid format for saving images."),
    video_save_framerate: int = typer.Option(5, "--video-save-framerate", help="Framerate for saving videos."),
    compile: bool = typer.Option(False, "--compile", help="Whether to compile the model or not."),
):
    """Run full evaluation with metrics computation."""
    lightning_module, test_dataloader, output_path, device = setup_pipeline(
        model_config,
        data_config,
        output_path,
        smplx_path,
        actor_id,
        background_color,
        compile,
    )

    evaluator = Evaluator(
        logging_dir=output_path,
        device=device,
        save_rgb=True,
        save_prims=False,
        render_save_format=render_save_format,
        render_grid_format=render_grid_format,
        video_save_framerate=video_save_framerate,
    )

    logger.info("Starting evaluation...")
    evaluator.evaluate(lightning_module, test_dataloader)
    logger.success("Evaluation completed successfully.")


@app.command()
def render(
    model_config: Path = typer.Option(..., "--model-config", help="Path to the model YAML config."),
    data_config: Path = typer.Option(..., "--dataset-config", help="Path to the test dataset configuration file."),
    output_path: Path = typer.Option(..., "--output-path", help="Path to save the results to."),
    smplx_path: Path = typer.Option(..., help="Path to SMPL-X files (.npz per gender)."),
    actor_id: str = typer.Option(None, help="Actor ID to use for the test dataset."),
    background_color: str = typer.Option("black", "--background-color", help="Background color for the renders."),
    render_save_format: str = typer.Option("png", "--render-save-format", help="Format to save rendered images."),
    render_grid_format: str = typer.Option("vertical", "--render-grid-format", help="Grid format for saving images."),
    video_save_framerate: int = typer.Option(5, "--video-save-framerate", help="Framerate for saving videos."),
    compile: bool = typer.Option(False, "--compile", help="Whether to compile the model or not."),
):
    """Run render-only mode without metrics computation."""
    lightning_module, test_dataloader, output_path, device = setup_pipeline(
        model_config,
        data_config,
        output_path,
        smplx_path,
        actor_id,
        background_color,
        compile,
    )

    evaluator = Evaluator(
        logging_dir=output_path,
        device=device,
        save_rgb=True,
        save_prims=False,
        render_save_format=render_save_format,
        render_grid_format=render_grid_format,
        video_save_framerate=video_save_framerate,
    )

    logger.info("Starting render-only mode...")
    evaluator.render_only(lightning_module, test_dataloader)
    logger.success("Rendering completed successfully.")


@app.command()
def freeview(
    model_config: Path = typer.Option(..., "--model-config", help="Path to the model YAML config."),
    data_config: Path = typer.Option(..., "--dataset-config", help="Path to the test dataset configuration file."),
    output_path: Path = typer.Option(..., "--output-path", help="Path to save the results to."),
    smplx_path: Path = typer.Option(..., help="Path to SMPL-X files (.npz per gender)."),
    actor_id: str = typer.Option(None, help="Actor ID to use for the test dataset."),
    compile: bool = typer.Option(False, "--compile", help="Whether to compile the model or not."),
    # Rendering options
    background_color: str = typer.Option("black", "--background-color", help="Background color for the renders."),
    render_save_format: str = typer.Option("png", "--render-save-format", help="Format to save rendered images."),
    render_grid_format: str = typer.Option("vertical", "--render-grid-format", help="Grid format for saving images."),
    video_save_framerate: int = typer.Option(5, "--video-save-framerate", help="Framerate for saving videos."),
    # General trajectory parameters
    trajectory_mode: str = typer.Option("orbit", "--trajectory-mode", help="Camera trajectory mode: 'orbit' or 'interpolate'."),
    resolution: str = typer.Option(None, "--resolution", help="Resolution for the rendered images (width,height)."),
    num_steps: int = typer.Option(100, "--num-steps", help="Number of steps for the camera trajectory."),
    num_steps_per_transition: int = typer.Option(None, "--num-steps-per-transition", help="Steps per transition in the camera trajectory."),
    loop: bool = typer.Option(
        False, "--loop", help="Whether to loop the camera trajectory or not: if True, last camera connects to the first."
    ),
    # Orbit trajectory parameters
    orbit_radius: float = typer.Option(None, "--orbit-radius", help="Orbit radius (if None, computed from initial camera)."),
    up: str = typer.Option("y", "--up", help="Up direction for orbit trajectory (default is 'y')."),
    fixed_y: float = typer.Option(None, "--fixed-y", help="Fixed Y coordinate for orbit (if None, uses initial camera Y)."),
    orbit_angle_per_step: float = typer.Option(0.05, "--orbit-angle-per-step", help="Angle increment per step (radians)."),
):
    if resolution is not None:
        resolution = tuple(map(int, resolution.split(",")))

    lightning_module, test_dataloader, output_path, device = setup_pipeline(
        model_config,
        data_config,
        output_path,
        smplx_path,
        actor_id,
        background_color,
        compile,
    )

    trajectory_data = create_trajectory(
        lightning_module,
        test_dataloader,
        trajectory_mode,
        up_vector=up,
        num_steps=num_steps,
        num_steps_per_transition=num_steps_per_transition,
        resolution=resolution,
        loop=loop,
        orbit_radius=orbit_radius,
        fixed_y=fixed_y,
        angle_step=orbit_angle_per_step,
    )

    evaluator = Evaluator(
        logging_dir=output_path,
        device=device,
        save_rgb=True,
        save_prims=False,
        render_save_format=render_save_format,
        render_grid_format=render_grid_format,
        video_save_framerate=video_save_framerate,
    )

    logger.info("Starting freeview rendering mode...")
    evaluator.freeview(lightning_module, test_dataloader, *trajectory_data)
    logger.success("Rendering completed successfully.")


if __name__ == "__main__":
    app()
