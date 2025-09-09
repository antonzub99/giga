import os
import sys
from pathlib import Path
from typing import Any, Optional

import lightning as pl
import torch
import typer
import wandb
from lightning.pytorch.plugins import environments
from loguru import logger
from torch.utils.data import ConcatDataset, DataLoader

from giga.data import extimage_collate_fn
from giga.data.datasets import instantiate_dataset, load_dataset_config
from giga.engine.body_template import TexelSMPLX, compute_texel_scales
from giga.engine.trainer.callbacks import EmergencyCheckpointCallback, PeriodicCheckpoint
from giga.engine.trainer.config import InstantiableConfig, MainConfig
from giga.engine.trainer.slurm_utils import get_current_slurm_job_id, is_slurm_environment

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


def setup_dataloaders(
    dataset_config_paths: list[Path],
    data_config: Any,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from multiple dataset configs."""
    train_datasets = []
    val_datasets = []

    for dataset_config_path in dataset_config_paths:
        logger.info(f"Loading dataset config: {dataset_config_path}")
        dataset_config = load_dataset_config(dataset_config_path)

        train_dataset = instantiate_dataset(dataset_config, split="train")
        val_dataset = instantiate_dataset(dataset_config, split="val")

        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset)

        logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    combined_train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    combined_val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    logger.info(f"Combined train dataset size: {len(combined_train_dataset)}")
    logger.info(f"Combined val dataset size: {len(combined_val_dataset)}")

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        shuffle=True,
        persistent_workers=data_config.persistent_workers,
        drop_last=data_config.drop_last,
        collate_fn=extimage_collate_fn,
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=max(data_config.batch_size // 2, 1),
        num_workers=max(data_config.num_workers // 2, 1),
        pin_memory=data_config.pin_memory,
        shuffle=False,
        persistent_workers=False,
        collate_fn=extimage_collate_fn,
    )

    return train_loader, val_loader


def setup_wandb_logger(config: MainConfig, log_dir: Path, resume_id: str | None = None, project_name: str = "monogiga"):
    """Setup Weights & Biases logger with proper resume handling."""
    run_id_file = log_dir / "wandb_run_id.txt"

    if resume_id:
        logger.info(f"Resuming W&B run with ID: {resume_id}")
        run_id = resume_id
    elif run_id_file.exists():
        logger.info(f"Found existing W&B run ID file: {run_id_file}")
        run_id = run_id_file.read_text().strip()
    else:
        logger.info("No existing W&B run ID found, generating a new one.")
        run_id = wandb.util.generate_id()
        run_id_file.write_text(run_id)

    return pl.pytorch.loggers.wandb.WandbLogger(
        project=project_name,
        name=config.experiment_name,
        id=run_id,
        save_dir=str(log_dir),
        resume="allow",
        config=config,
    )


def parse_dataset_configs(dataset_config_input: str) -> list[Path]:
    """
    Parse dataset config input - can be single path or comma-separated paths.

    Args:
        dataset_config_input: String containing one or more paths, separated by commas

    Returns:
        List of Path objects for dataset configs
    """
    paths_str = [path.strip() for path in dataset_config_input.split(",")]

    dataset_paths = []
    for path_str in paths_str:
        if not path_str:
            continue

        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Dataset config path is not a file: {path}")

        dataset_paths.append(path)

    if not dataset_paths:
        raise ValueError("No valid dataset config paths provided")

    return dataset_paths


def parse_cli_overrides(args: list[str]) -> list[str]:
    """Parse and validate CLI overrides from Typer context."""
    validated_args = []
    for arg in args:
        if "=" not in arg:
            raise typer.BadParameter(f"Invalid override: '{arg}'. Overrides must be in 'key=value' format.")
        validated_args.append(arg)
    return validated_args


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def main(
    ctx: typer.Context,
    model_config_path: Path = typer.Option(..., "--model-config", help="Path to the main training YAML config file."),
    dataset_config_path: str = typer.Option(
        ...,
        "--dataset-config",
        help="Path(s) to dataset YAML config file(s). Multiple paths can be separated by commas.",
    ),
    smplx_path: Path = typer.Option(..., help="Path to SMPL-X files (.npz per gender)."),
    attention_backend: str = typer.Option(
        "auto",
        "--attention-backend",
        help="Attention backend: auto, xformers, flash, efficient, cudnn, math. auto will select the best available (best are ordered left to right).",
    ),
    autoscale: bool = typer.Option(
        False,
        "--autoscale",
        help="Whether to automatically scale Gaussian scales based on mesh statistics. If False, uses the config settings directly.",
    ),
    resume_id: Optional[str] = typer.Option(
        None,
        "--resume-id",
        help="W&B run ID to resume from. If provided, will resume the specified run.",
    ),
    project_name: Optional[str] = typer.Option(
        "giga",
        "--project-name",
        help="W&B project name to use for logging. Defaults to 'giga'.",
    ),
    emergency_resubmit: bool = typer.Option(
        False,
        "--emergency-resubmit",
        help="Enable emergency checkpointing and automatic resubmission on SLURM signals.",
    ),
):
    if not torch.cuda.is_available():
        raise RuntimeError("Training without CUDA is not supported.")

    setup_globals()

    # override experiment_name, logging.output_dir and everything else from CLI
    cli_overrides = parse_cli_overrides(ctx.args)
    config: MainConfig = MainConfig.load(model_config_path, cli_overrides=cli_overrides)
    dataset_config_paths = parse_dataset_configs(dataset_config_path)

    # Get SLURM job ID if running on SLURM
    slurm_job_id = get_current_slurm_job_id()
    final_resume_id = resume_id or os.environ.get("WANDB_RUN_ID")

    log_dir = Path(config.logging.output_dir) / config.experiment_name
    if slurm_job_id:
        log_dir = log_dir / f"slurm_{slurm_job_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / "train.log")

    # Check for emergency W&B resume file
    ckpt_dir = log_dir / "checkpoints"
    emergency_wandb_file = ckpt_dir / "emergency.wandb_id"
    if emergency_wandb_file.exists():
        emergency_wandb_id = emergency_wandb_file.read_text().strip()
        if emergency_wandb_id and final_resume_id is None:
            logger.info(f"Found emergency W&B resume file with run ID: {emergency_wandb_id}")
            final_resume_id = emergency_wandb_id

    logger.info(f"Initializing training for experiment: {config.experiment_name}")
    logger.info(f"Output directory: {log_dir.resolve()}")
    logger.info(f"Main config path: {model_config_path.resolve()}")
    logger.info(f"Dataset config paths: {[p.resolve() for p in dataset_config_paths]}")

    if final_resume_id:
        logger.info(f"Resume ID provided: {final_resume_id}")

    config.save(log_dir / "config.yaml")

    train_loader, val_loader = setup_dataloaders(
        dataset_config_paths,
        config.data,
    )

    mesh_lbs_layer = TexelSMPLX(
        model_path=smplx_path,
        uvmap_path=smplx_path / "smplx_uv.npz",
        gender="all",
        flat_hand_mean=False,
    )

    if autoscale:
        texel_scales = compute_texel_scales(
            mesh_lbs_layer.to("cuda:0"),
            texture_resolution=config.trainer.texture_resolution,
            gender="neutral",
        ).cpu()  # (1, 1, H, W)
    else:
        texel_scales = None

    # Initialize model components
    model = InstantiableConfig.instantiate(config.model)
    conditioner = InstantiableConfig.instantiate(config.conditioner)
    optimizer = InstantiableConfig.instantiate(config.optimizer)
    scheduler = InstantiableConfig.instantiate(config.scheduler) if config.scheduler else None
    loss = InstantiableConfig.instantiate(config.loss)

    module_path, class_name = config.pl_model.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    pl_model = getattr(module, class_name)

    logger.info(f"Using model: {pl_model.__name__} from {module_path}")

    lightning_module = pl_model(
        model=model,
        conditioner=conditioner,
        optimizer=optimizer,
        loss_criterion=loss,
        mesh_lbs_layer=mesh_lbs_layer,
        scheduler=scheduler,
        trainer_config=config.trainer,
        logging_config=config.logging,
        texel_scales=texel_scales,
    )

    # Setup callbacks with emergency checkpoint handling
    ckpt_dir = log_dir / "checkpoints"
    callbacks = [
        PeriodicCheckpoint(
            save_dir=ckpt_dir,
            every_n_train_steps=config.logging.save_freq,
        ),
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            save_on_train_epoch_end=True,
            every_n_train_steps=config.logging.save_freq // 10,
            save_last=True,
        ),
    ]

    plugins = []
    # Add emergency checkpoint callback if on SLURM
    if slurm_job_id and emergency_resubmit:
        emergency_callback = EmergencyCheckpointCallback(
            checkpoint_dir=ckpt_dir,
        )
        callbacks.append(emergency_callback)
        logger.info("Added emergency checkpoint callback for SLURM requeue")
    elif is_slurm_environment():
        logger.info("Running in SLURM environment but emergency resubmit is disabled: using lightning version")
        plugins.append(environments.SLURMEnvironment(auto_requeue=True))

    wandb_logger = setup_wandb_logger(config, log_dir, final_resume_id, project_name)
    run_id_file = log_dir / "wandb_run_id.txt"
    if not run_id_file.exists():
        run_id_file.write_text(wandb_logger.experiment.id)
    # Setup attention backend before model initialization
    from giga.nn.attention import AttentionBackend, log_backend_status, setup_attention_backend

    log_backend_status()
    try:
        backend_enum = AttentionBackend(attention_backend.lower())
        setup_attention_backend(backend_enum)
    except ValueError:
        logger.error(f"Invalid attention backend: {attention_backend}. Available: {[b.value for b in AttentionBackend]}")
        setup_attention_backend(AttentionBackend.AUTO)

    DEBUG = os.environ.get("GIGA_DEBUG", "0") == "1"
    if DEBUG:
        detect_anomaly = True
    else:
        detect_anomaly = False

    accelerator = "gpu"
    if config.trainer.mixed_precision:
        if config.trainer.precision == "bf16":
            precision = "bf16-mixed"
        elif config.trainer.precision == "fp16" or config.trainer.precision == "16":
            logger.info("Using fp16 mixed precision training")
            precision = "16-mixed"
        else:
            precision = "bf16-mixed"
    else:
        precision = "32-true"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=config.trainer.num_devices,
        strategy="ddp" if config.trainer.ddp and config.trainer.num_devices > 1 else "auto",
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        precision=precision,
        gradient_clip_algorithm=config.trainer.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=config.trainer.get("gradient_clip_val", 10.0),
        accumulate_grad_batches=config.trainer.gradient_accumulation_steps,
        callbacks=callbacks,
        logger=wandb_logger,
        plugins=plugins,
        log_every_n_steps=config.logging.log_freq,
        val_check_interval=config.logging.eval_freq,
        check_val_every_n_epoch=None,
        deterministic="warn",  # will partially reset globals set above in setup_globals()
        num_sanity_val_steps=config.trainer.get("num_sanity_val_steps", 0),
        detect_anomaly=detect_anomaly,
    )

    # Enhanced checkpoint detection logic
    ckpt_path = None
    last_ckpt = ckpt_dir / "last.ckpt"
    if not last_ckpt.exists():
        all_ckpts = sorted(list(ckpt_dir.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
        if all_ckpts:
            last_ckpt = all_ckpts[0]
            logger.info(f"Found last checkpoint: {last_ckpt}")
        else:
            logger.warning("No last checkpoint found, starting fresh training.")
            last_ckpt = ckpt_dir / "last.ckpt"  # this is non-existent

    emergency_ckpt = EmergencyCheckpointCallback.get_emergency_checkpoint_path(ckpt_dir)
    fallback_emergency_ckpt = emergency_ckpt.with_suffix(".fallback.ckpt")
    minimal_emergency_ckpt = emergency_ckpt.with_suffix(".minimal.ckpt")

    # Priority: emergency checkpoint > fallback emergency > minimal emergency > specified resume > last checkpoint
    if emergency_ckpt.exists():
        ckpt_path = str(emergency_ckpt)
        logger.warning(f"Resuming from emergency checkpoint: {ckpt_path}")

        # Schedule cleanup of emergency checkpoint after successful start
        def cleanup_emergency():
            EmergencyCheckpointCallback.cleanup_emergency_checkpoint(ckpt_dir)

        # Register cleanup to happen after first successful training step
        import atexit

        atexit.register(cleanup_emergency)

    elif fallback_emergency_ckpt.exists():
        ckpt_path = str(fallback_emergency_ckpt)
        logger.warning(f"Resuming from fallback emergency checkpoint: {ckpt_path}")

        # Schedule cleanup
        def cleanup_emergency():
            EmergencyCheckpointCallback.cleanup_emergency_checkpoint(ckpt_dir)

        import atexit

        atexit.register(cleanup_emergency)

    elif minimal_emergency_ckpt.exists():
        ckpt_path = str(minimal_emergency_ckpt)
        logger.warning(f"Resuming from minimal emergency checkpoint: {ckpt_path}")
        logger.warning("Note: This checkpoint may be missing some training state (optimizers, schedulers)")

        # Schedule cleanup
        def cleanup_emergency():
            EmergencyCheckpointCallback.cleanup_emergency_checkpoint(ckpt_dir)

        import atexit

        atexit.register(cleanup_emergency)

    elif final_resume_id and last_ckpt.exists():
        ckpt_path = str(last_ckpt)
        logger.info(f"Resuming from checkpoint with W&B run ID {final_resume_id}: {ckpt_path}")
    elif last_ckpt.exists():
        ckpt_path = str(last_ckpt)
        logger.info(f"Resuming from checkpoint: {ckpt_path}")
    elif ckpt_dir.exists():
        checkpoints = sorted(list(ckpt_dir.glob("*.ckpt")), key=lambda p: p.stat().st_mtime, reverse=True)
        # Filter out emergency checkpoints from regular checkpoint list
        regular_checkpoints = [c for c in checkpoints if c.name != "emergency.ckpt"]
        if regular_checkpoints:
            ckpt_path = str(regular_checkpoints[0])
            logger.info(f"Resuming from most recent checkpoint: {ckpt_path}")
    elif final_resume_id:
        logger.warning(f"Resume ID {final_resume_id} provided but no checkpoint found at {last_ckpt}")
        logger.info("Starting fresh training with the specified W&B run ID")

    logger.info("Starting training...")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path,
    )

    logger.success("Training completed!")


if __name__ == "__main__":
    app()
