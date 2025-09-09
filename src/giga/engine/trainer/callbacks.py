import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import lightning as pl
import torch
from lightning.pytorch.callbacks import Callback
from loguru import logger

if sys.platform == "win32":
    DEFAULT_EMERGENCY_SIGNALS = (signal.SIGTERM, signal.SIGINT)
else:
    DEFAULT_EMERGENCY_SIGNALS = (signal.SIGUSR1, signal.SIGUSR2)


class PeriodicCheckpoint(Callback):
    """Save checkpoints every N training steps."""

    def __init__(
        self,
        save_dir: str | Path,
        every_n_train_steps: int,
        filename: str = "checkpoint_{step:06d}",
    ):
        """Initialize callback.

        Args:
            save_dir: Directory to save checkpoints
            every_n_train_steps: Save frequency in training steps
            filename: Checkpoint filename template
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.every_n_train_steps = every_n_train_steps
        self.filename = filename
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Check if checkpoint should be saved after batch."""
        global_step = trainer.global_step

        if global_step > 0 and global_step % self.every_n_train_steps == 0:
            filepath = self.save_dir / f"{self.filename}.ckpt"
            trainer.save_checkpoint(filepath.as_posix().format(step=global_step))


class EmergencyCheckpointCallback(Callback):
    """
    Custom callback for emergency checkpointing with SLURM requeuing.
    Based on PyTorch Lightning's signal handling approach.
    Note: This callback is designed to work with SLURM job scheduling and is called upon job requeuing.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        emergency_signals: tuple[int, ...] = DEFAULT_EMERGENCY_SIGNALS,
    ):
        """
        Args:
            checkpoint_dir: Directory to save emergency checkpoints
            emergency_signals: Signals that trigger emergency checkpointing
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.emergency_signals = emergency_signals
        self.trainer: Optional[pl.Trainer] = None
        self.emergency_triggered = False
        self.emergency_checkpoint_path = self.checkpoint_dir / "emergency.ckpt"

        for sig in self.emergency_signals:
            signal.signal(sig, self._handle_signal)

        logger.info("Emergency checkpoint callback initialized for SLURM requeue")
        logger.info(f"Listening for signals: {self.emergency_signals}")
        logger.info(f"Emergency checkpoint will be saved to: {self.emergency_checkpoint_path}")

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Setup callback with trainer reference."""
        self.trainer = trainer

        if self.emergency_checkpoint_path.exists():
            logger.warning(f"Found existing emergency checkpoint: {self.emergency_checkpoint_path}")
            logger.info("This suggests the job was requeued. The checkpoint will be used for resuming.")

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle emergency signals by triggering checkpoint save and requeue."""
        if self.emergency_triggered:
            logger.warning(f"Emergency already triggered, ignoring signal {signum}")
            return

        logger.warning(f"Received signal {signum} - handling SLURM requeue...")
        self.emergency_triggered = True

        try:
            self._save_emergency_checkpoint()
            self._save_wandb_id_for_resume()

            if hasattr(self.trainer, "loggers"):
                for logger_instance in self.trainer.loggers:
                    if hasattr(logger_instance, "finalize"):
                        logger_instance.finalize("requeued")

            self._requeue_slurm_job()

        except Exception as e:
            logger.error(f"Failed to handle emergency signal: {e}")
            raise
        finally:
            logger.info("Emergency checkpoint and requeue complete")

    def _save_wandb_id_for_resume(self) -> None:
        """Saves the W&B run ID to a file for seamless resuming."""
        if self.trainer is None:
            logger.error("Trainer not available, cannot save W&B run ID.")
            return

        wandb_logger = None
        if hasattr(self.trainer, "loggers"):
            for logger_instance in self.trainer.loggers:
                if isinstance(logger_instance, pl.pytorch.loggers.wandb.WandbLogger):
                    wandb_logger = logger_instance
                    break

        if not wandb_logger or not hasattr(wandb_logger, "experiment"):
            logger.error("W&B logger not found or not initialized. Cannot save run ID.")
            return

        wandb_id = wandb_logger.experiment.id
        if not wandb_id:
            logger.error("W&B run ID not available. Cannot save for resume.")
            return

        wandb_id_file = self.checkpoint_dir / "emergency.wandb_id"
        try:
            wandb_id_file.write_text(wandb_id)
            logger.info(f"Saved W&B run ID '{wandb_id}' to {wandb_id_file}")
        except IOError as e:
            logger.error(f"Failed to write W&B run ID to {wandb_id_file}: {e}")

    def _requeue_slurm_job(self) -> None:
        """Requeue the current SLURM job using scontrol."""
        import os
        import re

        # Get job ID (handle both regular and array jobs like PyTorch Lightning)
        array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        if array_job_id is not None:
            array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
            job_id = f"{array_job_id}_{array_task_id}"
        else:
            job_id = os.environ.get("SLURM_JOB_ID")

        if not job_id:
            logger.error("No SLURM_JOB_ID found, cannot requeue")
            return

        if not re.match(r"[0-9_-]+", job_id):
            logger.error(f"Invalid job ID format: {job_id}")
            return

        cmd = ["scontrol", "requeue", job_id]
        logger.info(f"Requeuing SLURM job {job_id}...")

        try:
            result = subprocess.call(cmd)
        except FileNotFoundError:
            logger.warning("scontrol not found, retrying with shell context")
            result = subprocess.call(" ".join(cmd), shell=True)

        if result == 0:
            logger.success(f"Successfully requeued SLURM job: {job_id}")
        else:
            logger.error(f"Failed to requeue SLURM job {job_id} with error code {result}")
            raise RuntimeError(f"SLURM requeue failed with code {result}")

    def _save_emergency_checkpoint(self) -> None:
        """Save emergency checkpoint using PyTorch Lightning's native checkpointing."""
        if self.trainer is None:
            logger.error("Trainer not available for emergency checkpoint")
            return

        logger.info(f"Saving emergency checkpoint to {self.emergency_checkpoint_path}")

        checkpoint_metadata = {
            "emergency_checkpoint": True,
            "original_global_step": self.trainer.global_step,
            "original_epoch": self.trainer.current_epoch,
            "slurm_job_id": self._get_slurm_job_id(),
            "save_timestamp": __import__("time").time(),
        }

        try:
            self.emergency_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Saving emergency checkpoint using PyTorch Lightning's native method...")

            # Create a temporary file in the target directory for atomic save
            import shutil
            import tempfile

            temp_dir = self.emergency_checkpoint_path.parent

            with tempfile.NamedTemporaryFile(dir=temp_dir, prefix="emergency_checkpoint_", suffix=".tmp", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            logger.info(f"Saving to temporary file: {temp_path}")

            # Use PyTorch Lightning's checkpoint connector to create complete checkpoint
            # This is the same method Lightning uses internally for save_checkpoint
            checkpoint = self.trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
            checkpoint.update(checkpoint_metadata)
            # Save the complete checkpoint directly
            torch.save(checkpoint, temp_path)

            if not temp_path.exists():
                raise RuntimeError(f"Temporary checkpoint was not created at {temp_path}")

            shutil.move(str(temp_path), str(self.emergency_checkpoint_path))

            if not self.emergency_checkpoint_path.exists():
                raise RuntimeError(f"Emergency checkpoint was not created at {self.emergency_checkpoint_path}")
            checkpoint_size = self.emergency_checkpoint_path.stat().st_size

            logger.success(f"Emergency checkpoint saved successfully at step {self.trainer.global_step}")
            logger.info(f"Checkpoint size: {checkpoint_size / (1024 * 1024):.1f} MB")
            logger.info("Emergency checkpoint includes complete PyTorch Lightning training state")

        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")

            try:
                if "temp_path" in locals() and temp_path.exists():
                    temp_path.unlink()
                    logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")

            raise e

    def _get_slurm_job_id(self) -> Optional[str]:
        """Get current SLURM job ID."""
        import os

        return os.environ.get("SLURM_JOB_ID")

    @staticmethod
    def get_emergency_checkpoint_path(checkpoint_dir: Path) -> Path:
        """Get the emergency checkpoint path for external use."""
        return Path(checkpoint_dir) / "emergency.ckpt"

    @staticmethod
    def has_emergency_checkpoint(checkpoint_dir: Path) -> bool:
        """Check if emergency checkpoint exists."""
        emergency_path = EmergencyCheckpointCallback.get_emergency_checkpoint_path(checkpoint_dir)
        return emergency_path.exists()

    @staticmethod
    def cleanup_emergency_checkpoint(checkpoint_dir: Path) -> None:
        """Clean up emergency checkpoint after successful resume."""
        emergency_path = EmergencyCheckpointCallback.get_emergency_checkpoint_path(checkpoint_dir)
        wandb_id_file = Path(checkpoint_dir) / "emergency.wandb_id"

        try:
            if emergency_path.exists():
                emergency_path.unlink()
                logger.info(f"Cleaned up emergency checkpoint: {emergency_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup emergency checkpoint {emergency_path}: {e}")

        try:
            if wandb_id_file.exists():
                wandb_id_file.unlink()
                logger.info(f"Cleaned up emergency wandb_id file: {wandb_id_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup emergency wandb_id file: {e}")
