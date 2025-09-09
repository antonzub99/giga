import os
from typing import Optional


def is_slurm_environment() -> bool:
    """Check if we're running in a SLURM environment."""
    return "SLURM_JOB_ID" in os.environ


def get_current_slurm_job_id() -> Optional[str]:
    """Get the current SLURM job ID if available."""
    return os.environ.get("SLURM_JOB_ID")
