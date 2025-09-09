import argparse
import hashlib
import subprocess
import time as time_module
from pathlib import Path


def submit_slurm_job(
    partition: str,
    cpus: int,
    gpus: str,
    memory: str,
    time: str,
    conda_env: str,
    venv_path: str,
    log_dir: str,
    job_name: str,
    train_args: list[str],
    requeue: bool = True,
):
    """Submit SLURM job for training."""

    # Generate unique submission code based on timestamp and job parameters
    timestamp = int(time_module.time() * 1000)  # milliseconds for uniqueness
    job_signature = f"{job_name}_{partition}_{cpus}_{gpus}_{memory}_{' '.join(train_args)}"
    unique_hash = hashlib.md5(job_signature.encode()).hexdigest()[:8]
    submission_code = f"{timestamp}_{unique_hash}"  # Add requeue option if enabled
    requeue_line = "#SBATCH --requeue" if requeue else ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mem={memory}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err
#SBATCH --signal=USR1@90
{requeue_line}

# Unique submission code: {submission_code}
# Generated at: {time_module.strftime("%Y-%m-%d %H:%M:%S")}
# Job signature: {job_signature}

source ~/.bashrc
micromamba activate {conda_env}
source {venv_path}/bin/activate

cd {Path.cwd()}

# Export submission code for tracking
export SUBMISSION_CODE="{submission_code}"

echo "Submission Code: $SUBMISSION_CODE"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Requeue enabled: {requeue}"

srun python scripts/train.py {" ".join(train_args)}
"""  # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Write main submission script with unique code in filename
    script_path = f"{log_dir}/submit_{job_name}_{submission_code}.sh"
    with open(script_path, "w") as f:
        f.write(sbatch_script)

    print(f"Created submission script: {script_path}")
    print(f"Submission code: {submission_code}")

    # Submit the job
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error submitting job: {result.stderr.strip()}")
        return None, None
    else:
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id}: {job_name}")
        if requeue:
            print("✅ SLURM requeuing enabled - job will automatically restart on preemption")
        else:
            print("❌ SLURM requeuing disabled - manual restart required on preemption")

    return job_id, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, required=True)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--gpus", type=str, default="1")
    parser.add_argument("--memory", default="128G")
    parser.add_argument("--time", default="06:00:00")
    parser.add_argument("--conda-env", default="giga")
    parser.add_argument("--venv-path", default="./.venv")
    parser.add_argument("--log-dir", default="./slurm_logs")
    parser.add_argument("--job-name", default="giga_train")
    parser.add_argument("--no-requeue", action="store_true", help="Disable automatic SLURM requeuing")

    args, train_args = parser.parse_known_args()

    job_id, _ = submit_slurm_job(
        partition=args.partition,
        cpus=args.cpus,
        gpus=args.gpus,
        memory=args.memory,
        time=args.time,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
        log_dir=args.log_dir,
        job_name=args.job_name,
        train_args=train_args,
        requeue=not args.no_requeue,
    )
