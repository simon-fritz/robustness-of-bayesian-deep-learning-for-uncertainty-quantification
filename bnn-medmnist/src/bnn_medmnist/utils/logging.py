"""Run-level logging: git commit hash, SLURM job id, hostname, timestamp."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def get_git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID")


def log_run_start(
    run_dir: str | Path | None = None,
    config: Any = None,
    logger: logging.Logger | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Print and persist run metadata (git commit, SLURM job, hostname, config)."""
    info: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
        "slurm_job_id": get_slurm_job_id(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
    }
    if extra:
        info.update(extra)

    lines = [f"{k}: {v}" for k, v in info.items()]
    if config is not None:
        lines.append("config:")
        lines.append(OmegaConf.to_yaml(config))
    text = "\n".join(lines)

    print(text)
    if logger is not None:
        logger.info(text)
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_info.txt").write_text(text)
