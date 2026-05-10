"""Run-level logging: git commit hash, SLURM job id, hostname, timestamp.

Called once at the start of every entry-point script so that experiment logs
are traceable back to a specific commit and cluster job.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_slurm_job_id() -> str | None:
    """Return ``$SLURM_JOB_ID`` if running under SLURM, else None."""
    return os.environ.get("SLURM_JOB_ID")


def log_run_start(logger: logging.Logger | None = None, extra: dict[str, Any] | None = None) -> None:
    """Log git commit, SLURM job id, and any extra metadata at run start."""
    # TODO: implement — emit a single structured INFO record.
    raise NotImplementedError
