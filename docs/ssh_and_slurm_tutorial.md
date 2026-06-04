# Accessing the Cluster, SSH & SLURM — Quick Tutorial

This document summarizes common steps and tips for connecting to a university cluster (VPN + SSH), configuring SSH shortcuts, and running jobs with SLURM.

## 1) VPN (if required)

- We require VPN access before SSH-ing to internal hosts. On macOS you can use Tunnelblick or the institution-provided client.
- Import the provided <a href="https://vpn.ito.cit.tum.de/">VPN profile (`aim`)</a>. If your VPN requires adding static routes, append them to the VPN configuration or your local routing table. Example routes (replace with your network values):

```bash
# route for servers
sudo route add -net 131.159.110.0/24 -interface <vpn-interface>
# route for workstations
sudo route add -net 131.159.128.0/24 -interface <vpn-interface>
```

Tips:
- If using Tunnelblick, you can add custom `route` commands to the profile's `up` script.
- Verify you can ping the cluster gateway (for example `131.159.110.9`) before attempting SSH.

## 2) SSH access

- Basic SSH command:

```bash
ssh <your-CIT-username>@131.159.110.9
```

- To simplify repeated access, add an SSH host shortcut to `~/.ssh/config`:

```text
Host selene
    HostName 131.159.110.9
    User <your-CIT-username>
    IdentityFile ~/.ssh/id_rsa
    TCPKeepAlive yes

# Example usage:
# ssh selene
```


## 3) Storage locations & best practices

- Many clusters provide multiple mount points. Prefer a project or personal data area for large files. Example recommended path:

```
/vol/miltank/users/<username>/
```

- Store code and small config files in your home or project area, but keep large datasets and checkpoints on the designated storage to avoid quotas on home directories.
- Organize outputs under `outputs/` and checkpoints under `checkpoints/` in your project repository so code and results are easy to track.


## 4) Submitting jobs with SLURM (`sbatch`)

- Typical flow:
  1. Create a shell script that sets up the environment and runs your program.
  2. Submit it with `sbatch run_job.sh`.
  3. Monitor with `squeue -u <username>` or `sacct`.

- Minimal example `run_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --output=logs/example_job-%j.out
#SBATCH --error=logs/example_job-%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module load anaconda/py39
source activate myenv

python train.py --config configs/experiment.yaml
```

Key `#SBATCH` options (common ones):
- `--job-name`: readable name for the job
- `--output`, `--error`: where STDOUT/STDERR go (use `%j` for job id)
- `--time`: runtime limit
- `--partition` or `--qos`: scheduler partition/queue
- `--gres=gpu:N`: request N GPUs
- `--cpus-per-task`, `--mem`: CPU cores and memory

Tips:
- Always write output and checkpoint paths to a persistent, shared storage.
- Use small test jobs (`--time=00:15:00`) to validate your script before large runs.

## 5) Monitoring and controlling jobs

- See your running jobs:

```bash
squeue -u <username>
```

- Show detailed info about a job:

```bash
scontrol show job <JobID>
```

- Cancel a job:

```bash
scancel <JobID>
```

- SLURM produces output files you can inspect:

```bash
cat logs/example_job-<JobID>.out
cat logs/example_job-<JobID>.err
tail -f logs/example_job-<JobID>.out  # follow live logs
```

More tips:
- Use `sacct -j <JobID>` to get accounting information after completion.
- `watch -n 5 squeue -u <username>` can be useful for live monitoring (but beware rate limits).

## 6) Interactive sessions and debugging

- Request an interactive GPU session for debugging:

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash
```

- Or request a small interactive node with `salloc`.
- Use `tmux` or `screen` inside interactive nodes to keep long-lived shells.

## 7) Environment management & reproducibility

- Prefer declarative environment specs: `environment.yml` (conda) or `requirements.txt` (pip).
- In batch scripts, always load modules or activate conda environments explicitly.
- Save the exact Git commit hash with job outputs to reproduce experiments:

```bash
git rev-parse --short HEAD > logs/commit_hash.txt
```

## 8) Transferring files

- Copy files to/from the cluster using `scp` or `rsync`:

```bash
# copy local -> remote
scp -r myproject/ <username>@131.159.110.9:/vol/miltank/users/<username>/projects/

# sync and resume interrupted transfers
rsync -avP mydata/ <username>@131.159.110.9:/vol/miltank/users/<username>/datasets/
```

Tips:
- For large datasets, upload once and reuse from shared storage.
- Compress files before transfer when possible: `tar -czf data.tgz data/`.


## 9) Short reference commands

```bash
# submit job
sbatch run_job.sh
# list jobs
squeue -u <username>
# job details
scontrol show job <JobID>
# cancel
scancel <JobID>
# follow output
tail -f logs/example_job-<JobID>.out
# interactive debugging
srun --partition=gpu --gres=gpu:1 --pty bash
```
