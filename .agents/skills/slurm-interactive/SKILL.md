---
name: slurm-interactive
description: "Allocate one persistent interactive Slurm shell for live GPU or CPU debugging. Reuse that shell for all commands. Use slurm-background for detached campaigns."
allowed-tools: ["Bash", "Read", "AskUserQuestion"]
---

# Slurm interactive session

Use one interactive allocation for bounded pipeline tests and iterative debugging.
Every command runs in the same shell on the compute node, preserving its working
directory and environment. Long campaigns belong in the `slurm-background` skill.

## Discover resources

Check for an existing allocation or an attached GPU before allocating another:

```bash
squeue -u "$USER" -o "%i %j %T %P %N"
nvidia-smi
sinfo -o "%P %G %l"
```

Reuse an allocation belonging to this task. Never cancel unrelated jobs or remove
another user's/session's tmux shell. Partition names and GPU types are specific
to each cluster; select values actually advertised by `sinfo`. Some sites require
an account or QOS as well. Increasing host `--mem` does not increase GPU memory.

## Allocate one persistent shell

In a tool with persistent PTY support, run:

```bash
srun --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=4 \
  --mem=32G --time=01:00:00 --job-name=chaperone_debug --pty bash --noprofile --norc
```

Replace the partition and GRES with configured values. Omit `--gres` for CPU
work. If the terminal tool returns a session ID, send every subsequent command to
that session; do not start a fresh `srun` per command. A site-provided `qlogin`
wrapper is also usable after checking its flags and defaults.

For terminals without persistent PTY support, run the same allocation command
inside a uniquely named tmux session, and send commands to that pane. Record the
job ID and session name owned by this task. Do not blindly kill a shared session.

After the shell opens, verify the node and visible device, then activate a Python
environment stored on shared storage:

```bash
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source /path/to/shared/.venv/bin/activate
cd /path/to/shared/chaperone
```

## Test inside the allocation

Run scripts in the foreground, using the engine's local mode so they use this
allocation rather than submitting nested GPU jobs:

```bash
python pipeline.py --local --test
```

Retain outputs and logs in the configured shared run directory. Node-local
`/tmp` is suitable for disposable build scratch, but cannot be inspected from
a login node and may disappear when the allocation ends. For work that needs to
survive disconnect, use the background skill and its monitor tools.

## Release and diagnose

Exit the allocated shell when finished. If it is unreachable, cancel only its
recorded job ID with `scancel <job-id>` and close only the tmux session you own.

| Problem | Action |
|---|---|
| Invalid GRES or partition | Check `sinfo`; use a registered GPU type and partition. |
| Pending allocation | Read the pending reason; choose a supported queue, smaller request, or wait. |
| CUDA unavailable | Confirm you are in the allocated shell and inspect GPU/container driver support. |
| GPU OOM | Reduce the workload or request a device with more GPU memory. |
| Shell disappeared | Check the recorded job's state with `sacct` before allocating again. |
