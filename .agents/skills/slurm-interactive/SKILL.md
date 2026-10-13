---
name: slurm-interactive
description: "Allocate a persistent INTERACTIVE SLURM session (GPU or CPU node) for live, iterative work — qlogin + tmux, one allocation, every command runs in the SAME shell on the node (like SSH). Use when the user says 'get me a gpu', 'allocate a gpu/cpu', 'I need a gpu', 'interactive node', 'gpu session', 'qlogin', or when you need to run/debug a sequence of commands live on a compute node. For detached batch jobs use the slurm-background skill instead."
allowed-tools: ["Bash", "Read", "AskUserQuestion"]
---

# SLURM Interactive Session

> **IPD-specific commands.** `qlogin`, the `gpu` / `cpu` partitions and the `gpu:small` / `gpu:large` GRES types below are the Baker-lab / IPD cluster. On another cluster, substitute your site's equivalent (`salloc`, or `srun --pty bash -l`) and look up its partitions and GPU types with `sinfo -o "%P %G"`. If you are unsure, `--gres=gpu:1` is accepted nearly everywhere and needs no partition name.

Hold a persistent **interactive** allocation on a GPU or CPU node and drive it from Claude Code via `tmux` + `qlogin`. **Every command runs inside the SAME shell on the allocated node** — like an SSH session — under **one** allocation (one job step total), not a new job per command.

## When to use interactive (vs background)

Use an **interactive** session (this skill) when you want a live shell on a node:
- Testing a block / short pipeline end-to-end (`--test`, 1 design) before scaling.
- Debugging on-node (import errors, CUDA, OOM, shapes) where you iterate command-by-command.
- Quick one-offs: inspect a structure, run `nvidia-smi`, probe an env.
- Anything where you read output and decide the next command.

Use **background** submission (`slurm-background` skill) instead when the work is long (pilot/production/campaign), fans into many jobs, or must survive disconnect.

**Rule of thumb: iterating → interactive; scaling → background.**

> **One interactive session at a time.** Before allocating, check `squeue -u $USER -n claude_sess` and `tmux list-sessions | grep slurm_sess`; reuse if present, release a stale one fully (see *Releasing*) before starting a new one. Also confirm no GPU is already attached locally (`nvidia-smi`).

## Why qlogin (one allocation), not salloc + per-command srun

`qlogin` opens **one** persistent interactive shell; commands are typed into it via `tmux send-keys`, so the whole session is a single job step. The old `salloc` + `srun --jobid=` pattern submits a **fresh `srun` step per command** — hundreds of `.0/.1/.2…` records in `sacct`, billed against fair-share and pushing your priority down.

Verified empirically: under qlogin, shell state (PID, env vars, cwd) **persists** across commands, and `sacct` shows exactly one job + `.extern` + a single `.0` step no matter how many commands you run. The interactive shell is a direct child of `slurmstepd` — it runs inside the allocation's cgroup on the node.

`qlogin` on Baker-lab clusters is a thin wrapper:
```bash
$ cat /usr/bin/qlogin
#!/bin/bash
srun -p cpu -q interactive -t 12:00:00 --pty -N 1 -n 1 -I300 $* $SHELL -l
```
`$*` is appended *before* the shell, so extra flags (`-p gpu --gres=...`) override the defaults.

## Allocating

### GPU node
```bash
tmux kill-session -t slurm_sess 2>/dev/null   # clear any stale session
tmux new-session -d -s slurm_sess \
  "qlogin -p gpu --gres=gpu:small:1 --mem=16G -t 2:00:00 -J claude_sess"
```
- **gres types are per-cluster.** IPD registers size classes, `gpu:small:N` and `gpu:large:N` (`sinfo -p gpu -o "%G" -h` shows only those), so model names like `gpu:a4000:1` are rejected there and the allocation dies instantly. Most other clusters register model names (`gpu:a100:1`) or accept only `gpu:1`. Chaperone does not check the type: whatever `gres` string you give reaches `sbatch` unchanged, so the cluster decides. Use `gpu:large:1` for a bigger card at IPD.
- Avoid `--partition=gpu-bf` for interactive — backfill jobs get preempted.

### CPU node
```bash
tmux kill-session -t slurm_sess 2>/dev/null
tmux new-session -d -s slurm_sess \
  "qlogin -p cpu -c 8 --mem=16G -t 2:00:00 -J claude_sess"
```
(no `--gres`; `qlogin` already defaults to `-p cpu`, shown explicitly here.)

### Wait for it
```bash
for i in $(seq 1 24); do
  squeue -u $USER -n claude_sess -h -o "%i %T %N" | grep RUNNING && break
  sleep 5
done
```
Once `RUNNING <node>`, the tmux pane is sitting at the node's shell. Track the **node name**.

## The `slurm_run` helper

Drives the tmux'd shell: writes the command, captures stdout/stderr, waits on a done-marker.

> **Critical:** markers MUST live on a **shared filesystem** (NFS `$HOME`), not `/tmp`. `/tmp` is **node-local** — a marker the compute node writes to its `/tmp` is invisible to the poller on the login node, so the helper hangs forever. Use `$HOME/.slurm_run`.

```bash
mkdir -p "$HOME/.slurm_run"
slurm_run() {
  local cmd="$*"; local id="r$(date +%s%N)"
  local out="$HOME/.slurm_run/${id}.out"; local done="$HOME/.slurm_run/${id}.done"
  tmux send-keys -t slurm_sess "{ ${cmd}; } > ${out} 2>&1; echo __EXIT_\$?__ >> ${out}; touch ${done}" Enter
  for _ in $(seq 1 600); do ls "$HOME/.slurm_run" >/dev/null 2>&1; [ -f "${done}" ] && break; sleep 1; done
  cat "${out}"; rm -f "${out}" "${done}"
}

slurm_run "hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
slurm_run "source /path/to/.venv/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'"
```
Tips: quote multi-word commands; chain with `&&` inside the quotes to `cd`/`source` first. Because it's one persistent shell, `export`/`cd` from one `slurm_run` **persist** into the next. The node's own `/tmp` is fine for the *work* (scratch reads/writes) — only the *markers* need the shared path.

## Running a script
```bash
slurm_run "source /path/to/.venv/bin/activate && cd /path/to/repo && python my_script.py"
```
Long jobs: run in the foreground (just wait), or — if it's genuinely long-running — prefer the `slurm-background` skill.

## State tracking / recovery
```bash
squeue -u $USER -n claude_sess -h -o "%i %T %N"
tmux list-sessions | grep slurm_sess
```
If the tmux session is gone but the job is still `RUNNING`, re-attach the pane, or `scancel` and redo.

## Releasing
```bash
scancel -u $USER -n claude_sess 2>/dev/null
tmux kill-session -t slurm_sess 2>/dev/null
rm -rf "$HOME/.slurm_run"
```
Always release when done, or when the user ends the conversation / no longer needs it.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Allocation dies instantly / `Invalid generic resource (gres) specification` | The GPU type is not registered on this cluster. At IPD, model names (`gpu:a4000:1`) are invalid: use `gpu:small:1` / `gpu:large:1`. Elsewhere, check `sinfo -o "%P %G"` for the valid types, or use `gpu:1`. |
| `slurm_run` hangs forever | Markers on node-local `/tmp`. Use `$HOME/.slurm_run` (shared NFS) as above. |
| `slurm_run` returns empty | The qlogin shell died. `squeue -u $USER -n claude_sess`; re-allocate if gone. |
| Job stays `PENDING` | Try `--gres=gpu:1` (any), a shorter `-t`, or check `sinfo -p gpu`. |
| `tmux: duplicate session` | `tmux kill-session -t slurm_sess` then retry. |
| GPU OOM | Request `--gres=gpu:large:1` or raise `--mem`. |
| `qlogin` rejects `-p gpu` | Fall back: `srun -q interactive -p gpu --gres=gpu:small:1 --mem=16G -t 2:00:00 --pty -N 1 -n 1 -I300 bash -l` inside the tmux session. |
