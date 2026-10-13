Follow the appended task prompt. It is the authority for the scientific goal,
acceptance criteria, independence requirements, and `designs.csv` schema. You
have {{AGENT_HOURS}} hours to deliver exactly {{REQUIRED_N}} genuine designs.

## Workspace and harness

Work in the current directory. Its real shared-filesystem path is
`$HARBOR_HOST_WORKSPACE`; `/workspace` is an alias, but Pipelines commands and
generated job paths must use the real path so Slurm workers can resolve them.
Task inputs are in `inputs/`, and the writable harness checkout is `pipelines/`.
When a Slurm-visible absolute input path is required, derive it as
`Path(os.environ["HARBOR_HOST_WORKSPACE"]) / "inputs" / <name>`; do not infer
the task-input directory by walking upward from a campaign script.
The checkout is a history-free source snapshot and intentionally has no
`.git/` directory; do not run Git commands inside it.

Read `pipelines/AGENTS.md`; it is authoritative for harness behavior. Ordinary
working artifacts, caches, native compaction, and prompt caching remain
available; there is no campaign ledger or monitor bookkeeping protocol.

Run every computational stage through Pipelines. Keep campaign code, public
metrics, and results under `pipelines/campaigns/{{TASK_NAME}}/`. Never access
or reconstruct the held-out scorer. `pipeline.py` may mechanically generate
whatever internal metrics and working artifacts it needs; do not package them
for evaluation.

Use `inputs/runtime_capabilities.json` as the verified tool manifest. Use any
supplied public scorer directly for scientific selection without copying it
into the submission.

## Runtime

The shell and guarded workers already run in the hardened Pipelines image. Do
not rebuild it or launch nested `apptainer exec`. Invoke it directly:

```bash
/opt/pipelines/.venv/bin/pipelines --help
/opt/pipelines/.venv/bin/python pipelines/path/to/campaign.py
```

Campaign scripts created with `pipelines.init_cli()` support `-h`, `--help`,
`--helpshort`, and `--helpfull` for read-only flag inspection.

Use configured shared checkpoints/caches without downloading duplicates.
Install missing open-source dependencies only inside the workspace; everything
outside it is read-only.

For AF3-like prediction blocks, `specified_smiles` is always a mapping from the
exact ligand residue name to its SMILES string, for example
`specified_smiles={"LIG": "c1ccccc1"}`; never pass a bare SMILES string.

## Compute routing

The outer process has four CPU cores, limited memory, and no GPU. Keep
`PIPELINES_COMPUTE_BACKEND=auto` and route all work with explicit Pipelines
`JobArgs`:

- Use `partition="local"` only for lightweight CPU work that comfortably fits
  alongside the agent.
- Use `partition="cpu"` for larger or uncertain-memory CPU work, without
  `gres` or another GPU request.
- Use `partition="gpu"` with `gres="gpu:small:N"` for every GPU workload. The
  controller pins accepted jobs to A4000 nodes; do not add a constraint
  yourself. Untyped GPUs, model-specific GPU types, other GPU classes, and
  other partitions are rejected.
- At most five small-class GPUs may be reserved across the entire trial,
  queued/running jobs and array concurrency. A one-GPU array may use `%5`, not
  `%6`; an active three-GPU job leaves two GPUs for all other work.
- Independently, each GPU submission may request at most 30 GPUs in aggregate
  across all array tasks, ignoring its `%N` concurrency throttle. For example,
  a one-GPU `--array=0-29%5` is allowed, but `--array=0-30%5` is rejected.
  `HARBOR_MAX_REQUESTED_GPUS_PER_SUBMISSION=30` reports this limit.
- Set `JobArgs(slurm_array_parallelism=5)` on one-GPU arrays (or a lower value
  while other GPU jobs are active). Choose `rows_per_job` from per-worker
  memory and runtime; never enlarge chunks merely to reduce array concurrency.
- Do not call `sbatch`, `srun`, or `salloc` manually; Pipelines/Submitit owns
  submission through the guard.

## Submission

Write `submission/designs.csv` using only the columns allowed by the task
prompt; do not add internal metrics, structures, coordinates, or paths.
`PipelineDataHandler.unpack()` returns a pandas DataFrame by default, so write
it with `DataFrame.to_csv(...)`. Pass `to_pandas=False` only when deliberately
using Polars and its `DataFrame.write_csv(...)` method.
This sequence table is the only evaluation deliverable. Harbor retains the
transcript, and the controller measures compute and publishes audit metadata.
Legacy auxiliary files are ignored.
