"""SLURM job configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


def _create_slurm_executor(folder: str):
    """Create Submitit's Slurm executor with the configured shared Python."""
    import submitit

    from chaperone import get_setting_field

    kwargs = {}
    slurm_python = get_setting_field("PIPELINES_SLURM_PYTHON", None)
    if slurm_python:
        kwargs["slurm_python"] = slurm_python
    try:
        return submitit.AutoExecutor(folder=folder, cluster="slurm", **kwargs)
    except RuntimeError as exc:
        if 'Could not detect "srun"' not in str(exc):
            raise
        raise RuntimeError(
            "Slurm execution was requested, but Submitit cannot find 'srun'. "
            "Expose the cluster's srun executable in the submission environment; "
            "Pipelines will not silently run the workload locally."
        ) from exc


@dataclass
class JobArgs:
    """SLURM job configuration. Use partition='local' for local execution."""

    partition: str = field(
        default="cpu",
        metadata={
            "description": "SLURM partition ('cpu', 'gpu', etc.) or 'local' for local execution"
        },
    )
    mem_gb: int = field(
        default=8,
        metadata={"description": "Memory per job in GB"},
    )
    time_limit: str = field(
        default="5:00:00",
        metadata={"description": "Time limit in HH:MM:SS format"},
    )
    job_name: str | None = field(
        default=None,
        metadata={"description": "Custom SLURM job name (default: 'submitit')"},
    )
    gres: str | None = field(
        default=None,
        metadata={
            "description": "SLURM gres specification (e.g., 'gpu:a100:1', 'gpu:1')"
        },
    )
    constraint: str | None = field(
        default=None,
        metadata={
            "description": "SLURM node feature constraint (for example 'Ampere')"
        },
    )
    cpus_per_task: int = field(
        default=1,
        metadata={"description": "CPU cores allocated per task/job"},
    )
    rows_per_job: int | None = field(
        default=None,
        metadata={
            "description": "Number of rows to process per job (overrides block default from resource_requirements.yaml)"
        },
    )
    num_chunks: int | None = field(
        default=None,
        metadata={
            "description": "Number of chunks to split data into (overrides rows_per_job when set)"
        },
    )
    exclude: str | None = field(
        default=None,
        metadata={
            "description": "SLURM nodes to exclude as a comma-separated string. "
            "Alias for exclude_nodes."
        },
    )
    exclude_nodes: str | None = field(
        default=None,
        metadata={
            "description": "Backward-compatible alias for exclude. "
            "SLURM nodes to exclude as a comma-separated string."
        },
    )
    local_workers: int | None = field(
        default=None,
        metadata={
            "description": "Number of parallel threads for local execution. "
            "When None (default), chunks run sequentially. "
            "Has no effect when partition != 'local'."
        },
    )
    slurm_array_parallelism: int | None = field(
        default=None,
        metadata={
            "description": (
                "Maximum simultaneously running Slurm array tasks. This caps "
                "concurrency without changing rows_per_job or chunk memory."
            )
        },
    )
    nice: int | None = field(
        default=None,
        metadata={
            "description": "SLURM --nice value (higher = lower priority). "
            "None (default) omits the flag entirely."
        },
    )

    def __post_init__(self):
        if (
            self.slurm_array_parallelism is not None
            and self.slurm_array_parallelism < 1
        ):
            raise ValueError("slurm_array_parallelism must be >= 1")
        if self.exclude and self.exclude_nodes and self.exclude != self.exclude_nodes:
            raise ValueError(
                "JobArgs received both exclude and exclude_nodes with different values. "
                "Use one or make them match."
            )
        if self.exclude is None and self.exclude_nodes is not None:
            self.exclude = self.exclude_nodes
        elif self.exclude is not None and self.exclude_nodes is None:
            self.exclude_nodes = self.exclude

    # ── helpers ──────────────────────────────────────────────────────────

    def effective_rows_per_job(
        self, total_rows: int, default: int | None = None
    ) -> int:
        """Return the rows-per-chunk this config implies for ``total_rows``.

        This is the single source of truth for chunk-layout math: cache
        checks, dispatch, and chunk-correction all route through it so they
        cannot diverge.

        Resolution order:
            1. ``num_chunks`` set → ``ceil(total_rows / num_chunks)``.
            2. ``rows_per_job`` set → that value.
            3. ``default`` (the block's ``_rows_per_job``) if provided.
            4. ``total_rows`` (single chunk) as a last-resort fallback so
               callers never get ``None`` back.
        """
        if self.num_chunks is not None:
            return max(1, -(-total_rows // self.num_chunks))  # ceil division
        if self.rows_per_job is not None:
            return self.rows_per_job
        if default is not None:
            return default
        return max(1, total_rows)

    @property
    def excluded_nodes_set(self) -> set[str]:
        """Parse excluded SLURM nodes into a set."""
        value = self.exclude or self.exclude_nodes
        if not value:
            return set()
        return {n.strip() for n in value.split(",") if n.strip()}

    def with_excluded_nodes(self, nodes: set[str]) -> JobArgs:
        """Return a copy with additional nodes excluded."""
        merged = self.excluded_nodes_set | nodes
        if not merged:
            return self
        merged_str = ",".join(sorted(merged))
        return replace(self, exclude=merged_str, exclude_nodes=merged_str)

    @staticmethod
    def _parse_time_to_minutes(time_str: str) -> int:
        """Parse HH:MM:SS or MM:SS or MM to minutes."""
        parts = time_str.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) // 60
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) // 60
        return int(parts[0])

    @property
    def timeout_min(self) -> int:
        return self._parse_time_to_minutes(self.time_limit)

    def effective_slurm_array_parallelism(self, override: int | None = None) -> int:
        """Resolve explicit, per-job, environment, then library concurrency."""
        if override is not None:
            value = override
        elif self.slurm_array_parallelism is not None:
            value = self.slurm_array_parallelism
        else:
            from chaperone import get_setting_field

            value = get_setting_field("PIPELINES_SLURM_ARRAY_PARALLELISM", None)
        value = 1000 if value is None else int(value)
        if value < 1:
            raise ValueError("slurm_array_parallelism must be >= 1")
        return value

    def create_executor(
        self,
        log_folder: str = "slurm_logs",
        slurm_array_parallelism: int | None = None,
        **extra_params,
    ):
        """Create a configured ``submitit.AutoExecutor`` from these args.

        Returns:
            A ``submitit.AutoExecutor`` ready for ``.submit()`` / ``.map_array()``.
        """
        executor = _create_slurm_executor(folder=f"{log_folder}/%j")
        params: dict = {
            "slurm_partition": self.partition,
            "timeout_min": self.timeout_min,
            "mem_gb": self.mem_gb,
            **extra_params,
        }
        params["slurm_array_parallelism"] = self.effective_slurm_array_parallelism(
            slurm_array_parallelism
        )
        if self.gres:
            params["slurm_gres"] = self.gres
        if self.constraint:
            params["slurm_constraint"] = self.constraint
        params["cpus_per_task"] = self.cpus_per_task
        if self.exclude:
            params["slurm_exclude"] = self.exclude
        if self.job_name:
            params["name"] = self.job_name
        if self.nice is not None:
            params.setdefault("slurm_additional_parameters", {})["nice"] = self.nice
        executor.update_parameters(**params)
        return executor
