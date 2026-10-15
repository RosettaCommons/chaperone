"""Pipeline manifest for tracking execution state."""

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

from chaperone.execution_tracking import ExecutionSummary, summarize_execution_dir


logger = logging.getLogger(__name__)


def _from_dict(cls, data: dict, context: str):
    """Build a dataclass from *data*, dropping keys the dataclass does not define.

    ``pipeline.json`` lives in a shared rundir and is routinely read by a
    checkout other than the one that wrote it.  Splatting the stored dict
    straight into the constructor makes every field ever added to the schema a
    hard ``TypeError`` for older readers, which takes down the whole manifest
    rather than the one field they cannot interpret.  Drop unknown keys with a
    warning instead so an old reader degrades rather than dies.
    """
    known = {f.name for f in fields(cls)}
    extra = sorted(set(data) - known)
    if extra:
        logger.warning(
            "%s: ignoring unknown field(s) %s -- this manifest was written by a "
            "newer version of chaperone than the one reading it",
            context,
            extra,
        )
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class StepInfo:
    """Information about a completed pipeline step."""

    block_name: str  # _name of the block
    name: str
    pipeline_input_names: List[str]  # names of the blocks that were used as input.
    parquet: str  # Relative path from rundir
    execution_summary: ExecutionSummary | None = None

    def __post_init__(self):
        if isinstance(self.execution_summary, dict):
            self.execution_summary = _from_dict(
                ExecutionSummary,
                self.execution_summary,
                f"manifest step {self.name!r} execution_summary",
            )


class PipelineManifest:
    """Tracks completed pipeline steps and their parquet files."""

    def __init__(self, rundir: Path):
        self.rundir = Path(rundir)
        self.path = self.rundir / "pipeline.json"
        self.steps: List[StepInfo] = []

    @classmethod
    def load(cls, rundir: Path) -> "PipelineManifest":
        """Load ``pipeline.json`` from rundir, or create an empty manifest."""
        manifest = cls(rundir)
        if manifest.path.exists():
            with open(manifest.path) as f:
                data = json.load(f)
            manifest.steps = [
                _from_dict(StepInfo, step, f"manifest step {step.get('name', '?')!r}")
                for step in data.get("steps", [])
            ]
        return manifest

    def save(self) -> None:
        """Save ``pipeline.json`` atomically (temp file + ``os.replace``)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"steps": [asdict(step) for step in self.steps]}
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def add_step(
        self,
        block_name: str,
        name: str,
        parquet: Path,
        pipeline_input_names: List[str],
    ) -> None:
        """Add a completed step to the manifest.

        If a step with the same name already exists:
        - If parquet paths match: reposition it to the end (so step order
          reflects the latest execution; idempotent re-registration must not
          leave a step stranded mid-list, which corrupts funnel chaining) and
          refresh ``pipeline_input_names`` from the new call when provided.
        - If parquet paths differ: raise error (conflict)
        """
        parquet_relative = str(Path(parquet).relative_to(self.rundir))

        # Check for existing step with same name
        existing = self.get_step(name)
        if existing is not None:
            if existing.parquet == parquet_relative:
                # Same step, same parquet — reposition + refresh lightweight
                # fields so the manifest tracks the current run's order and
                # input lineage even on a full cache hit. Guard the parent
                # update so an empty re-register can't clobber a recorded parent.
                self.steps = [s for s in self.steps if s.name != name]
                if pipeline_input_names:
                    existing.pipeline_input_names = pipeline_input_names
                self.steps.append(existing)
                return
            else:
                raise ValueError(
                    f"Step '{name}' already exists with different parquet path.\n"
                    f"  Existing: {existing.parquet}\n"
                    f"  New:      {parquet_relative}"
                )

        self.steps.append(
            StepInfo(
                block_name=block_name,
                name=name,
                parquet=parquet_relative,
                pipeline_input_names=pipeline_input_names,
            )
        )

    def clear_namespace(self, namespace: str) -> None:
        """Remove all steps belonging to a namespace.

        Steps are considered part of a namespace if their name starts with the
        namespace prefix (e.g., "opt/cycle_0/rfd3" belongs to "opt/cycle_0").

        Args:
            namespace: Namespace prefix to clear
        """
        prefix = f"{namespace}/"
        self.steps = [s for s in self.steps if not s.name.startswith(prefix)]

    def invalidate_downstream(self, step_name: str) -> List[StepInfo]:
        """Remove all steps registered after the given step.

        Steps after ``step_name`` in the ordered list are considered
        downstream.

        Args:
            step_name: Name of the step whose downstream should be invalidated.

        Returns:
            List of removed StepInfo objects (caller handles file cleanup).
        """
        idx = None
        for i, s in enumerate(self.steps):
            if s.name == step_name:
                idx = i
                break

        if idx is None:
            return []  # Step not in manifest yet (first run)

        removed = self.steps[idx + 1 :]
        self.steps = self.steps[: idx + 1]

        return removed

    def step_names(self) -> List[str]:
        """Get ordered list of step names."""
        return [s.name for s in self.steps]

    def first_step_name(self) -> Optional[str]:
        """Get the first step name, or None if empty."""
        return self.steps[0].name if self.steps else None

    def last_step_name(self) -> Optional[str]:
        """Get the last step name, or None if empty."""
        return self.steps[-1].name if self.steps else None

    def get_step(self, name: str) -> Optional[StepInfo]:
        """Get step info by name."""

        if name == "first":
            name = self.first_step_name()
        elif name == "last":
            name = self.last_step_name()

        for s in self.steps:
            if s.name == name:
                return s
        return None

    def get_parquet_path(self, name: str) -> Path:
        """Get absolute path to step's parquet file."""
        step = self.get_step(name)
        if step is None:
            raise ValueError(f"Step '{name}' not found in manifest")
        return self.rundir / step.parquet

    def set_step_execution_summary(
        self, step_name: str, summary: ExecutionSummary | None
    ) -> None:
        """Attach an execution summary to a step."""
        step = self.get_step(step_name)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found in manifest")
        step.execution_summary = summary

    def refresh_step_execution_summary(self, step_name: str) -> ExecutionSummary | None:
        """Recompute and store a step summary from durable execution records."""
        summary = summarize_execution_dir(self.rundir / step_name)
        self.set_step_execution_summary(step_name, summary)
        return summary

    def execution_totals(self) -> ExecutionSummary:
        """Aggregate all execution summaries stored in the manifest."""
        totals = ExecutionSummary()
        for step in self.steps:
            if step.execution_summary is not None:
                totals.add(step.execution_summary)
        return totals

    def set_last_step(self, name: str) -> None:
        """Set a step as the last step by moving it to the end of the steps list.

        This affects what "last" resolves to in pipeline input resolution.
        Useful for switching between branches in a pipeline.

        Args:
            name: Name of the step to make last

        Raises:
            ValueError: If step not found in manifest
        """
        step = self.get_step(name)
        if step is None:
            raise ValueError(f"Step '{name}' not found in manifest")

        # Remove from current position and append to end
        self.steps = [s for s in self.steps if s.name != name]
        self.steps.append(step)

    def __len__(self) -> int:
        return len(self.steps)

    def resolve_updates(self, updates: Optional[str]) -> Optional[Path]:
        """Resolve updates target to parquet path, or None for new step."""
        if updates is None:
            return None
        step_name = self._resolve_step_ref(updates)
        return self.get_parquet_path(step_name)

    def _resolve_step_ref(self, ref: str) -> str:
        """Resolve a step reference ('last', 'first', or exact name)."""
        if ref == "last":
            name = self.last_step_name()
            if name is None:
                raise ValueError("No steps in manifest to resolve 'last'")
            return name
        if ref == "first":
            name = self.first_step_name()
            if name is None:
                raise ValueError("No steps in manifest to resolve 'first'")
            return name
        return ref

    def __repr__(self) -> str:
        if not self.steps:
            return "PipelineManifest(empty)"

        lines = [f"PipelineManifest({len(self.steps)} steps)"]
        last_name = self.last_step_name()
        for i, step in enumerate(self.steps):
            # Build dependency arrow
            if step.pipeline_input_names:
                inputs = ", ".join(step.pipeline_input_names)
                arrow = f"← [{inputs}]"
            else:
                arrow = "(root)"

            # Mark last step
            marker = " *" if step.name == last_name else ""

            # Show step with its block type and inputs
            prefix = "└──" if i == len(self.steps) - 1 else "├──"
            lines.append(f"  {prefix} {step.name} ({step.block_name}) {arrow}{marker}")

        return "\n".join(lines)
