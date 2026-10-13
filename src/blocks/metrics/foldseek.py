"""Foldseek structural clustering block."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import Field

from blocks.utils.geometry import get_chain_annotation
from pipelines import BlockFunction, PipelineDataHandler
from pipelines import get_setting_field

logger = logging.getLogger(__name__)


def _num_unique_chains(atom_array):
    return len(np.unique(get_chain_annotation(atom_array)))


def _threshold_key(tm: float) -> str:
    """Format a TM threshold as a metric key suffix: 0.5 -> '0_50', 0.65 -> '0_65'."""
    return f"{tm:.2f}".replace(".", "_")


class FoldseekBlock(BlockFunction):
    """Cluster structures by structural similarity using Foldseek.

    Runs foldseek easy-multimercluster (multimers) or easy-cluster (monomers)
    on all structures in the handler. Each row receives cluster_id and n_clusters
    metrics at each TM threshold.

    Multimer mode is auto-detected. Falls back to easy-cluster if
    easy-multimercluster fails.
    """

    _name = "foldseek"
    _display_name = "Foldseek"
    _description = "Cluster structures by structural similarity using Foldseek"
    _creates_new_step = False
    _inline = True
    _chunked = False

    foldseek_exec: str = Field(
        default_factory=lambda: get_setting_field("FOLDSEEK_EXEC", check_bin=True),
        description="Path to foldseek binary",
    )
    tm_thresholds: List[float] = Field(
        default=[0.50, 0.60, 0.70, 0.80],
        description="TM-score thresholds for clustering. Results are output as "
        "cluster_id.<threshold> and n_clusters.<threshold> for each.",
    )
    alignment_type: int = Field(
        default=1,
        description="1=structure only, 2=structure+sequence",
    )
    min_seq_id: float = Field(
        default=0.0,
        description="Min sequence identity for clustering (alignment_type=2 only)",
    )
    multimer_tm_threshold: float = Field(
        default=0.65,
        description="TM threshold for multimer clustering",
    )
    interface_lddt_threshold: float = Field(
        default=0.65,
        description="Interface lDDT threshold for multimer clustering",
    )
    chains: Optional[List[str]] = Field(
        default=None,
        description="Chain IDs to cluster on. If None, all chains are used.",
    )
    group_by_task: bool = Field(
        default=True,
        description="If True, cluster within each task group separately "
        "(uses the first structure's metadata.task field).",
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        from atomworks.io.utils.io_utils import to_pdb_string

        tmp_dir = Path(sandbox_dir) / "foldseek"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        input_dir = tmp_dir / "samples"
        input_dir.mkdir(parents=True, exist_ok=True)

        # Stream structures to disk one at a time to avoid holding all in memory
        contain_multimers = False
        loaded_meta = []
        for i, loaded in enumerate(handler.get_data(which="last", load=True)):
            arr = loaded.atom_array

            if self.chains is not None:
                chain_annot = get_chain_annotation(arr)
                mask = np.isin(chain_annot, self.chains)
                if not mask.any():
                    # chain_iid format like "A_1" — strip suffix and retry
                    chain_bases = np.array([c.split("_")[0] for c in chain_annot])
                    mask = np.isin(chain_bases, self.chains)
                    if not mask.any():
                        raise ValueError(
                            f"Chains {self.chains} not found in {loaded.path}"
                        )
                arr = arr[mask]

            dest = input_dir / f"{i}.pdb"
            with open(dest, "w") as f:
                f.write(to_pdb_string(arr))

            if not contain_multimers and _num_unique_chains(arr) > 1:
                contain_multimers = True

            # Keep the LoadedStructure for output but drop the atom array
            loaded.atom_array = None
            loaded_meta.append(loaded)

        n = len(loaded_meta)
        if n == 0:
            raise ValueError(
                "FoldseekBlock requires at least one structure in the handler."
            )

        # Determine task groups for per-group clustering
        if self.group_by_task:
            task_for_idx = {}
            for i, loaded in enumerate(loaded_meta):
                task = loaded.metadata.get("task", None)
                task_for_idx[i] = task
            unique_tasks = sorted(set(task_for_idx.values()))
            groups = {
                t: [i for i, tt in task_for_idx.items() if tt == t]
                for t in unique_tasks
            }
        else:
            groups = {"_all": list(range(n))}

        # Prepare group symlink dirs (shared across thresholds)
        group_input_dirs = {}
        for group_name, indices in groups.items():
            group_input_dir = tmp_dir / f"group_{group_name}" / "samples"
            group_input_dir.mkdir(parents=True, exist_ok=True)
            for local_idx, global_idx in enumerate(indices):
                src = input_dir / f"{global_idx}.pdb"
                dst = group_input_dir / f"{local_idx}.pdb"
                dst.symlink_to(src)
            group_input_dirs[group_name] = group_input_dir

        # Run clustering at each threshold for each group
        # results[tm_key][global_idx] = cluster_id
        # n_clusters_results[tm_key][group_name] = n_clusters
        results = {}
        n_clusters_results = {}

        for tm in sorted(self.tm_thresholds):
            tk = _threshold_key(tm)
            cluster_assignments = [None] * n
            n_clusters_per_group = {}
            global_cluster_offset = 0

            for group_name, indices in groups.items():
                group_input_dir = group_input_dirs[group_name]
                # Each threshold gets its own work dir to avoid collisions
                work_dir = tmp_dir / f"group_{group_name}" / f"tm_{tk}"
                work_dir.mkdir(parents=True, exist_ok=True)

                group_assignments = self._run_clustering(
                    group_input_dir, work_dir, contain_multimers, len(indices), tm
                )

                if group_assignments is not None:
                    n_group_clusters = len(set(group_assignments))
                    n_clusters_per_group[group_name] = n_group_clusters
                    for local_idx, global_idx in enumerate(indices):
                        cluster_assignments[global_idx] = (
                            group_assignments[local_idx] + global_cluster_offset
                        )
                    global_cluster_offset += n_group_clusters
                else:
                    n_clusters_per_group[group_name] = None

            total = sum(v for v in n_clusters_per_group.values() if v is not None)
            logger.info(
                "Foldseek (tm=%s): %d total clusters from %d structures (%d groups)",
                tk,
                total,
                n,
                len(groups),
            )

            results[tk] = cluster_assignments
            n_clusters_results[tk] = n_clusters_per_group

        # Build output rows with metrics for all thresholds
        rows_out = []
        for i, loaded in enumerate(loaded_meta):
            task = loaded.metadata.get("task", "_all") if self.group_by_task else "_all"
            metrics = {}
            for tm in sorted(self.tm_thresholds):
                tk = _threshold_key(tm)
                metrics[f"cluster_id.{tk}"] = results[tk][i]
                metrics[f"n_clusters.{tk}"] = n_clusters_results[tk].get(task)
            rows_out.append(loaded.to_row_with_updates(metrics={self.name: metrics}))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return PipelineDataHandler(rows_out)

    def _run_clustering(
        self, input_dir, tmp_dir, contain_multimers, n_structures, tm_threshold
    ):
        """Run foldseek clustering, falling back from multimercluster to easy-cluster."""
        res_prefix = str(tmp_dir / "res")
        tsv_path = tmp_dir / "res_cluster.tsv"

        if contain_multimers:
            cmd = [
                self.foldseek_exec,
                "easy-multimercluster",
                str(input_dir),
                res_prefix,
                str(tmp_dir),
                "--alignment-type",
                str(self.alignment_type),
                "--cov-mode",
                "0",
                "--multimer-tm-threshold",
                str(self.multimer_tm_threshold),
                "--chain-tm-threshold",
                str(tm_threshold),
                "--interface-lddt-threshold",
                str(self.interface_lddt_threshold),
            ]
            try:
                subprocess.run(cmd, check=True, text=True, capture_output=True)
                return self._parse_cluster_tsv(tsv_path, n_structures)
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "easy-multimercluster failed, falling back to easy-cluster: %s",
                    e.stderr.strip(),
                )
                # Clean up failed results before fallback
                for p in tmp_dir.glob("res*"):
                    if p.is_file():
                        p.unlink()

        cmd = [
            self.foldseek_exec,
            "easy-cluster",
            str(input_dir),
            res_prefix,
            str(tmp_dir),
            "--alignment-type",
            str(self.alignment_type),
            "--cov-mode",
            "0",
            "--min-seq-id",
            str(self.min_seq_id),
            "--tmscore-threshold",
            str(tm_threshold),
        ]
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("Foldseek clustering failed: %s\nstderr: %s", e, e.stderr)
            return None

        return self._parse_cluster_tsv(tsv_path, n_structures)

    @staticmethod
    def _parse_cluster_tsv(tsv_path, n_structures):
        """Parse foldseek res_cluster.tsv into a list of cluster IDs per input file.

        Foldseek names entries as '{file_stem}_{chain}' (e.g., '0_A', '12_B').
        This method extracts the file index from each member and maps back to
        a per-file cluster assignment.
        """
        rep_for_member = {}  # member -> representative
        with open(tsv_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    rep_for_member[parts[1]] = parts[0]

        # Extract file index from foldseek names (e.g., "12_A" -> 12, "12" -> 12)
        file_to_rep = {}
        for member, rep in rep_for_member.items():
            # Split off chain suffix: "12_A" -> "12"
            file_idx = int(member.rsplit("_", 1)[0]) if "_" in member else int(member)
            if file_idx not in file_to_rep:
                file_to_rep[file_idx] = rep

        if len(file_to_rep) != n_structures:
            logger.error(
                "Foldseek clustered %d structures, expected %d",
                len(file_to_rep),
                n_structures,
            )
            return None

        # Map unique representatives to sequential cluster IDs
        unique_reps = sorted(set(file_to_rep.values()))
        cluster_map = {rep: idx for idx, rep in enumerate(unique_reps)}
        return [cluster_map[file_to_rep[i]] for i in range(n_structures)]
