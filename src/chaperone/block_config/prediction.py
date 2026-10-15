"""AF3LikePredictionBlock: Base class for structure prediction / oracle blocks."""

import getpass
import logging
import warnings
from collections import defaultdict
from typing import Callable, ClassVar, List, Optional

from pydantic import Field

from chaperone.block_config.base import TIER_PRIMARY, TIER_SECONDARY
from chaperone.block_config.compiled import CompiledBlockFunction

logger = logging.getLogger(__name__)


def select_best_models(
    paths: list[str],
    group_fn: Callable[[str], object],
    score_fn: Callable[[str], Optional[float]],
) -> list[str]:
    """Pick the highest-scoring output per design group.

    Shared "best model" selector for the oracle compile paths (Boltz2, Chai1,
    OpenFold3, Protenix), each of which previously hand-rolled the same
    group-then-argmax loop:

      * ``group_fn(path)`` → a hashable grouping key (parent dir, query id,
        design name, …). Paths sharing a key are competing samples.
      * ``score_fn(path)`` → a numeric score (higher = better) or ``None`` when
        the score is unavailable for that path.

    Groups with a single member skip scoring. Within a group, the first path is
    the fallback when no score is available.
    """
    by_group: dict[object, list[str]] = defaultdict(list)
    for path in paths:
        by_group[group_fn(path)].append(path)

    best_paths = []
    for group_paths in by_group.values():
        if len(group_paths) == 1:
            best_paths.append(group_paths[0])
            continue
        best_score = -float("inf")
        best_path = group_paths[0]
        for path in group_paths:
            score = score_fn(path)
            if score is not None and score > best_score:
                best_score = score
                best_path = path
        best_paths.append(best_path)
    return best_paths


# ── Canonical interface-confidence metric keys ───────────────────────────
#
# Oracles emit one of two *different* interface-confidence quantities. They
# must NOT be aliased onto a single key, because they have opposite polarity
# and are not interchangeable in a filter:
#
#   MIN_PAE_INTERACTION_KEY ("min_pae_interaction")
#       PAE-derived (Predicted Aligned Error, Å). Lower is better. Emitted by
#       oracles whose confidence output exposes a chain-pair PAE matrix
#       (AF3: chain_pair_pae_min off-diagonal min; RF3: chain_pair_pae_min
#       anti-diagonal min). This is the canonical cross-oracle PAE key — a
#       filter on it works on every oracle that produces a PAE matrix.
#
#   MIN_INTERFACE_IPTM_KEY ("min_interface_iptm")
#       ipTM-derived (interface predicted TM-score, 0–1). Higher is better.
#       Emitted by oracles whose confidence output exposes only a pairwise
#       ipTM matrix and NO PAE matrix (Boltz2: pair_chains_iptm off-diagonal
#       min; OpenFold3: chain_pair_iptm off-diagonal min). These oracles
#       genuinely cannot produce min_pae_interaction, so it is intentionally
#       absent rather than aliased to the ipTM value.
#   INTERFACE_MEAN_PAE_KEY ("interface_mean_pae")
#       PAE-derived (Å, lower is better). Unlike MIN_PAE_INTERACTION_KEY, which
#       takes the single best (min) chain-pair PAE entry, this averages the full
#       per-token PAE (both directions) between the ligand and only the pocket
#       residues — protein residues with any atom within a contact cutoff (Å) of
#       the ligand (AF3 field `mean_pae_contact_distance`, default 5.0). By
#       excluding distant, non-interface residues it reflects the actual binding
#       interface and is far less protein-length-dependent than an all-pairs mean.
#       The ligand side defaults to all heavy atoms; set the AF3 field
#       `mean_pae_core_atoms` to a list of ligand atom names to score only the
#       buried core, which gates BOTH the pocket-residue selection and the PAE
#       columns (so an exposed linker neither recruits residues nor dilutes the
#       average). Compare `RemapLigandBlock.core_atoms` / `RMSDBlock.atom_groups`.
#       Pocket residues are additionally gated on sidechain orientation when
#       `mean_pae_sidechain_angle` is set (AF3, default 90°): a residue counts
#       only if its sidechain projects toward the ligand rather than away.
#
#   FULL_MEAN_PAE_KEY ("full_mean_pae")
#       PAE-derived (Å, lower is better). Like INTERFACE_MEAN_PAE_KEY this is a
#       protein↔ligand cross-block mean (both directions), but with NO gating on
#       the protein side: EVERY binder token contributes, not only pocket
#       residues, and there is no sidechain-orientation filter. It is therefore
#       protein-length-dependent by construction. This mirrors bcov's
#       `ligand_mean_pae` in `run_alphafold_sm.py` (binder-mask × ligand-mask).
#       The ligand side still honours `mean_pae_core_atoms` (bcov's `-rmsd_atoms`
#       equivalent). Use `interface_mean_pae` for a length-robust pocket-focused
#       score and `full_mean_pae` for the bcov-comparable whole-binder mean.
MIN_PAE_INTERACTION_KEY = "min_pae_interaction"
MIN_INTERFACE_IPTM_KEY = "min_interface_iptm"
INTERFACE_MEAN_PAE_KEY = "interface_mean_pae"
FULL_MEAN_PAE_KEY = "full_mean_pae"


class AF3LikePredictionBlock(CompiledBlockFunction):
    """Base class for structure prediction / oracle blocks (AF3, Boltz2, RF3, Chai1).

    Provides standardized fields for prediction samples, confidence reading,
    recycling, diffusion steps, and ligand handling.

    Interface-confidence metric keys
    --------------------------------
    Oracles that report an interface confidence emit ONE of two keys, never
    both aliased to the same number (see ``MIN_PAE_INTERACTION_KEY`` /
    ``MIN_INTERFACE_IPTM_KEY`` module constants):

    * ``min_pae_interaction`` — PAE-derived (Å, lower is better). The canonical
      cross-oracle PAE key. Emitted by AF3 and RF3.
    * ``min_interface_iptm`` — ipTM-derived (0–1, higher is better). Emitted by
      Boltz2 and OpenFold3, which expose only an ipTM matrix in their
      confidence output and therefore cannot produce a PAE-derived metric.

    Filter on ``min_pae_interaction`` for PAE-based oracles; on
    ``min_interface_iptm`` for ipTM-based oracles. Do not assume one key works
    across oracles of different families.
    """

    _category: ClassVar[str] = "prediction"
    _accepts_sequence_input: ClassVar[bool] = True

    # Standardized samples field
    diffusion_batch_size: int = Field(
        5,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="Number of diffusion samples (prediction structures) per input.",
    )

    # Whether to return only the most confident model or all samples
    read_most_confident_model: bool = Field(
        True,
        json_schema_extra={"tier": TIER_PRIMARY},
        description="If True, read only the highest-confidence model; False, read all samples.",
    )

    # Recycling and diffusion steps
    n_recycles: Optional[int] = Field(
        None,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Number of trunk recycling steps. None uses the model default.",
    )
    num_diffusion_steps: int = Field(
        200,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Number of diffusion sampling steps.",
    )

    # Checkpoint path
    ckpt_path: Optional[str] = Field(
        None,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Optional checkpoint path for model weights.",
    )

    # Compile as metric (path_handle instead of path)
    compile_as_metric: bool = Field(
        False,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="If True, compile outputs as metrics with path_handle instead of path.",
    )

    # Shared ligand handling fields (used by AF3, Boltz2, Chai1; ignored by RF3)
    remove_ligands: bool = Field(
        False,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="If True, skip ligands in input preparation.",
    )
    specified_smiles: dict = Field(
        default_factory=dict,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Map ligand name → SMILES string.",
    )
    specified_ccd: dict = Field(
        default_factory=dict,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="Map ligand name → CCD code.",
    )
    disable_ccd_detection: bool = Field(
        False,
        json_schema_extra={"tier": TIER_SECONDARY},
        description="If True, disable automatic CCD code detection.",
    )

    # Which structure(s) in each row's lineage to feed to the oracle. Default
    # "last" uses the most recent structure. Matters for cycling / FilterBlock
    # lineage where the most recent entry may be a no-op filter duplicate;
    # callers can pass explicit indices (e.g. [-6, -1]) to skip them.
    which: List[int] | str = Field(
        "last",
        json_schema_extra={"tier": TIER_SECONDARY},
        description=(
            "Which structure(s) in each row's lineage to predict. 'last' "
            "(default) uses the most recent; a list of ints selects by index."
        ),
    )

    # ── Throughput: pre-chunk length sort ────────────────────────────────────
    # Sorting inputs by token length before chunking keeps each chunk within a
    # narrow JIT-bucket range, cutting recompilation/padding waste. Safe to
    # default on: the engine merges chunk outputs back to parents by structure
    # path/lineage, not row position, so re-ordering never corrupts lineage.
    sort_by_length: bool = Field(
        True,
        json_schema_extra={"tier": TIER_SECONDARY},
        description=(
            "If True (default), sort the input handler by `sort_length_key` "
            "before chunking so each chunk hits a narrow JIT-bucket range. "
            "Falls back to no sort (with a warning) if the key is absent."
        ),
    )
    sort_length_key: str = Field(
        "rfd3.num_residues",
        json_schema_extra={"tier": TIER_SECONDARY},
        description=(
            "Metric column to sort on when sort_by_length=True (dot-path into "
            "metrics JSON, as exposed by handler.unpack()). If missing from the "
            "handler, sorting is skipped with a warning rather than erroring."
        ),
    )

    def pre_chunk_sort(self, handler):
        """Engine-called hook: re-order input handler before chunking.

        No-op when sort_by_length=False. When True, sorts by
        `self.sort_length_key` ascending so each `rows_per_job` chunk covers a
        narrow length range (→ each chunk hits 1-2 JIT buckets instead of
        thrashing across many). Because this defaults on and not every input
        carries the length key (e.g. folding from sequences/CSVs), a missing
        key degrades gracefully to no sort with a warning instead of raising.
        """
        if not self.sort_by_length:
            return handler
        from chaperone.data import PipelineDataHandler

        df = handler.unpack(to_pandas=False)
        if self.sort_length_key not in df.columns:
            available = sorted(c for c in df.columns if "." in c)
            logger.warning(
                f"[{self.name}] sort_by_length=True but column "
                f"'{self.sort_length_key}' not found in handler metrics; "
                f"skipping length sort. Available metric columns: {available}. "
                f"Set sort_length_key to one of these to enable sorting."
            )
            return handler
        tie_breaker_col = "__pre_chunk_sort_idx"
        sorted_df = df.with_row_index(tie_breaker_col).sort(
            [self.sort_length_key, tie_breaker_col],
            nulls_last=True,
            maintain_order=True,
        )
        logger.info(
            f"[{self.name}] pre_chunk_sort: sorted {len(sorted_df)} rows by "
            f"'{self.sort_length_key}'"
        )
        # Reorder the original rows. Repacking the flattened view can discard
        # nested residue maps and replace a rewound context with a stashed path.
        result = PipelineDataHandler(handler)
        result._df = handler._df[sorted_df[tie_breaker_col].to_list()]
        return result

    def _jit_safe_env(self) -> dict:
        """Env vars that force JIT/temp writes to writable in-container paths.

        Many oracles JIT-compile CUDA kernels at first use (DeepSpeed evoformer,
        Protenix fast_layer_norm, etc.) and write intermediates under TMPDIR /
        TORCH_EXTENSIONS_DIR. A stale SLURM scratch TMPDIR inherited from the
        host (or a read-only container filesystem) breaks nvcc / the torch
        extension build when running --local. Pin both to /tmp.

        Subclasses extend the returned dict with model-specific env (cache
        dirs, allocator config) before passing it to
        ``image.build_command(..., env=env)``.
        """
        # User-scope the torch-extensions dir: a shared /tmp/torch_extensions is
        # owned by whichever user first JIT-builds on a node, so other users hit
        # PermissionError on busy nodes. Per-user keeps node-local /tmp writable.
        user = getpass.getuser()
        return {
            "TMPDIR": "/tmp",
            "TORCH_EXTENSIONS_DIR": f"/tmp/torch_extensions_{user}",
        }

    @classmethod
    def _remap_legacy_samples_field(cls, data: dict, old_name: str) -> dict:
        """Remap a legacy samples field name to diffusion_batch_size with a deprecation warning."""
        if isinstance(data, dict) and old_name in data:
            if "diffusion_batch_size" not in data:
                warnings.warn(
                    f"'{old_name}' is deprecated, use 'diffusion_batch_size' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                data["diffusion_batch_size"] = data.pop(old_name)
            else:
                # Both provided; drop the old one silently
                data.pop(old_name)
        return data

    @classmethod
    def _remap_legacy_field(cls, data: dict, old_name: str, new_name: str) -> dict:
        """Remap a legacy field name to a new name with a deprecation warning."""
        if isinstance(data, dict) and old_name in data:
            if new_name not in data:
                warnings.warn(
                    f"'{old_name}' is deprecated, use '{new_name}' instead.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                data[new_name] = data.pop(old_name)
            else:
                data.pop(old_name)
        return data

    def _assert_unsupported_fields_default(self, **field_defaults):
        """Assert that unsupported fields are at their default values.

        Call from model_validator(mode="after") on blocks that don't support
        certain inherited fields. Example:
            self._assert_unsupported_fields_default(ckpt_path=None, n_recycles=None)
        """
        for field_name, default in field_defaults.items():
            actual = getattr(self, field_name)
            if actual != default:
                raise ValueError(
                    f"{self.__class__.__name__} does not support '{field_name}' "
                    f"(got {actual!r}). This parameter is not passed to the "
                    f"underlying model."
                )
