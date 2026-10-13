"""Select a maximally diverse subset of designs under pairwise cutoffs.

For the common requirement "N designs, no two more similar than X". Sequence
identity comes from biotite's Needleman-Wunsch with BLOSUM62, so no additional
dependency is needed; TM-score from an upstream ``foldseek`` step can be
applied as a second, structural cutoff.
"""

import logging
from typing import Optional

import polars as pl
from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler
from pipelines.data.schema import Column, dump_sequences

from ._sequence_resolution import resolve_sequences

logger = logging.getLogger(__name__)


def _identity_matrix(sequences: list[str]) -> list[list[float]]:
    """All-pairs Needleman-Wunsch identity under BLOSUM62.

    O(n^2) alignments; for the few-hundred-design pools this block is meant
    for that is seconds, and it is exact rather than a clustering proxy.
    """
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import (
        SubstitutionMatrix,
        align_optimal,
        get_sequence_identity,
    )

    matrix = SubstitutionMatrix.std_protein_matrix()
    parsed = [ProteinSequence(s) for s in sequences]
    size = len(parsed)
    identity = [[0.0] * size for _ in range(size)]
    for i in range(size):
        identity[i][i] = 1.0
        for j in range(i + 1, size):
            alignment = align_optimal(
                parsed[i], parsed[j], matrix, gap_penalty=(-10, -0.5), max_number=1
            )[0]
            value = get_sequence_identity(alignment)
            identity[i][j] = identity[j][i] = value
    return identity


def _independent_set(
    conflicts: list[set[int]], ranking: list[int], target: Optional[int]
) -> list[int]:
    """Greedy maximum independent set, best of several orderings.

    Exact MIS is NP-hard, so try a quality-ordered pass and repeated
    minimum-degree elimination, and keep whichever selects most. Deterministic:
    no RNG, so a rerun reproduces the same subset.
    """
    best: list[int] = []

    def grow(order: list[int]) -> list[int]:
        chosen: list[int] = []
        blocked: set[int] = set()
        for node in order:
            if node in blocked:
                continue
            chosen.append(node)
            blocked |= conflicts[node]
            if target is not None and len(chosen) == target:
                break
        return chosen

    best = grow(ranking)
    if target is not None and len(best) >= target:
        return best

    # Minimum-degree first: nodes conflicting with least else are cheapest to
    # take, which usually admits more of the pool than quality order alone.
    remaining = set(range(len(conflicts)))
    order: list[int] = []
    while remaining:
        node = min(remaining, key=lambda n: (len(conflicts[n] & remaining), n))
        order.append(node)
        remaining.discard(node)
    candidate = grow(order)
    return candidate if len(candidate) > len(best) else best


class SelectDiverseBlock(BlockFunction):
    """Keep a subset of rows no two of which exceed the similarity cutoffs.

    Sequence identity is computed exactly (Needleman-Wunsch, BLOSUM62). If a
    TM-score metric is already present from an upstream ``foldseek`` step, it
    is applied as a second, structural cutoff.

    Example::

        SelectDiverse = Blocks.get("select_diverse")
        block = SelectDiverse(max_identity=0.5, target_count=96)
    """

    _name = "select_diverse"
    _display_name = "Select Diverse"
    _description = "Select a diverse design subset under sequence/structure cutoffs"
    _inline = True
    _chunked = False

    max_identity: float = Field(
        default=0.5,
        description="Reject a design sharing more than this identity with a kept one",
    )
    target_count: Optional[int] = Field(
        default=None,
        description="Stop once this many are selected; None keeps every non-conflicting design",
    )
    rank_by: Optional[str] = Field(
        default=None,
        description="Metric column to prefer when breaking conflicts (best first)",
    )
    rank_ascending: bool = Field(
        default=True,
        description="True when a lower rank_by value is better (e.g. PAE)",
    )
    sequence_key: str = Field(
        default="sequence",
        description=(
            "Column holding the amino-acid sequence. The default resolves standard "
            "sequence_<CHAIN> columns and structure-backed protein sequences."
        ),
    )
    require_target: bool = Field(
        default=False,
        description="Raise if target_count cannot be met, rather than warning",
    )
    required_structure_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Only select rows containing outputs from every named structure step "
            "(for example ['rf3']); use this to prevent unscreened top-ups"
        ),
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        frame = handler.unpack(all=True, to_pandas=True)
        resolution = resolve_sequences(handler, frame, self.sequence_key)
        candidate_indices = list(range(len(frame)))
        if self.required_structure_steps:
            eligible = frame.index.to_series().notna()
            for step in self.required_structure_steps:
                column = f"{step}.path"
                if column not in frame.columns:
                    raise ValueError(
                        f"select_diverse requires structure step {step!r}, but "
                        f"column {column!r} is absent"
                    )
                eligible &= frame[column].notna() & frame[column].astype(str).ne("")
            dropped = int((~eligible).sum())
            if dropped:
                logger.info(
                    "select_diverse excluded %d row(s) missing required structure "
                    "steps %s",
                    dropped,
                    self.required_structure_steps,
                )
            candidate_indices = [
                index for index, keep in enumerate(eligible.tolist()) if keep
            ]
            frame = frame.loc[eligible].reset_index(drop=True)
            if frame.empty:
                raise ValueError(
                    "select_diverse found no rows containing required structure "
                    f"steps {self.required_structure_steps}"
                )
        sequences = [resolution.values[index] for index in candidate_indices]
        missing = [i for i, s in enumerate(sequences) if not s or s == "nan"]
        if missing:
            # A blank sequence would align against everything at identity 0 and
            # sail through the filter as spuriously "diverse".
            raise ValueError(
                f"{len(missing)} row(s) have no sequence in {self.sequence_key!r}; "
                "refusing to treat them as diverse"
            )

        identity = _identity_matrix(sequences)
        size = len(sequences)
        conflicts: list[set[int]] = [set() for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                if identity[i][j] > self.max_identity:
                    conflicts[i].add(j)
                    conflicts[j].add(i)

        if self.rank_by:
            if self.rank_by not in frame.columns:
                raise ValueError(f"rank_by column {self.rank_by!r} is not present")
            ranking = list(
                frame[self.rank_by]
                .rank(ascending=self.rank_ascending, method="first")
                .sort_values()
                .index
            )
        else:
            ranking = list(range(size))

        keep = sorted(_independent_set(conflicts, ranking, self.target_count))
        logger.info(
            "select_diverse kept %d/%d designs (max_identity=%.2f)",
            len(keep),
            size,
            self.max_identity,
        )
        if self.target_count is not None and len(keep) < self.target_count:
            message = (
                f"select_diverse found {len(keep)} mutually dissimilar designs "
                f"under max_identity={self.max_identity} from a pool of {size}; "
                f"target_count was {self.target_count}."
            )
            if self.require_target:
                raise ValueError(message)
            logger.warning(message)

        # Preserve the selected packed rows rather than round-tripping the flat
        # analysis table. This retains structure lineage and works for native
        # sequence-only handlers. If sequences were derived from coordinates,
        # persist the chain map so downstream submission blocks do not need to
        # parse the structures a second time.
        selected_indices = [candidate_indices[index] for index in keep]
        selected = handler._df.gather(selected_indices)
        if resolution.sequence_maps is not None:
            selected = selected.with_columns(
                pl.Series(
                    str(Column.SEQUENCES),
                    [
                        dump_sequences(resolution.sequence_maps[index])
                        for index in selected_indices
                    ],
                    dtype=pl.Utf8,
                )
            )
        return PipelineDataHandler(selected)
