"""Shared utilities for design-eval submissions and held-out scorers.

The public eval contract is intentionally a small, flat table. CSV is preferred
for easy inspection while debugging, and parquet is accepted for larger runs.
Scorers validate the table before spending compute, then convert each row into
an internal ``PipelineDataHandler`` for use with structure-prediction blocks.
Submitted coordinates and structure paths are never accepted.

Single-sequence tasks use a bare ``sequence`` column by default.  It is
normalized internally to the task's configured chain.  The legacy
``sequence_A`` spelling remains accepted for compatibility.  Multi-chain tasks
use chain-qualified columns — ``sequence_H``, ``sequence_L``, … — because one
bare column cannot represent more than one submitted chain.

The second half of this module is the scorer toolkit: flags, run directories,
criteria evaluation, and the report/per-design writers that every ``score.py``
shares. Scorers should contain their science and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl

from chaperone.data.handler import PipelineDataHandler
from chaperone.data.schema import Column, make_row

if TYPE_CHECKING:
    from chaperone.block_config.base import BlockFunction

CANONICAL_AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
CANONICAL_RNA_PATTERN = re.compile(r"^[ACGU]+$")
MOTIF_INDEX_PATTERN = re.compile(r"^([A-Za-z])([1-9][0-9]*)$")
SEQUENCE_COLUMN_PATTERN = re.compile(r"^sequence_([A-Za-z][A-Za-z0-9]*)$")

BASE_COLUMNS = {"design_id", "motif_indices"}

# One-letter code per three-letter residue name, for motif identity checks.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip


class SequenceSubmissionError(ValueError):
    """Raised when a submitted sequence table violates its contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("Invalid sequence submission:\n- " + "\n- ".join(errors))


class FixedRegionError(ValueError):
    """Raised when a sequence does not decompose against a fixed-region spec."""


@dataclass(frozen=True)
class VariableRegion:
    """One designable span of an otherwise fixed reference chain.

    ``start``/``end`` are 1-based inclusive indices into that chain's *reference*
    sequence. ``min_length``/``max_length`` bound the designed segment's length
    in the submitted sequence; both default to the reference span's own length,
    i.e. a fixed-length region.
    """

    name: str
    chain: str
    start: int
    end: int
    min_length: int | None = None
    max_length: int | None = None

    @property
    def reference_length(self) -> int:
        return self.end - self.start + 1

    def length_bounds(self) -> tuple[int, int]:
        low = self.reference_length if self.min_length is None else self.min_length
        high = self.reference_length if self.max_length is None else self.max_length
        return low, high


@dataclass(frozen=True)
class FixedRegionSpec:
    """A contract where only declared spans may differ from a reference sequence.

    Antibody CDR design, nanobody CDR grafting and TCR loop design all share the
    same shape: the scaffold (framework) is *supplied*, and only a handful of
    loops are designed. Mutating the scaffold is not a bad design — it is a
    different molecule, and the submission is invalid. Checking that belongs to
    the contract, before any oracle is run, which is why it lives here and not
    in a single benchmark's ``score.py``.

    ``reference`` maps chain ID to that chain's full reference sequence.
    ``regions`` declares every designable span. Everything else must be
    reproduced verbatim, in order.

    Because a designed region may change length, the check is a decomposition
    rather than a position-wise comparison: the submitted sequence must equal
    ``fixed_0 + designed_0 + fixed_1 + … + fixed_n`` for some choice of designed
    segment lengths inside the declared bounds.
    """

    reference: Mapping[str, str]
    regions: tuple[VariableRegion, ...]

    def __post_init__(self):
        if not self.reference:
            raise ValueError("FixedRegionSpec requires at least one reference chain")
        if not self.regions:
            raise ValueError("FixedRegionSpec requires at least one VariableRegion")
        seen_names: set[str] = set()
        for region in self.regions:
            if region.name in seen_names:
                raise ValueError(f"Duplicate VariableRegion name {region.name!r}")
            seen_names.add(region.name)
            if region.chain not in self.reference:
                raise ValueError(
                    f"Region {region.name!r} names chain {region.chain!r} which has "
                    f"no reference sequence"
                )
            length = len(self.reference[region.chain])
            if not (1 <= region.start <= region.end <= length):
                raise ValueError(
                    f"Region {region.name!r} span {region.start}-{region.end} is "
                    f"outside chain {region.chain} (length {length})"
                )
            low, high = region.length_bounds()
            if low < 1 or high < low:
                raise ValueError(
                    f"Region {region.name!r} has invalid length bounds {low}-{high}"
                )
        for chain in self.chains:
            ordered = self.regions_for(chain)
            for previous, region in zip(ordered, ordered[1:]):
                if region.start <= previous.end:
                    raise ValueError(
                        f"Regions {previous.name!r} and {region.name!r} overlap on "
                        f"chain {chain}"
                    )

    @property
    def chains(self) -> tuple[str, ...]:
        return tuple(self.reference)

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(region.name for region in self.regions)

    def regions_for(self, chain: str) -> tuple[VariableRegion, ...]:
        return tuple(
            sorted((r for r in self.regions if r.chain == chain), key=lambda r: r.start)
        )

    def fixed_segments(self, chain: str) -> tuple[str, ...]:
        """The invariant pieces of ``chain``, in order. May include empty ends."""
        sequence = self.reference[chain]
        segments: list[str] = []
        cursor = 0
        for region in self.regions_for(chain):
            segments.append(sequence[cursor : region.start - 1])
            cursor = region.end
        segments.append(sequence[cursor:])
        return tuple(segments)

    def _decompose(self, chain: str, sequence: str) -> list[tuple[int, int]]:
        """1-based inclusive spans of each designable region in ``sequence``.

        Raises :class:`FixedRegionError` when no decomposition exists (the
        scaffold was mutated, or a region's length is out of bounds) or when
        more than one exists (the decomposition, and therefore every
        region-level measurement derived from it, would be arbitrary).
        """
        fixed = self.fixed_segments(chain)
        variable = self.regions_for(chain)
        solutions: list[list[tuple[int, int]]] = []

        def walk(position: int, index: int, spans: list[tuple[int, int]]) -> None:
            if len(solutions) > 1:
                return
            segment = fixed[index]
            if not sequence.startswith(segment, position):
                return
            position += len(segment)
            if index == len(variable):
                if position == len(sequence):
                    solutions.append(list(spans))
                return
            low, high = variable[index].length_bounds()
            for length in range(low, high + 1):
                end = position + length
                if end > len(sequence):
                    break
                spans.append((position + 1, end))
                walk(end, index + 1, spans)
                spans.pop()

        walk(0, 0, [])

        if not solutions:
            raise FixedRegionError(
                f"chain {chain} does not preserve the supplied fixed regions: the "
                f"sequence outside {[r.name for r in variable]} must reproduce the "
                f"reference verbatim, and each designed region must respect its "
                f"declared length bounds "
                f"{ {r.name: r.length_bounds() for r in variable} }"
            )
        if len(solutions) > 1:
            raise FixedRegionError(
                f"chain {chain} decomposes ambiguously against the fixed regions; "
                "a designed segment spells part of the following fixed segment"
            )
        return solutions[0]

    def region_spans(self, sequences: Mapping[str, str]) -> dict[str, tuple[int, int]]:
        """``{region_name: (start, end)}`` 1-based inclusive, in the submission."""
        spans: dict[str, tuple[int, int]] = {}
        for chain in self.chains:
            if chain not in sequences:
                raise FixedRegionError(f"submission has no sequence for chain {chain}")
            found = self._decompose(chain, sequences[chain])
            for region, span in zip(self.regions_for(chain), found):
                spans[region.name] = span
        return spans

    def region_sequences(self, sequences: Mapping[str, str]) -> dict[str, str]:
        """``{region_name: designed_segment}`` for every declared region."""
        spans = self.region_spans(sequences)
        by_name = {region.name: region for region in self.regions}
        return {
            name: sequences[by_name[name].chain][start - 1 : end]
            for name, (start, end) in spans.items()
        }

    def fixed_positions(self, chain: str, sequence: str) -> list[tuple[int, int]]:
        """``(design_resid, reference_resid)`` pairs for every fixed position.

        Both are 1-based. This is what lets a scorer measure the scaffold it
        supplied — backbone RMSD over exactly the residues it fixed — even when
        a designed region changed length and shifted the numbering.
        """
        found = self._decompose(chain, sequence)
        regions = self.regions_for(chain)
        pairs: list[tuple[int, int]] = []
        design_position = 1
        reference_position = 1
        for index, segment in enumerate(self.fixed_segments(chain)):
            for offset in range(len(segment)):
                pairs.append((design_position + offset, reference_position + offset))
            design_position += len(segment)
            reference_position += len(segment)
            if index < len(regions):
                design_position = found[index][1] + 1
                reference_position = regions[index].end + 1
        return pairs

    def check(self, sequences: Mapping[str, str]) -> None:
        """Raise :class:`FixedRegionError` unless every chain decomposes."""
        self.region_spans(sequences)


@dataclass(frozen=True)
class SequenceSubmissionSpec:
    """Task-specific controls for the common sequence submission contract.

    ``chains`` lists the chain IDs a submission must supply a sequence for, in
    order. A single-chain task uses a bare ``sequence`` column by default and
    also accepts the legacy chain-qualified spelling.

    ``motif_reference_ids`` turns on the ``motif_index_map`` contract used by
    motif-bearing tasks (enzymes above all). A sequence-only submission carries
    no structure, so the scorer cannot tell which residues are meant to be the
    catalytic motif — the map supplies that, naming where each reference
    position landed in the submitted sequence. ``motif_residue_types`` optionally
    pins the expected one-letter residue at each reference position, so a
    mis-scaffolded motif fails the contract instead of quietly scoring badly.

    ``fixed_regions`` turns on the scaffold contract used by grafting tasks
    (antibody CDR design above all): the framework is supplied, only the
    declared regions are designed, and a submission that mutated the framework
    is rejected before any oracle is run. See :class:`FixedRegionSpec`.

    ``oligomer_copies`` covers homo-oligomer tasks, where the design *is* one
    protomer and the assembly is that protomer repeated. The submission stays a
    single ``sequence``; the scorer replicates it into ``oligomer_copies``
    identical chains before folding (see :func:`oligomer_chain_ids`). Asking a
    submitter to paste the same sequence into ``sequence_A``/``sequence_B``/
    ``sequence_C`` would make the contract express a hetero-oligomer that merely
    happens to be identical, and would silently accept a submission whose chains
    differ.

    ``molecule_type`` selects the submitted alphabet and structure materializer.
    Proteins use the canonical 20-letter alphabet and CA placeholders; RNA uses
    ``ACGU`` and RNA-typed C3' placeholders so a folding oracle cannot silently
    interpret the sequence as a peptide.

    ``motif_map_format="zero_based_indices"`` exposes the RNA-benchmark public
    contract: a JSON ``motif_map`` whose values are 0-based sequence indices.
    It is normalized internally to the repository's ``motif_index_map``
    residue-ID form. The default keeps the existing semicolon-delimited,
    one-based ``REF:A45`` contract used by protein motif tasks.

    ``max_pairwise_identity`` is a contract-wide diversity ceiling evaluated
    with exact global Needleman-Wunsch alignments. Proteins use the existing
    BLOSUM62 implementation with affine gap penalties -10/-0.5; RNA uses the
    nucleotide identity implementation from ``sequence_identity_gate``. For
    fixed-framework tasks it applies to the concatenated variable regions
    rather than the intentionally shared framework.
    """

    required_n: int = 96
    chains: tuple[str, ...] = ("A",)
    molecule_type: str = "protein"
    sequence_length_range: tuple[int, int] | None = None
    require_motif_indices: bool = False
    allow_motif_indices: bool = True
    motif_reference_ids: tuple[str, ...] = ()
    motif_residue_types: Mapping[str, str] = field(default_factory=dict)
    motif_contiguity_groups: tuple[tuple[str, ...], ...] = ()
    motif_map_format: str = "residue_ids"
    fixed_regions: "FixedRegionSpec | None" = None
    oligomer_copies: int = 1
    max_pairwise_identity: float = 0.70

    def __post_init__(self):
        if not self.chains:
            raise ValueError("SequenceSubmissionSpec requires at least one chain")
        if self.molecule_type not in {"protein", "rna"}:
            raise ValueError(
                "molecule_type must be 'protein' or 'rna', got "
                f"{self.molecule_type!r}"
            )
        if self.motif_map_format not in {"residue_ids", "zero_based_indices"}:
            raise ValueError(
                "motif_map_format must be 'residue_ids' or "
                f"'zero_based_indices', got {self.motif_map_format!r}"
            )
        if self.motif_map_format == "zero_based_indices" and not self.is_single_chain:
            raise ValueError(
                "zero_based_indices motif maps require a single submitted chain"
            )
        if self.sequence_length_range is not None:
            low, high = self.sequence_length_range
            if low < 1 or high < low:
                raise ValueError(
                    f"invalid sequence_length_range {self.sequence_length_range}"
                )
        duplicates = {c for c in self.chains if list(self.chains).count(c) > 1}
        if duplicates:
            raise ValueError(f"Duplicate chain IDs in spec: {sorted(duplicates)}")
        if self.oligomer_copies < 1:
            raise ValueError(
                f"oligomer_copies must be >= 1, got {self.oligomer_copies}"
            )
        if self.oligomer_copies > 1:
            if not self.is_single_chain:
                raise ValueError(
                    "oligomer_copies > 1 requires a single submitted chain; a "
                    f"homo-oligomer is one designed protomer, got {list(self.chains)}"
                )
            oligomer_chain_ids(self.chains[0], self.oligomer_copies)  # validates
        if not 0.0 <= self.max_pairwise_identity <= 1.0:
            raise ValueError(
                "max_pairwise_identity must be between 0 and 1, got "
                f"{self.max_pairwise_identity}"
            )
        unknown = set(self.motif_residue_types) - set(self.motif_reference_ids)
        if unknown:
            raise ValueError(
                f"motif_residue_types names positions absent from "
                f"motif_reference_ids: {sorted(unknown)}"
            )
        for group in self.motif_contiguity_groups:
            if len(group) < 2:
                raise ValueError(
                    "motif_contiguity_groups entries must contain at least two "
                    f"reference IDs, got {group}"
                )
            unknown_group = set(group) - set(self.motif_reference_ids)
            if unknown_group:
                raise ValueError(
                    f"motif contiguity group {group} names positions absent from "
                    f"motif_reference_ids: {sorted(unknown_group)}"
                )
        if self.fixed_regions is not None:
            extra = set(self.fixed_regions.chains) - set(self.chains)
            if extra:
                raise ValueError(
                    f"fixed_regions references chains {sorted(extra)} that the "
                    f"submission does not supply; chains are {list(self.chains)}"
                )

    @property
    def require_motif_index_map(self) -> bool:
        return bool(self.motif_reference_ids)

    @property
    def sequence_columns(self) -> tuple[str, ...]:
        return tuple(f"sequence_{chain}" for chain in self.chains)

    @property
    def is_single_chain(self) -> bool:
        return len(self.chains) == 1

    @property
    def allowed_columns(self) -> set[str]:
        allowed = BASE_COLUMNS | set(self.sequence_columns)
        if not self.allow_motif_indices:
            allowed.discard("motif_indices")
        allowed.add(self.motif_map_column)
        if self.is_single_chain:
            allowed.add("sequence")
        return allowed

    @property
    def motif_map_column(self) -> str:
        return (
            "motif_map"
            if self.motif_map_format == "zero_based_indices"
            else "motif_index_map"
        )

    @property
    def assembly_chains(self) -> tuple[str, ...]:
        """The chain IDs the scorer actually folds.

        Equal to ``chains`` for every ordinary task; for a homo-oligomer it is
        the submitted protomer chain replicated ``oligomer_copies`` times.
        """
        if self.oligomer_copies == 1:
            return self.chains
        return oligomer_chain_ids(self.chains[0], self.oligomer_copies)


def oligomer_chain_ids(chain: str, copies: int) -> tuple[str, ...]:
    """Chain IDs for ``copies`` identical protomers starting at ``chain``.

    ``("A", 3) -> ("A", "B", "C")``. Consecutive letters keep the assembly's
    chain naming conventional and predictable for downstream blocks, which is
    what lets a scorer name ``binder_chains=["A"], target_chains=["B"]`` for one
    of the interfaces without knowing how the CIF was built.
    """
    if copies < 1:
        raise ValueError(f"copies must be >= 1, got {copies}")
    if len(chain) != 1 or not chain.isalpha():
        raise ValueError(
            f"oligomer replication needs a single-letter chain ID, got {chain!r}"
        )
    start = ord(chain.upper())
    if start + copies - 1 > ord("Z"):
        raise ValueError(
            f"{copies} copies starting at chain {chain!r} runs past chain Z"
        )
    return tuple(chr(start + offset) for offset in range(copies))


def sequence_column_for(chain: str) -> str:
    return f"sequence_{chain}"


def chain_of(column: str) -> str | None:
    """The chain ID a sequence column refers to, if it is one."""
    match = SEQUENCE_COLUMN_PATTERN.fullmatch(column)
    return match.group(1) if match else None


def parse_motif_index_map(value: object) -> dict[str, str] | None:
    """Normalize a submitted motif map to ``{reference_id: residue_id}``.

    CSV encodes it as ``H57:A45;D102:A46``; parquet may supply a mapping
    directly. Returns ``None`` when nothing was submitted.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, Mapping):
        return {str(k): str(v) for k, v in value.items()}
    if not isinstance(value, str):
        raise ValueError(
            f"motif_index_map must be a mapping or string, got {type(value)}"
        )

    parsed: dict[str, str] = {}
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if pair.count(":") != 1:
            raise ValueError(
                f"malformed motif_index_map entry {pair!r}; expected REF:RESID"
            )
        reference, residue = (part.strip() for part in pair.split(":"))
        if not reference or not residue:
            raise ValueError(
                f"malformed motif_index_map entry {pair!r}; expected REF:RESID"
            )
        if reference in parsed:
            raise ValueError(f"duplicate motif reference {reference!r}")
        parsed[reference] = residue
    return parsed or None


def parse_zero_based_motif_map(value: object, *, chain: str) -> dict[str, str] | None:
    """Normalize a JSON ``{reference: 0-based sequence index}`` motif map.

    CSV carries the object as JSON while parquet may carry a mapping directly.
    The shared scorer uses one-based ``A45`` residue IDs internally, so this is
    the only boundary at which the public index convention is translated.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"motif_map must be a JSON object: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"motif_map must be a JSON object, got {type(value).__name__}")

    parsed: dict[str, str] = {}
    for reference, raw_index in value.items():
        reference = str(reference)
        if not reference:
            raise ValueError("motif_map reference IDs must not be empty")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise ValueError(
                f"motif_map index for {reference!r} must be an integer, "
                f"got {raw_index!r}"
            )
        if raw_index < 0:
            raise ValueError(
                f"motif_map index for {reference!r} must be >= 0, got {raw_index}"
            )
        parsed[reference] = f"{chain}{raw_index + 1}"
    return parsed or None


def format_motif_index_map(mapping: Mapping[str, str]) -> str:
    """The CSV spelling of a motif map."""
    return ";".join(f"{reference}:{residue}" for reference, residue in mapping.items())


def _validate_motif_index_map(
    mapping: dict[str, str],
    sequences: Mapping[str, str],
    spec: SequenceSubmissionSpec,
    row_index: int,
) -> list[str]:
    """Check a parsed motif map against the spec and the submitted sequences."""
    errors: list[str] = []

    if spec.motif_reference_ids:
        expected = set(spec.motif_reference_ids)
        submitted = set(mapping)
        missing = sorted(expected - submitted)
        extra = sorted(submitted - expected)
        if missing:
            errors.append(f"row {row_index}: motif_index_map is missing {missing}")
        if extra:
            errors.append(
                f"row {row_index}: motif_index_map has unknown reference positions "
                f"{extra}; expected {sorted(expected)}"
            )

    seen_residues: dict[str, str] = {}
    for reference, residue_id in mapping.items():
        match = MOTIF_INDEX_PATTERN.fullmatch(residue_id)
        if match is None:
            errors.append(
                f"row {row_index}: motif {reference!r} maps to invalid residue ID "
                f"{residue_id!r}"
            )
            continue
        chain, index = match.group(1), int(match.group(2))
        if chain not in spec.chains:
            errors.append(
                f"row {row_index}: motif {reference!r} maps to chain {chain}, "
                f"not one of {list(spec.chains)}"
            )
            continue
        sequence = sequences.get(chain, "")
        if index > len(sequence):
            errors.append(
                f"row {row_index}: motif {reference!r} maps to {residue_id} which "
                f"exceeds chain {chain} length {len(sequence)}"
            )
            continue
        # Two motif positions cannot occupy the same residue.
        if residue_id in seen_residues:
            errors.append(
                f"row {row_index}: {residue_id} is mapped by both "
                f"{seen_residues[residue_id]!r} and {reference!r}"
            )
        seen_residues[residue_id] = reference

        expected_aa = spec.motif_residue_types.get(reference)
        if expected_aa:
            expected_one = THREE_TO_ONE.get(expected_aa.upper(), expected_aa.upper())
            actual = sequence[index - 1]
            if actual != expected_one:
                errors.append(
                    f"row {row_index}: motif {reference!r} maps to {residue_id} "
                    f"which is {actual}, expected {expected_one}"
                )

    for group in spec.motif_contiguity_groups:
        if not all(reference in mapping for reference in group):
            continue
        matches = [
            MOTIF_INDEX_PATTERN.fullmatch(mapping[reference]) for reference in group
        ]
        if not all(matches):
            continue
        placed = [(match.group(1), int(match.group(2))) for match in matches]
        first_chain, first_index = placed[0]
        expected = [(first_chain, first_index + offset) for offset in range(len(group))]
        if placed != expected:
            errors.append(
                f"row {row_index}: motif positions {list(group)} must be "
                f"contiguous and in order, got {[mapping[r] for r in group]}"
            )
    return errors


def load_sequence_submission(
    path: str | Path,
    spec: SequenceSubmissionSpec = SequenceSubmissionSpec(),
) -> pl.DataFrame:
    """Load and strictly validate a sequence-only CSV or parquet table.

    Required columns:
      - ``sequence`` for a single-chain task (the legacy
        ``sequence_<CHAIN>`` spelling is also accepted).
      - ``sequence_<CHAIN>`` for every chain in a multi-chain task.

    Optional columns:
      - ``design_id``: unique non-empty identifier. Deterministically generated
        from row order when omitted.
      - ``motif_indices``: residue IDs such as ``["A10", "A42"]`` in parquet
        or ``A10;A42`` in CSV. A task may make this column mandatory through
        ``SequenceSubmissionSpec``.

    A motif-aware spec additionally accepts its configured map column:
    ``motif_index_map`` with one-based residue IDs by default, or a JSON
    ``motif_map`` with 0-based indices when
    ``motif_map_format="zero_based_indices"``.

    Returns an internal frame with ``sequence_<CHAIN>`` columns regardless of
    which public spelling was submitted.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise SequenceSubmissionError(
            ["submission must be a .csv (preferred) or .parquet file"]
        )

    try:
        frame = pl.read_csv(path) if suffix == ".csv" else pl.read_parquet(path)
    except Exception as exc:
        raise SequenceSubmissionError([f"cannot read {suffix[1:]}: {exc}"]) from exc

    errors: list[str] = []
    unexpected = sorted(set(frame.columns) - spec.allowed_columns)
    if unexpected:
        errors.append(
            f"unexpected columns {unexpected}; submissions may contain only "
            f"{sorted(spec.allowed_columns)}"
        )

    # Map each required chain to the column that supplies it.
    column_for_chain: dict[str, str] = {}
    for chain in spec.chains:
        canonical = sequence_column_for(chain)
        if spec.is_single_chain and "sequence" in frame.columns:
            column_for_chain[chain] = "sequence"
        elif canonical in frame.columns:
            column_for_chain[chain] = canonical
        else:
            required = "sequence" if spec.is_single_chain else canonical
            errors.append(f"missing required '{required}' column")
    if spec.is_single_chain and "sequence" in frame.columns:
        sole = sequence_column_for(spec.chains[0])
        if sole in frame.columns:
            errors.append(f"supply either 'sequence' or '{sole}', not both")

    if spec.require_motif_indices and "motif_indices" not in frame.columns:
        errors.append("missing task-required 'motif_indices' column")
    if spec.require_motif_index_map and spec.motif_map_column not in frame.columns:
        example = (
            '\'{"H57": 44, "D102": 45}\''
            if spec.motif_map_format == "zero_based_indices"
            else "'H57:A45;D102:A46'"
        )
        errors.append(
            f"missing task-required '{spec.motif_map_column}' column; map every "
            "reference "
            f"position {list(spec.motif_reference_ids)} to a residue in the "
            f"submitted sequence, e.g. {example}"
        )
    if len(frame) != spec.required_n:
        errors.append(f"expected {spec.required_n} sequences, found {len(frame)}")
    if errors:
        raise SequenceSubmissionError(errors)

    records = frame.to_dicts()
    normalized: list[dict] = []
    seen_ids: set[str] = set()
    seen_designs: set[tuple[str, ...]] = set()

    for row_index, row in enumerate(records):
        design_id = row.get("design_id")
        if design_id is None:
            design_id = f"design_{row_index:04d}"
        if not isinstance(design_id, str) or not design_id.strip():
            errors.append(f"row {row_index}: design_id must be a non-empty string")
        elif design_id != design_id.strip():
            errors.append(
                f"row {row_index}: design_id has leading or trailing whitespace"
            )
        elif design_id in seen_ids:
            errors.append(f"row {row_index}: duplicate design_id {design_id!r}")
        else:
            seen_ids.add(design_id)

        sequences: dict[str, str] = {}
        for chain in spec.chains:
            sequence = row.get(column_for_chain[chain])
            label = sequence_column_for(chain)
            if not isinstance(sequence, str) or not sequence:
                errors.append(f"row {row_index}: {label} must be a non-empty string")
                sequence = ""
            alphabet = (
                CANONICAL_RNA_PATTERN
                if spec.molecule_type == "rna"
                else CANONICAL_AA_PATTERN
            )
            if sequence and not alphabet.fullmatch(sequence):
                allowed = (
                    "uppercase RNA letters A, C, G, U"
                    if spec.molecule_type == "rna"
                    else "uppercase canonical amino-acid letters"
                )
                errors.append(f"row {row_index}: {label} must contain only {allowed}")
            if sequence and spec.sequence_length_range is not None:
                low, high = spec.sequence_length_range
                if not low <= len(sequence) <= high:
                    errors.append(
                        f"row {row_index}: {label} length {len(sequence)} is outside "
                        f"the required range {low}-{high}"
                    )
            sequences[chain] = sequence

        # A design is the full chain combination, not any single chain: two rows
        # sharing a binder but differing elsewhere are distinct designs.
        fingerprint = tuple(sequences[chain] for chain in spec.chains)
        if all(fingerprint):
            if fingerprint in seen_designs:
                errors.append(f"row {row_index}: duplicate design")
            seen_designs.add(fingerprint)

        # The supplied scaffold is part of the contract, not the science: a
        # mutated framework is a different molecule and must be rejected before
        # any oracle runs.
        if spec.fixed_regions is not None and all(fingerprint):
            try:
                spec.fixed_regions.check(sequences)
            except FixedRegionError as exc:
                errors.append(f"row {row_index}: {exc}")

        motif_indices = row.get("motif_indices")
        if suffix == ".csv" and isinstance(motif_indices, str):
            motif_indices = [
                motif.strip() for motif in motif_indices.split(";") if motif.strip()
            ]
        if spec.require_motif_indices and not motif_indices:
            errors.append(f"row {row_index}: motif_indices must not be empty")
        if motif_indices is not None:
            if not isinstance(motif_indices, (list, tuple)):
                errors.append(
                    f"row {row_index}: motif_indices must be a list of residue IDs"
                )
                motif_indices = []
            else:
                motif_indices = list(motif_indices)
                seen_motifs: set[str] = set()
                for motif in motif_indices:
                    match = (
                        MOTIF_INDEX_PATTERN.fullmatch(motif)
                        if isinstance(motif, str)
                        else None
                    )
                    if match is None:
                        errors.append(
                            f"row {row_index}: invalid motif residue ID {motif!r}"
                        )
                        continue
                    chain_id, residue_index = match.groups()
                    if chain_id not in spec.chains:
                        errors.append(
                            f"row {row_index}: motif {motif!r} must use one of "
                            f"chains {list(spec.chains)}"
                        )
                    elif int(residue_index) > len(sequences[chain_id]):
                        errors.append(
                            f"row {row_index}: motif {motif!r} exceeds chain "
                            f"{chain_id} length {len(sequences[chain_id])}"
                        )
                    if motif in seen_motifs:
                        errors.append(
                            f"row {row_index}: duplicate motif index {motif!r}"
                        )
                    seen_motifs.add(motif)

        motif_index_map: dict[str, str] | None = None
        if spec.motif_map_column in frame.columns or spec.require_motif_index_map:
            try:
                raw_map = row.get(spec.motif_map_column)
                motif_index_map = (
                    parse_zero_based_motif_map(raw_map, chain=spec.chains[0])
                    if spec.motif_map_format == "zero_based_indices"
                    else parse_motif_index_map(raw_map)
                )
            except ValueError as exc:
                errors.append(f"row {row_index}: {exc}")
                motif_index_map = {}
            if motif_index_map is None:
                if spec.require_motif_index_map:
                    errors.append(
                        f"row {row_index}: {spec.motif_map_column} must not be empty"
                    )
                motif_index_map = {}
            errors.extend(
                _validate_motif_index_map(motif_index_map, sequences, spec, row_index)
            )

        normalized_row = {"design_id": design_id}
        for chain in spec.chains:
            normalized_row[sequence_column_for(chain)] = sequences[chain]
        if "motif_indices" in frame.columns:
            normalized_row["motif_indices"] = motif_indices
        if motif_index_map is not None:
            normalized_row["motif_index_map"] = motif_index_map
        normalized.append(normalized_row)

    if errors:
        raise SequenceSubmissionError(errors)

    # Every submitted row must represent a genuinely distinct sequence design,
    # not a close sibling used to pad the requested count.  Use the same exact
    # Needleman-Wunsch/BLOSUM62 calculation as the public select_diverse block.
    # For a fixed-framework task (currently antibody CDR design), comparing the
    # full sequences would reject every row because the framework is supposed
    # to be identical; compare the concatenated designed regions instead.
    if spec.fixed_regions is not None:
        diversity_sequences = []
        for row in normalized:
            sequences = {
                chain: row[sequence_column_for(chain)] for chain in spec.chains
            }
            regions = spec.fixed_regions.region_sequences(sequences)
            diversity_sequences.append(
                "".join(regions[region.name] for region in spec.fixed_regions.regions)
            )
        diversity_label = "concatenated designed-region sequence"
    else:
        diversity_sequences = [
            "".join(row[sequence_column_for(chain)] for chain in spec.chains)
            for row in normalized
        ]
        diversity_label = "submitted sequence"

    if spec.molecule_type == "rna":
        from blocks.metrics.sequence_identity_gate import nw_identity

        identity = [
            [nw_identity(left, right) for right in diversity_sequences]
            for left in diversity_sequences
        ]
    else:
        from blocks.util_blocks.select_diverse import _identity_matrix

        identity = _identity_matrix(diversity_sequences)
    conflicts = []
    for i in range(len(identity)):
        for j in range(i + 1, len(identity)):
            if identity[i][j] > spec.max_pairwise_identity:
                conflicts.append(
                    (
                        normalized[i]["design_id"],
                        normalized[j]["design_id"],
                        identity[i][j],
                    )
                )
    if conflicts:
        examples = ", ".join(
            f"{left}/{right}={value:.3f}" for left, right, value in conflicts[:5]
        )
        raise SequenceSubmissionError(
            [
                f"pairwise global identity for the {diversity_label} must be <= "
                f"{spec.max_pairwise_identity:.2f}; violating pairs: {examples}"
            ]
        )
    return pl.DataFrame(normalized)


def load_context_chains(
    structure: str | Path, chains: Sequence[str] | None = None
) -> "object":
    """Scorer-supplied context to co-fold with every design.

    Binder tasks can template a single receptor chain, but an enzyme needs its
    substrate *and* its metal — several chains, including hetero ones. Loading
    them once here and attaching them to each design's input keeps the oracle a
    plain fold of the full assembly.
    """
    from chaperone.io import parse_to_atom_array

    atom_array = parse_to_atom_array(structure)
    if chains is not None:
        mask = np.isin(atom_array.chain_id, list(chains))
        if not mask.any():
            raise ValueError(f"No atoms for chains {list(chains)} in {structure}")
        atom_array = atom_array[mask]
    return atom_array


def _sequences_to_cif(
    sequences: Mapping[str, str],
    output_path: Path,
    context: "object | None" = None,
) -> Path:
    """Write one protein CA-only, coordinate-free CIF for supplied chains.

    ``context`` is an optional AtomArray of fully-coordinate chains (substrate,
    metal, cofactor) appended verbatim. RNA submissions bypass this helper and
    use ``rna_design_handler`` in :func:`sequence_submission_to_handler`.
    """
    from biotite.structure import AtomArray

    from chaperone.data.handler import AA_ONE_TO_THREE

    residues = [
        (chain, aa)
        for chain, sequence in sequences.items()
        for aa in sequence.upper()
        if aa in AA_ONE_TO_THREE
    ]
    if not residues:
        raise ValueError(
            f"No valid amino acids in submission chains: {list(sequences)}"
        )

    atom_array = AtomArray(len(residues))
    residue_index_by_chain: dict[str, int] = {}
    for index, (chain, aa) in enumerate(residues):
        residue_index_by_chain[chain] = residue_index_by_chain.get(chain, 0) + 1
        atom_array.coord[index] = [np.nan, np.nan, np.nan]
        atom_array.chain_id[index] = chain
        atom_array.res_id[index] = residue_index_by_chain[chain]
        atom_array.res_name[index] = AA_ONE_TO_THREE[aa]
        atom_array.atom_name[index] = "CA"
        atom_array.element[index] = "C"

    if context is not None and len(context):
        clash = set(np.unique(context.chain_id)) & set(sequences)
        if clash:
            raise ValueError(
                f"Context chains {sorted(clash)} collide with submitted sequence "
                "chains; the design and its context must not share chain IDs"
            )
        atom_array = atom_array + context

    from atomworks.io.utils.io_utils import to_cif_file

    to_cif_file(atom_array, str(output_path))
    return output_path


def sequence_submission_to_handler(
    submission: pl.DataFrame,
    rundir: str | Path,
    *,
    name: str = "submission",
    context: "object | None" = None,
    oligomer_copies: int = 1,
    molecule_type: str = "protein",
    ligands: Sequence[str] = (),
) -> PipelineDataHandler:
    """Materialize validated sequences as scorer-owned internal inputs.

    Every ``sequence_<CHAIN>`` column becomes a chain of the built structure.
    ``context`` (see :func:`load_context_chains`) is appended to every design so
    the oracle folds the full assembly.

    ``oligomer_copies`` > 1 replicates the single submitted chain into that many
    identical chains (``A`` -> ``A``/``B``/``C``), which is how a homo-oligomer
    task folds an assembly from a one-protomer submission. Pass
    ``spec.oligomer_copies`` so the built structure matches the declared
    contract.

    RNA takes the dedicated Chaperone ingest path, which writes nucleotide
    residue names/C3' anchors and emits ``diffused_index_map`` metadata from the
    normalized motif map. ``ligands`` are scorer-owned CCD codes (for example a metal
    ion code), identical for every design.
    """

    chains = [
        chain_of(column)
        for column in submission.columns
        if chain_of(column) is not None
    ]
    if not chains:
        raise ValueError(
            "Submission has no sequence_<CHAIN> columns; was it validated by "
            "load_sequence_submission?"
        )
    if oligomer_copies > 1 and len(chains) != 1:
        raise ValueError(
            "oligomer_copies > 1 requires a single submitted chain; found "
            f"{sorted(chains)}"
        )

    if molecule_type == "rna":
        if context is not None and len(context):
            raise ValueError(
                "RNA sequence materialization does not support AtomArray context; "
                "supply fixed cofactors through ligands"
            )
        if oligomer_copies != 1 or len(chains) != 1:
            raise ValueError("RNA sequence materialization requires one RNA chain")

        from blocks.util_blocks.rna_designs import rna_design_handler

        chain = chains[0]
        designs = []
        for row in submission.to_dicts():
            motif_index_map = row.get("motif_index_map") or {}
            motif_map = {
                reference: int(residue_id[1:]) - 1
                for reference, residue_id in motif_index_map.items()
            }
            design = {
                "design_id": row["design_id"],
                "sequence": row[sequence_column_for(chain)],
                sequence_column_for(chain): row[sequence_column_for(chain)],
            }
            if motif_map:
                design["motif_map"] = motif_map
            designs.append(design)
        return rna_design_handler(
            designs,
            Path(rundir).resolve() / "submission_inputs",
            ligands=list(ligands),
            chain=chain,
        )

    if molecule_type != "protein":
        raise ValueError(f"Unsupported molecule_type {molecule_type!r}")
    if ligands:
        raise ValueError("ligands are currently supported only for RNA submissions")

    input_dir = Path(rundir).resolve() / "submission_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_index, row in enumerate(submission.to_dicts()):
        sequences = {chain: row[sequence_column_for(chain)] for chain in chains}
        fingerprint = "".join(sequences[chain] for chain in chains)
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
        cif_path = input_dir / f"{row_index:04d}_{digest}.cif"
        if not cif_path.exists():
            built = (
                sequences
                if oligomer_copies == 1
                else {
                    copy: sequences[chains[0]]
                    for copy in oligomer_chain_ids(chains[0], oligomer_copies)
                }
            )
            _sequences_to_cif(built, cif_path, context=context)
        metrics = {"design_id": row["design_id"]}
        metrics.update({sequence_column_for(c): sequences[c] for c in chains})

        # ``motif`` is the cross-block key MPNN and the RMSD MOTIF_ALLATOM
        # selection already read, so deriving it from the map makes motif-aware
        # blocks work on sequence-only submissions with no changes.
        motif_index_map = row.get("motif_index_map") or {}
        motif = row.get("motif_indices") or (
            sorted(motif_index_map.values()) if motif_index_map else None
        )
        metadata = {"motif": motif}
        if motif_index_map:
            metadata["motif_index_map"] = motif_index_map
        rows.append(
            make_row(name=name, path=cif_path, metadata=metadata, metrics=metrics)
        )
    return PipelineDataHandler(rows)


def motif_rmsd(
    design_structure: str | Path,
    reference_structure: str | Path,
    motif_index_map: Mapping[str, str],
    *,
    backbone_only: bool = False,
) -> float:
    """All-atom RMSD of a design's mapped motif against the reference geometry.

    ``motif_index_map`` names, per reference position (``"H293"``), the residue
    of the design that plays it (``"A54"``). Atoms are paired by name within
    each residue, superimposed on those pairs, and scored. This is what makes a
    sequence-only motif submission measurable: without the map there is no
    correspondence between design residues and reference positions.
    """
    from biotite.structure import superimpose

    from chaperone.io import parse_to_atom_array

    design = parse_to_atom_array(design_structure)
    reference = parse_to_atom_array(reference_structure)
    backbone = {"N", "CA", "C", "O"}

    design_indices: list[int] = []
    reference_indices: list[int] = []
    for position, residue_id in motif_index_map.items():
        match = MOTIF_INDEX_PATTERN.fullmatch(str(residue_id))
        if match is None:
            raise ValueError(f"invalid design residue ID {residue_id!r}")
        design_chain, design_res = match.group(1), int(match.group(2))
        ref_match = re.fullmatch(r"([A-Za-z]*)([1-9][0-9]*)", str(position))
        if ref_match is None:
            raise ValueError(f"invalid reference position {position!r}")
        ref_res = int(ref_match.group(2))

        design_mask = np.flatnonzero(
            (design.chain_id == design_chain) & (design.res_id == design_res)
        )
        reference_mask = np.flatnonzero(reference.res_id == ref_res)
        if design_mask.size == 0:
            raise ValueError(f"design has no residue {residue_id}")
        if reference_mask.size == 0:
            raise ValueError(f"reference has no residue {position}")

        # Pair by atom name so differing atom order or missing side-chain atoms
        # never produce a silently mismatched superposition.
        reference_atoms = {
            str(reference.atom_name[i]): i
            for i in reference_mask
            if not backbone_only or str(reference.atom_name[i]) in backbone
        }
        for index in design_mask:
            atom_name = str(design.atom_name[index])
            if backbone_only and atom_name not in backbone:
                continue
            if atom_name in reference_atoms:
                design_indices.append(int(index))
                reference_indices.append(int(reference_atoms[atom_name]))

    if not design_indices:
        raise ValueError("No motif atoms could be paired with the reference")

    design_atoms = design[design_indices]
    reference_atoms = reference[reference_indices]
    fitted, _ = superimpose(reference_atoms, design_atoms)
    deviation = fitted.coord - reference_atoms.coord
    return float(np.sqrt((deviation**2).sum(axis=1).mean()))


def require_finite_metrics(frame: pd.DataFrame, columns: list[str]) -> None:
    """Require every named scoring metric to exist and be finite."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Scoring output is missing required columns: {missing}")
    bad = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        indices = np.flatnonzero(~np.isfinite(values))
        if indices.size:
            bad[column] = indices[:10].tolist()
    if bad:
        raise ValueError(
            "Scoring produced NaN/non-finite values; refusing to silently drop them: "
            f"{bad}"
        )


def structure_chain_ids(path: str | Path) -> set[str]:
    """Chain IDs present in a predicted structure; empty if it cannot be read."""
    from chaperone.io import parse_to_atom_array

    try:
        atom_array = parse_to_atom_array(str(path))
    except Exception:
        return set()
    return {str(chain) for chain in np.unique(atom_array.chain_id)}


def _expected_assembly_chains(
    submission: pl.DataFrame,
    spec: "SequenceSubmissionSpec",
    context: "object | None",
    oracle: "object | None" = None,
) -> list[str]:
    """Every chain the oracle should have produced for one design.

    The submitted chains (replicated when the contract is an oligomer), plus
    whatever the scorer attached as context, plus any target the *oracle*
    supplies itself. Mirrors how :func:`sequence_submission_to_handler` builds
    the input, so the two cannot disagree about what a complete prediction
    looks like.

    The oracle half matters: ``af3_with_target_template`` takes its receptor as
    a block argument rather than as context, so a task using it has no context
    chains at all and an apo fold would otherwise look complete.
    """
    chains = [
        chain_of(column)
        for column in submission.columns
        if chain_of(column) is not None
    ]
    if spec.oligomer_copies > 1 and len(chains) == 1:
        chains = list(oligomer_chain_ids(chains[0], spec.oligomer_copies))
    if context is not None and len(context):
        chains += [str(chain) for chain in np.unique(context.chain_id)]
    for attribute in ("target_chain", "target_chains"):
        supplied = getattr(oracle, attribute, None)
        if isinstance(supplied, str):
            chains.append(supplied)
        elif isinstance(supplied, (list, tuple)):
            chains += [str(chain) for chain in supplied]
    return chains


def check_folded_assembly(
    frame: pd.DataFrame, path_column: str, expected_chains: Sequence[str]
) -> tuple[int, str | None]:
    """Confirm the oracle folded the assembly the gates are written against.

    Returns ``(n_incomplete, harness_fault)``. Every gate downstream of a
    co-fold reads geometry between the design and something the *scorer*
    supplied — a target chain, a substrate, a symmetry mate. If the oracle
    never placed it, each of those gates fails by construction and the run
    reports a clean ``0/96`` that is indistinguishable from a genuinely hard
    benchmark. This failure (a sequence-only submission folded apo) can go
    unnoticed until the oracle inputs are read by hand.

    A fault is raised only when *every* design is missing a chain: some designs
    losing one is a per-design failure the gates should judge, but all of them
    losing one is the harness, not the submission.
    """
    expected = sorted(set(expected_chains))
    if not expected or path_column not in frame.columns:
        return 0, None
    incomplete = [
        index
        for index, path in enumerate(frame[path_column])
        if not set(expected) <= structure_chain_ids(path)
    ]
    n_incomplete = len(incomplete)
    if len(frame) and n_incomplete == len(frame):
        return n_incomplete, (
            f"No folded structure contains all of chains {expected}. The oracle "
            "did not fold the assembly these gates measure, so every one of "
            "them fails by construction and this score says nothing about the "
            "submission."
        )
    return n_incomplete, None


# ─────────────────────────────────────────────────────────────────────────────
# Scorer toolkit
# ─────────────────────────────────────────────────────────────────────────────


def define_score_flags(required_n: int = 96) -> None:
    """Register the flags every held-out scorer takes.

    Call once at import time, before ``init_cli()``.
    """
    from absl import flags

    flags.DEFINE_string(
        "designs",
        None,
        "Sequence-only CSV (preferred) or parquet submission table.",
    )
    flags.DEFINE_string("out", "score.json", "Aggregate score report JSON.")
    flags.DEFINE_string(
        "per_design_out",
        "score_per_design.csv",
        "Per-design metrics and pass/fail table.",
    )
    flags.DEFINE_integer(
        "required_n",
        required_n,
        "Required sequence count. Override to 1 only for a local smoke test.",
    )
    flags.DEFINE_string(
        "run_meta",
        None,
        "Optional JSON with the submitter's cost telemetry "
        "({gpu_hours, cpu_hours, wallclock_min}); enables pass-rate-per-cost.",
    )
    flags.DEFINE_string(
        "chaperone_repo",
        os.environ.get("CHAPERONE_REPO"),
        "Chaperone checkout supplying scoring assets. Passed by the Harbor "
        "controller; scorers that resolve their own assets may ignore it.",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_addressed_rundir(designs_path: str | Path, root: str | Path) -> Path:
    """A run directory keyed by submission content.

    Two different submissions never share a run directory, and re-scoring the
    same submission reuses its cached compute.
    """
    return Path(root) / file_sha256(designs_path)[:16]


def isolated_scoring_root(default: str | Path) -> Path:
    """Use the controller-assigned scoring root when one is present.

    Task scorers retain their source-local default for direct development use,
    while managed evaluations keep all mutable manifests and outputs beneath
    the trial's explicitly bound ``RUNDIR``.
    """
    configured = os.environ.get("RUNDIR")
    return Path(configured or default).expanduser().resolve()


@dataclass
class Criterion:
    """One pass/fail gate over the per-design frame."""

    name: str
    passed: pd.Series
    hard: bool = True


@dataclass
class ScoreReport:
    """The canonical report every eval writes.

    Keys are fixed across evals so downstream tooling — the published page,
    comparisons between benchmarks — never has to special-case a task.
    """

    eval: str
    n_reported: int
    required_n: int
    contract_ok: bool
    validation_oracle: str
    n_pass: int = 0
    pass_rate: float = 0.0
    n_soft_pass: int = 0
    soft_pass_rate: float = 0.0
    per_criterion: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    metric_summaries: dict = field(default_factory=dict)
    contract_errors: list[str] = field(default_factory=list)
    cost: dict | None = None
    efficiency: dict | None = None
    submission_sha256: str | None = None
    per_design: str | None = None
    # Non-null means the run is void: the oracle did not fold what the gates
    # measure, so the score says nothing about the submission. Downstream
    # tooling (eval_status.py) keys on this to exclude the run from any mean.
    harness_fault: str | None = None
    n_incomplete: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "eval": self.eval,
            "n_reported": self.n_reported,
            "required_n": self.required_n,
            "contract_ok": self.contract_ok,
            "n_pass": self.n_pass,
            "pass_rate": self.pass_rate,
            "n_soft_pass": self.n_soft_pass,
            "soft_pass_rate": self.soft_pass_rate,
            "per_criterion": self.per_criterion,
            "thresholds": self.thresholds,
            "metric_summaries": self.metric_summaries,
            "validation_oracle": self.validation_oracle,
            "harness_fault": self.harness_fault,
            "n_incomplete": self.n_incomplete,
            "cost": self.cost,
            "efficiency": self.efficiency,
            "submission_sha256": self.submission_sha256,
            "per_design": self.per_design,
        }
        if self.contract_errors:
            payload["contract_errors"] = self.contract_errors
        payload.update(self.extra)
        return payload


def summarize(values: pd.Series) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def evaluate_criteria(
    frame: pd.DataFrame,
    criteria: Sequence[Criterion],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add ``pass_*``/``pass``/``soft_pass`` columns and count each gate.

    Hard criteria must all hold for ``pass``; soft criteria additionally gate
    ``soft_pass``. Returns the augmented frame and the per-criterion counts.
    """
    if not criteria:
        raise ValueError("evaluate_criteria requires at least one criterion")

    frame = frame.copy()
    per_criterion: dict[str, int] = {}
    for criterion in criteria:
        column = f"pass_{criterion.name}"
        passed = criterion.passed.fillna(False).astype(bool)
        frame[column] = passed
        per_criterion[criterion.name] = int(passed.sum())

    hard = [f"pass_{c.name}" for c in criteria if c.hard]
    soft = [f"pass_{c.name}" for c in criteria if not c.hard]
    if not hard:
        raise ValueError("evaluate_criteria requires at least one hard criterion")

    frame["pass"] = frame[hard].all(axis=1)
    frame["soft_pass"] = (
        frame["pass"] & frame[soft].all(axis=1) if soft else frame["pass"]
    )
    return frame, per_criterion


def write_contract_failure(
    eval_name: str,
    errors: list[str],
    n_reported: int,
    *,
    required_n: int,
    validation_oracle: str,
    out: str | Path,
) -> ScoreReport:
    """Write the canonical failure report without spending held-out compute."""
    report = ScoreReport(
        eval=eval_name,
        n_reported=n_reported,
        required_n=required_n,
        contract_ok=False,
        validation_oracle=validation_oracle,
        contract_errors=errors,
    )
    Path(out).write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nContract failed; wrote {out} without running held-out compute.")
    return report


def load_submission_or_fail(
    designs: str | Path | None,
    spec: SequenceSubmissionSpec,
    *,
    eval_name: str,
    validation_oracle: str,
    out: str | Path,
) -> pl.DataFrame | None:
    """Validate a submission, or write the failure report and return ``None``.

    This is the single gate every scorer runs before touching a GPU.
    """
    if not designs:
        raise SystemExit("--designs is required")
    designs_path = Path(designs).resolve()
    try:
        return load_sequence_submission(designs_path, spec)
    except SequenceSubmissionError as exc:
        try:
            frame = (
                pl.read_csv(designs_path)
                if designs_path.suffix.lower() == ".csv"
                else pl.read_parquet(designs_path)
            )
            n_reported = len(frame)
        except Exception:
            n_reported = 0
        write_contract_failure(
            eval_name,
            exc.errors,
            n_reported,
            required_n=spec.required_n,
            validation_oracle=validation_oracle,
            out=out,
        )
        return None


def load_cost(
    run_meta: str | Path | None, n_pass: int
) -> tuple[dict | None, dict | None]:
    """Cost telemetry and the derived pass-per-hour efficiency, if supplied."""
    if not run_meta:
        print("WARNING: no --run_meta; pass-rate-per-cost unavailable.")
        return None, None
    cost = json.loads(Path(run_meta).read_text())
    gpu_hours = cost.get("gpu_hours") or 0.0
    cpu_hours = cost.get("cpu_hours") or 0.0
    efficiency = {
        "pass_per_gpu_hour": (n_pass / gpu_hours) if gpu_hours else None,
        "pass_per_cpu_hour": (n_pass / cpu_hours) if cpu_hours else None,
    }
    return cost, efficiency


def write_score_outputs(
    report: ScoreReport,
    frame: pd.DataFrame,
    *,
    out: str | Path,
    per_design_out: str | Path,
    identity_columns: Sequence[str] | None = None,
    metric_columns: Sequence[str] = (),
) -> None:
    """Write the per-design CSV and the report JSON.

    The per-design table always leads with the design's identity, then the
    named metrics, then every ``pass_*``/verdict column.
    """
    per_design_path = Path(per_design_out)
    per_design_path.parent.mkdir(parents=True, exist_ok=True)

    if identity_columns is None:
        identity_columns = [
            column
            for column in frame.columns
            if column == "design_id" or chain_of(column) is not None
        ]
    verdict_columns = [
        column
        for column in frame.columns
        if column.startswith("pass_") or column in ("pass", "soft_pass")
    ]
    columns = [
        column
        for column in [*identity_columns, *metric_columns, *verdict_columns]
        if column in frame.columns
    ]
    frame[columns].to_csv(per_design_path, index=False)

    report_path = Path(out)
    report.per_design = os.path.relpath(
        per_design_path.resolve(), start=report_path.resolve().parent
    )
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nWrote {out} and {per_design_path}")


def metric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """A numeric view of a metric column, raising when it is absent.

    Deliberately does not fill missing values: a NaN metric is an upstream bug,
    not a failing design, and must surface via ``require_finite_metrics``.
    """
    if column not in frame.columns:
        raise KeyError(f"Scoring output is missing required column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_over_paths(
    frame: pd.DataFrame, column: str, fn: Callable[[str], object]
) -> list:
    """Map a per-structure function over a path column, failing loudly by row."""
    results = []
    for row_index, path in enumerate(frame[column]):
        try:
            results.append(fn(path))
        except Exception as exc:
            raise RuntimeError(
                f"{fn.__name__} failed for row {row_index}: {exc}"
            ) from exc
    return results


def ca_radius_of_gyration(structure: str | Path, *, chain_id: str) -> float:
    """Unweighted C-alpha radius of gyration for one predicted protein chain.

    The centroid and radius are computed from the requested chain's C-alpha
    coordinates only.  Restricting the measurement to one protomer keeps it
    independent of how an oligomer is oriented or translated and avoids making
    the value depend on side-chain composition.
    """
    from chaperone.io import parse_to_atom_array

    atom_array = parse_to_atom_array(str(structure))
    mask = (atom_array.chain_id == chain_id) & (atom_array.atom_name == "CA")
    coordinates = np.asarray(atom_array.coord[mask], dtype=float)
    if len(coordinates) < 2:
        raise ValueError(
            f"{structure}: chain {chain_id!r} has fewer than two C-alpha atoms"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(
            f"{structure}: chain {chain_id!r} has non-finite C-alpha coordinates"
        )
    centered = coordinates - coordinates.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


# ─────────────────────────────────────────────────────────────────────────────
# One-call scorer
#
# Every held-out ``score.py`` repeats the same shape: validate the submission,
# fold it with an oracle, run a few metric blocks, gate the results, and write
# the canonical report. ``run_sequence_scorer`` is that shape as one typed
# function, so a scorer supplies only its science.
#
# Gates are the repo's own filter-expression strings — the exact ``expr`` DSL
# ``FilterBlock`` accepts (``compile_filter_expression``) — evaluated as pass/
# fail columns rather than as row filters, so no design is dropped. This keeps a
# single filter language across the codebase instead of an eval-only predicate
# type, and puts each gate's threshold in one place: the expression itself.
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_gate_expressions(
    handler: PipelineDataHandler, gates: Mapping[str, str]
) -> dict[str, dict[str, bool]]:
    """Evaluate each ``name -> filter expression`` as a per-design pass column.

    Reuses ``FilterBlock``'s ``compile_filter_expression`` on the packed metrics
    (so gate keys are the same dotted metric keys a filter uses, e.g.
    ``holo.plddt >= 80``), but as a boolean column, not a row filter. Returns
    ``{gate_name: {design_id: passed}}`` — keyed by ``design_id`` so the verdict
    joins back to the frame by identity, never by position.
    """
    from blocks.util_blocks.filter import compile_filter_expression

    from chaperone.data.schema import Column

    packed = handler.to_pandas()
    metrics_column = str(Column.METRICS)
    design_ids = [
        (json.loads(blob) if blob else {}).get("design_id")
        for blob in packed[metrics_column]
    ]
    eval_df = pl.DataFrame({metrics_column: list(packed[metrics_column])})

    results: dict[str, dict[str, bool]] = {}
    for name, expr in gates.items():
        column = (
            eval_df.select(compile_filter_expression(expr).alias("m"))["m"]
            .fill_null(False)
            .to_list()
        )
        results[name] = dict(zip(design_ids, column))
    return results


def merge_submission_columns(
    frame: pd.DataFrame, submission: pl.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Attach submission columns to the scored frame, keyed on ``design_id``.

    The scored frame does **not** come back in submission order:
    ``AlphaFold3Block.compile()`` collects outputs by glob and the handler merge
    iterates the new rows, so a positional ``zip`` silently pairs each structure
    with another design's data — every value stays finite and plausible, so
    nothing downstream catches it. Joining on ``design_id`` with a set-equality
    assertion is the only safe way to carry a ``motif_index_map`` (or any
    per-design input) into scoring.
    """
    if "design_id" not in frame.columns:
        raise RuntimeError(
            "Scored frame has no design_id column; cannot align submission data "
            "to structures. Refusing to fall back to positional order."
        )
    lookup = {row["design_id"]: row for row in submission.to_dicts()}
    frame_ids = list(frame["design_id"])
    if set(frame_ids) != set(lookup) or len(frame_ids) != len(set(frame_ids)):
        missing = sorted(set(lookup) - set(frame_ids))
        extra = sorted(set(frame_ids) - set(lookup))
        raise RuntimeError(
            "Scored design_ids do not match the submission's "
            f"(missing={missing[:5]}, unexpected={extra[:5]}, "
            f"n_frame={len(frame_ids)}, n_submitted={len(lookup)})"
        )
    frame = frame.copy()
    for column in columns:
        frame[column] = [lookup[design_id][column] for design_id in frame_ids]
    return frame


def attach_measurements_by_design_id(
    handler: PipelineDataHandler,
    values_by_design_id: Mapping[str, dict],
) -> PipelineDataHandler:
    """Attach measurements keyed by the submission's logical ``design_id``.

    Prediction blocks may replace a handler row's physical ``design_id`` with
    an output artifact ID (for example ``d0`` becomes ``d0_model``).  The
    scorer keeps the submitted identity in the row's metrics so ``unpack()``
    and every score table remain keyed by the public submission contract.
    ``PipelineDataHandler.with_metrics()``, however, joins on the physical row
    ID.  Build the bridge from the two identities carried by each *same* row,
    then let ``with_metrics()`` perform its normal strict physical-key join.

    This is not a positional join to the submission: both IDs below come from
    one handler row.  Requiring unique, exactly equal logical-key sets ensures
    that a missing, duplicated, or unexpected measurement still aborts rather
    than being attached to the wrong design.
    """
    packed = handler.to_pandas()
    metrics_column = str(Column.METRICS)
    physical_column = str(Column.DESIGN_ID)

    physical_ids = list(packed[physical_column])
    logical_ids = [
        (json.loads(blob) if isinstance(blob, str) else (blob or {})).get("design_id")
        for blob in packed[metrics_column]
    ]
    if any(design_id is None for design_id in logical_ids):
        raise RuntimeError(
            "Scored handler row has no logical design_id in metrics; cannot "
            "attach per-design measurements"
        )
    if len(physical_ids) != len(set(physical_ids)):
        raise RuntimeError(
            "Scored handler has duplicate physical design_ids; cannot attach "
            "per-design measurements unambiguously"
        )
    if len(logical_ids) != len(set(logical_ids)):
        raise RuntimeError(
            "Scored handler has duplicate logical design_ids; cannot attach "
            "per-design measurements unambiguously"
        )

    measured_ids = set(values_by_design_id)
    logical_id_set = set(logical_ids)
    if measured_ids != logical_id_set:
        missing = sorted(logical_id_set - measured_ids)
        extra = sorted(measured_ids - logical_id_set)
        raise KeyError(
            "Per-design measurement keys do not match the scored handler "
            f"(missing={missing[:5]}, unexpected={extra[:5]})"
        )

    values_by_physical_id = {
        physical_id: values_by_design_id[logical_id]
        for physical_id, logical_id in zip(physical_ids, logical_ids)
    }
    return handler.with_metrics(values_by_physical_id)


def run_sequence_scorer(
    *,
    eval_name: str,
    validation_oracle: str,
    designs: str | Path,
    out: str | Path,
    per_design_out: str | Path,
    required_n: int,
    spec: SequenceSubmissionSpec,
    run_root: str | Path,
    oracle: "BlockFunction",
    gates: Mapping[str, str],
    metric_columns: Sequence[str],
    diagnostic_columns: Sequence[str] = (),
    oracle_job_args: Mapping[str, object] | None = None,
    metrics: Sequence["BlockFunction"] = (),
    metric_job_args: Mapping[str, object] | None = None,
    context: object | None = None,
    ligands: Sequence[str] = (),
    measure: Callable[[PipelineDataHandler, pl.DataFrame], Mapping[str, dict]]
    | None = None,
    soft: Sequence[str] = (),
    summary_columns: Sequence[str] | None = None,
    extra: Callable[[pd.DataFrame], Mapping[str, object]]
    | Mapping[str, object]
    | None = None,
    run_meta: str | Path | None = None,
    reinit: bool = False,
) -> ScoreReport | None:
    """Run the standard held-out scoring flow and write its outputs.

    Steps, in order: validate the submission (returning ``None`` and writing the
    contract-failure report if it fails, before any GPU); fold it with ``oracle``
    plus ``context``; run ``metrics``; attach any bespoke per-design metrics from
    ``measure``; require ``metric_columns`` finite; evaluate ``gates``; and write
    the canonical report and per-design table. ``diagnostic_columns`` are also
    written per design but are not required to be numeric or summarized; use
    them for structured/string diagnostics such as classified interaction
    vectors.

    ``gates`` maps a criterion name to a filter-expression string in the repo's
    own DSL — e.g. ``{"plddt": "holo.plddt >= 80", "aromatic": "p.pi_stacking >= 1
    or p.pi_cation >= 1"}``. Names in ``soft`` gate ``soft_pass`` only. The
    expressions double as the reported thresholds, so the gate and the published
    criterion never drift.

    ``measure(handler, submission)`` returns ``{design_id: {namespace: value}}``
    for anything the metric blocks cannot produce (a motif RMSD, a geometric
    distance); it is attached with :meth:`PipelineDataHandler.with_metrics` so
    those values become dotted metrics the gates and ``metric_columns`` address
    like any other. Use :func:`merge_submission_columns` inside it to reach
    submission data by identity. Returns the :class:`ScoreReport`, or ``None`` on
    contract failure.
    """
    from chaperone import ExecutionEngine

    designs_path = Path(designs).resolve()
    submission = load_submission_or_fail(
        designs_path,
        spec,
        eval_name=eval_name,
        validation_oracle=validation_oracle,
        out=out,
    )
    if submission is None:
        return None

    run_dir = content_addressed_rundir(designs_path, isolated_scoring_root(run_root))
    engine = ExecutionEngine(rundir=run_dir, reinit=reinit)
    inputs = sequence_submission_to_handler(
        submission,
        run_dir,
        context=context,
        oligomer_copies=spec.oligomer_copies,
        molecule_type=spec.molecule_type,
        ligands=ligands,
    )
    print(f"Loaded {len(submission)} designs; held-out run directory: {run_dir}")

    scored = engine.run(oracle, data=inputs, job_args=dict(oracle_job_args or {}))
    if metrics:
        scored = engine.run(
            list(metrics), data=scored, job_args=dict(metric_job_args or {})
        )
    if measure is not None:
        scored = attach_measurements_by_design_id(
            scored, dict(measure(scored, submission))
        )

    verdicts = evaluate_gate_expressions(scored, gates)

    frame = scored.unpack(all=True)
    # Prediction and metric blocks may return only the successful rows. A
    # partial frame must never be published as a valid score for a complete
    # submission, even if every surviving row has finite metrics.
    frame = merge_submission_columns(frame, submission, ())

    # Before any metric is trusted: did the oracle actually fold the assembly?
    # This runs ahead of require_finite_metrics because a missing chain surfaces
    # there as an opaque wall of NaNs rather than as the one fact that explains
    # them.
    n_incomplete, harness_fault = check_folded_assembly(
        frame,
        f"{getattr(oracle, 'name', '')}.path",
        _expected_assembly_chains(submission, spec, context, oracle),
    )
    criteria = [
        Criterion(
            name,
            frame["design_id"].map(verdicts[name]).astype(bool),
            hard=name not in set(soft),
        )
        for name in gates
    ]
    frame, per_criterion = evaluate_criteria(frame, criteria)

    if harness_fault:
        # Report the fault rather than the NaNs it causes. require_finite_metrics
        # would raise here and leave no score.json at all, which reads
        # downstream as "the scorer crashed" instead of "this run is void".
        print(f"ERROR: {harness_fault}")
        void = ScoreReport(
            eval=eval_name,
            n_reported=len(frame),
            required_n=required_n,
            contract_ok=True,
            validation_oracle=validation_oracle,
            per_criterion=per_criterion,
            thresholds=dict(gates),
            submission_sha256=file_sha256(designs_path),
            harness_fault=harness_fault,
            n_incomplete=n_incomplete,
        )
        write_score_outputs(
            void, frame, out=out, per_design_out=per_design_out, metric_columns=[]
        )
        return void

    require_finite_metrics(frame, list(metric_columns))

    n_reported = len(frame)
    n_pass = int(frame["pass"].sum())
    n_soft_pass = int(frame["soft_pass"].sum())
    cost, efficiency = load_cost(run_meta, n_pass)

    summary_cols = (
        list(summary_columns) if summary_columns is not None else list(metric_columns)
    )
    metric_summaries = {
        column: summarize(frame[column])
        for column in summary_cols
        if column in frame.columns
    }

    if callable(extra):
        extra_payload = dict(extra(frame))
    elif extra:
        extra_payload = dict(extra)
    else:
        extra_payload = {}

    report = ScoreReport(
        eval=eval_name,
        n_reported=n_reported,
        required_n=required_n,
        contract_ok=True,
        validation_oracle=validation_oracle,
        n_pass=n_pass,
        pass_rate=n_pass / n_reported,
        n_soft_pass=n_soft_pass,
        soft_pass_rate=n_soft_pass / n_reported,
        per_criterion=per_criterion,
        thresholds=dict(gates),
        metric_summaries=metric_summaries,
        cost=cost,
        efficiency=efficiency,
        submission_sha256=file_sha256(designs_path),
        n_incomplete=n_incomplete,
        extra=extra_payload,
    )
    write_score_outputs(
        report,
        frame,
        out=out,
        per_design_out=per_design_out,
        metric_columns=[*metric_columns, *diagnostic_columns],
    )
    return report
