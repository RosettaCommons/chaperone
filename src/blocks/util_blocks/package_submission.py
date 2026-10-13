"""Validate and write a sequence submission table.

Checks an exact design count, canonical residues, and uniqueness, then writes
``designs.csv`` and optional ``run_meta.json``.

The checks raise rather than warn: a table that is short, duplicated, or
carries non-canonical residues will not satisfy strict output requirements, and
failing here is cheaper than failing downstream.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from pydantic import Field

from pipelines import BlockFunction, PipelineDataHandler

from ._sequence_resolution import resolve_sequences

logger = logging.getLogger(__name__)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


class PackageSubmissionBlock(BlockFunction):
    """Write a validated ``designs.csv`` (and optional ``run_meta.json``).

    Example::

        Package = Blocks.get("package_submission")
        block = Package(output_dir="submission", expected_count=96,
                        metadata={"ligand_name": "LIG"})
    """

    _name = "package_submission"
    _display_name = "Package Submission"
    _description = "Validate and write a sequence-only design submission"
    _inline = True
    _chunked = False

    output_dir: str = Field(
        description="Directory to write designs.csv (and run_meta.json) into",
    )
    expected_count: Optional[int] = Field(
        default=None,
        description="Exact number of designs required; None accepts any count",
    )
    sequence_key: str = Field(
        default="sequence",
        description=(
            "Column holding the amino-acid sequence. The default resolves standard "
            "sequence_<CHAIN> columns and structure-backed protein sequences."
        ),
    )
    allow_duplicates: bool = Field(
        default=False,
        description="Permit repeated sequences (off: duplicates are a violation)",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Written verbatim to run_meta.json; omitted when None",
    )
    extra_columns: Optional[list[str]] = Field(
        default=None,
        description="Additional columns to carry into designs.csv alongside sequence",
    )

    def __call__(self, handler, sandbox_dir=None, **kwargs):
        frame = handler.unpack(all=True, to_pandas=True)
        resolution = resolve_sequences(handler, frame, self.sequence_key)
        sequences = resolution.values

        if self.expected_count is not None and len(sequences) != self.expected_count:
            raise ValueError(
                f"Submission must contain exactly {self.expected_count} designs, "
                f"got {len(sequences)}."
            )

        bad = {
            index: sorted(set(seq) - CANONICAL_AA)
            for index, seq in enumerate(sequences)
            if not seq or set(seq) - CANONICAL_AA
        }
        if bad:
            preview = list(bad.items())[:5]
            raise ValueError(
                f"{len(bad)} sequence(s) contain non-canonical residues or are "
                f"empty; first offenders (row: characters): {preview}"
            )

        if not self.allow_duplicates:
            seen: dict[str, int] = {}
            duplicates = []
            for index, seq in enumerate(sequences):
                if seq in seen:
                    duplicates.append((seen[seq], index))
                else:
                    seen[seq] = index
            if duplicates:
                raise ValueError(
                    f"{len(duplicates)} duplicate sequence(s) in the submission "
                    f"(row pairs: {duplicates[:5]})."
                )

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        columns = [self.sequence_key]
        for name in self.extra_columns or []:
            if name not in frame.columns:
                raise ValueError(f"extra_columns entry {name!r} is not a column")
            columns.append(name)

        designs = out / "designs.csv"
        with designs.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for index, (_, row) in enumerate(frame.iterrows()):
                writer.writerow(
                    [sequences[index]] + [row[c] for c in (self.extra_columns or [])]
                )

        if self.metadata is not None:
            (out / "run_meta.json").write_text(json.dumps(self.metadata, indent=2))

        logger.info(
            "package_submission wrote %d designs to %s", len(sequences), designs
        )
        return PipelineDataHandler(handler)
