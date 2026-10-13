"""Shared FASTA/lineage helpers for MPNN implementations."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.common import strip_name
from pipelines.data import make_sequence_row

MANIFEST_NAME = "_mpnn_input_manifest.json"


def parse_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Parse FASTA records without introducing a BioPython dependency."""
    records: list[tuple[str, str]] = []
    header = None
    sequence: list[str] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line[1:].strip()
            sequence = []
        elif header is None:
            raise ValueError(f"Malformed FASTA {path}: sequence before header")
        else:
            sequence.append(line)
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def parse_header_fields(header: str) -> dict[str, object]:
    """Parse comma-separated MPNN ``key=value`` metadata."""
    result: dict[str, object] = {"header": header}
    tokens = [token.strip() for token in header.split(",")]
    if tokens:
        result["record_name"] = tokens[0].split()[0]
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = (part.strip() for part in token.split("=", 1))
        try:
            numeric = float(value)
            result[key] = int(numeric) if numeric.is_integer() else numeric
        except ValueError:
            result[key] = value
    return result


def write_manifest(sandbox_dir: str | Path, records: dict[str, dict]) -> Path:
    path = Path(sandbox_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(sandbox_dir: str | Path) -> dict[str, dict]:
    path = Path(sandbox_dir) / MANIFEST_NAME
    if not path.is_file():
        raise RuntimeError(f"MPNN input manifest is missing: {path}")
    return json.loads(path.read_text())


def manifest_entry_for_fasta(path: str | Path, manifest: dict[str, dict]) -> dict:
    """Match a FASTA filename to the staged MPNN input record."""
    stem = Path(path).stem
    candidates = [stem, strip_name(stem)]
    for candidate in candidates:
        if candidate in manifest:
            return manifest[candidate]
    matches = [
        (len(name), record)
        for name, record in manifest.items()
        if stem == name or stem.startswith(f"{name}.") or stem.startswith(f"{name}_")
    ]
    if matches:
        return max(matches, key=lambda item: item[0])[1]
    raise ValueError(
        f"Could not match FASTA {path} to MPNN inputs {sorted(manifest)}"
    )


def split_sequence(sequence: str, record: dict, *, delimiter: str | None) -> dict[str, str]:
    """Restore chain-qualified sequences from an MPNN FASTA record."""
    chain_ids = [str(chain) for chain in record["chain_ids"]]
    if delimiter and delimiter in sequence:
        parts = sequence.split(delimiter)
    else:
        lengths = [int(length) for length in record["chain_lengths"]]
        parts = []
        offset = 0
        for length in lengths:
            parts.append(sequence[offset : offset + length])
            offset += length
        if offset != len(sequence):
            raise ValueError(
                f"MPNN sequence length {len(sequence)} does not match parent "
                f"protein-chain length {offset} for {record['parent_design_id']}"
            )
    if len(parts) != len(chain_ids):
        raise ValueError(
            f"MPNN emitted {len(parts)} chains for {record['parent_design_id']}, "
            f"expected {len(chain_ids)} ({chain_ids})"
        )
    return dict(zip(chain_ids, parts))


def sequence_row(
    *,
    block_name: str,
    parent: dict,
    header: str,
    sequence: str,
    delimiter: str | None,
    fallback_index: int,
) -> dict:
    fields = parse_header_fields(header)
    record_name = str(fields.pop("record_name", ""))
    sample = fields.get("id", fallback_index)
    if record_name and record_name != parent["input_name"]:
        design_id = record_name
    else:
        design_id = f"{parent['parent_design_id']}__{block_name}_{sample}"
    fields["source_fasta_header"] = fields.pop("header")
    fields["output_mode"] = "sequences"
    return make_sequence_row(
        design_id=design_id,
        parent_design_id=parent["parent_design_id"],
        sequences=split_sequence(sequence, parent, delimiter=delimiter),
        metrics={block_name: fields},
    )
