"""Resolve protein sequences from standard pipeline handlers.

Sequence artifacts unpack to chain-qualified columns (``sequence_A``), while
structure artifacts may carry their sequence only in their coordinates.  The
submission utility blocks need one consistent seam for both representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SequenceResolution:
    values: list[str]
    sequence_maps: list[dict[str, str]] | None = None
    sources: tuple[str, ...] = ()


def _varying(names: list[str], values: dict[str, list[str]]) -> list[str]:
    return [name for name in names if len(set(values[name])) > 1]


def resolve_sequences(
    handler, frame: pd.DataFrame, sequence_key: str
) -> SequenceResolution:
    """Resolve one comparison/submission sequence per row.

    ``sequence_key='sequence'`` is the user-facing default alias.  It accepts a
    sole chain-qualified column, concatenates the chain-qualified columns that
    vary across a multi-chain design, or derives protein sequences from the
    current structure when the handler carries no explicit sequence artifact.
    A chain can be forced with ``sequence_key='sequence_<CHAIN>'``.
    """
    if sequence_key in frame.columns:
        return SequenceResolution(
            values=frame[sequence_key].astype(str).tolist(),
            sources=(sequence_key,),
        )

    chain_columns = sorted(
        column
        for column in frame.columns
        if column.startswith("sequence_") and len(column) > len("sequence_")
    )
    requested_chain = (
        sequence_key.removeprefix("sequence_")
        if sequence_key.startswith("sequence_")
        else None
    )

    if chain_columns:
        if requested_chain is not None:
            requested_column = f"sequence_{requested_chain}"
            if requested_column not in chain_columns:
                raise ValueError(
                    f"Sequence chain {requested_chain!r} is not present; available "
                    f"chains: {[c.removeprefix('sequence_') for c in chain_columns]}"
                )
            selected = [requested_column]
        elif sequence_key == "sequence":
            values = {
                column: frame[column].astype(str).tolist() for column in chain_columns
            }
            selected = _varying(chain_columns, values) or chain_columns
        else:
            raise ValueError(
                f"Sequence column {sequence_key!r} is not present; available "
                f"sequence columns: {chain_columns}"
            )
        return SequenceResolution(
            values=[
                "".join(str(frame.iloc[index][column]) for column in selected)
                for index in range(len(frame))
            ],
            sources=tuple(selected),
        )

    if sequence_key != "sequence" and requested_chain is None:
        raise ValueError(
            f"Sequence column {sequence_key!r} is not present and is not a "
            "chain-qualified key such as 'sequence_A'."
        )

    try:
        prediction_inputs = list(handler.get_prediction_inputs(load_context=True))
    except Exception as error:
        raise ValueError(
            f"No explicit sequence columns are present and protein sequences "
            f"could not be derived from the current structures: {error}"
        ) from error

    if len(prediction_inputs) != len(frame):
        raise ValueError(
            "Sequence derivation returned a different number of rows than the "
            f"input table ({len(prediction_inputs)} != {len(frame)})."
        )
    sequence_maps = [
        {str(chain): str(sequence) for chain, sequence in item.sequences.items()}
        for item in prediction_inputs
    ]
    if not sequence_maps or any(not mapping for mapping in sequence_maps):
        raise ValueError("One or more structure-backed rows have no protein sequence.")

    chain_sets = [set(mapping) for mapping in sequence_maps]
    if any(chains != chain_sets[0] for chains in chain_sets[1:]):
        raise ValueError(
            "Structure-backed rows do not share the same protein chain IDs; "
            "set an explicit, consistently available sequence_key."
        )
    chains = sorted(chain_sets[0])
    if requested_chain is not None:
        if requested_chain not in chains:
            raise ValueError(
                f"Sequence chain {requested_chain!r} is not present; available "
                f"chains: {chains}"
            )
        selected_chains = [requested_chain]
    else:
        by_chain = {
            chain: [mapping[chain] for mapping in sequence_maps] for chain in chains
        }
        selected_chains = _varying(chains, by_chain) or chains

    return SequenceResolution(
        values=[
            "".join(mapping[chain] for chain in selected_chains)
            for mapping in sequence_maps
        ],
        sequence_maps=sequence_maps,
        sources=tuple(f"structure:{chain}" for chain in selected_chains),
    )
