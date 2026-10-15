#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""DNA binder design pipeline.

De novo design of proteins that bind a target double-stranded DNA sequence.
The pipeline uses RFDiffusion3 to diffuse protein backbones around a fixed
DNA duplex, then uses LigandMPNN (which treats DNA as a ligand context) for
sequence design and AF3 for structure prediction and validation.

Target preparation:
  files/dna_target.cif is a full-atom B-form duplex structure derived from
  PDB 1BNA (CGCGAATTCGCG, 12 bp). For your own campaign, replace this file
  with an AF3-predicted structure for your target sequence (see README.md).

  build_rfd3_config() runs at startup to regenerate dna_binder.json from
  the constants below. Hbond conditioning (major-groove donor/acceptor
  annotations) requires a full-atom structure and is included in the JSON
  when using dna_target.cif from 1BNA.

Pipeline: rfd3 -> filter -> mpnn -> af3 -> rmsd -> filter
       -> mpnn_2 -> af3_2 -> rmsd_2 -> filter_2
"""

import json
from pathlib import Path

from blocks import Blocks
from blocks.util_blocks.filter import MetricFilter
from chaperone import ExecutionEngine, init_cli
from chaperone.data.schema import SELECTIONS


SCRIPT_DIR = Path(__file__).resolve().parent
FILES_DIR = SCRIPT_DIR / "files"

# ── DNA target definition ────────────────────────────────────────────────

# The target DNA sequence (sense strand, 5'->3').
# Default: CGCGAATTCGCG from PDB 1BNA (Dickerson-Drew dodecamer).
# The antisense strand (CGCGAATTCGCG RC = CGCGAATTCGCG) is in chain B.
DNA_SEQUENCE = "CGCGAATTCGCG"

# Protein binder length range (residues, excluding DNA).
BINDER_LENGTH_MIN = 120
BINDER_LENGTH_MAX = 150

# Major-groove hbond conditioning map: which atoms to use per DNA residue type.
# From the source notebook (raktim_dna_monomer_binder.ipynb):
#   DA → N6 (donor), N7 (acceptor)
#   DG → O6, N7 (acceptors)
#   DC → N4 (donor)
#   DT → O4 (acceptor)
HBOND_MAP = {
    "DA": {"donor": ["N6"], "acceptor": ["N7"]},
    "DG": {"donor": [], "acceptor": ["O6", "N7"]},
    "DC": {"donor": ["N4"], "acceptor": []},
    "DT": {"donor": [], "acceptor": ["O4"]},
}

# Base residue name for each single-letter code.
_DNA_RES = {"A": "DA", "C": "DC", "G": "DG", "T": "DT"}


def build_rfd3_config(sequence: str, target: Path, output_dir: Path) -> Path:
    """Validate the supplied duplex and write conditioning under the run directory."""
    import numpy as np
    from chaperone.io import parse_to_atom_array
    from biotite.structure import get_residue_starts

    sequence = sequence.upper()
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("DNA sequence must contain only A/C/G/T")
    target = target.resolve()
    atoms = parse_to_atom_array(target)
    if not np.isfinite(atoms.coord).all():
        raise ValueError("DNA target coordinates must all be finite")
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    donor, acceptor = {}, {}
    for chain, expected in (("A", sequence), ("B", reverse)):
        part = atoms[atoms.chain_id == chain]
        starts = get_residue_starts(part)
        if len(starts) != len(expected):
            raise ValueError(
                f"DNA chain {chain} does not match the requested sequence length"
            )
        if part.res_name[starts].tolist() != [
            _DNA_RES[b] for b in expected
        ] or part.res_id[starts].tolist() != list(range(1, len(expected) + 1)):
            raise ValueError(
                f"DNA chain {chain} sequence/numbering does not match the supplied target"
            )
        for i, base in enumerate(expected, 1):
            for kind, mapping in (("donor", donor), ("acceptor", acceptor)):
                names = HBOND_MAP[_DNA_RES[base]][kind]
                if not set(names) <= set(part.atom_name[part.res_id == i]):
                    raise ValueError(
                        f"DNA target lacks conditioning atoms at {chain}{i}"
                    )
                # Preserve the source recipe's sense-strand-only conditioning.
                if names and chain == "A":
                    mapping[f"{chain}{i}"] = ",".join(names)
    # Every duplex atom is a hotspot. Its centroid equals the fixed motif
    # centroid, so the hotspot outward-normal strategy divides by zero.
    # Use the supported finite motif-COM origin without changing conditioning.
    n = len(sequence)
    spec = {
        "dna_binder": {
            "input": str(target),
            "contig": f"A1-{n},/0,B1-{n},/0,{BINDER_LENGTH_MIN}-{BINDER_LENGTH_MAX}",
            "length": f"{BINDER_LENGTH_MIN+2*n}-{BINDER_LENGTH_MAX+2*n}",
            "select_hotspots": f"A1-{n},B1-{n}",
            "infer_ori_strategy": "com",
            "is_non_loopy": True,
            "select_hbond_donor": donor,
            "select_hbond_acceptor": acceptor,
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config = output_dir / "dna_binder.json"
    config.write_text(json.dumps(spec, indent=2))
    return config


# ── Block configuration ─────────────────────────────────────────────────


def rounds(config: Path, checkpoint: str | None = None):
    RFD3, MPNN, AF3, RMSD, Filter = Blocks.getall(
        "rfd3", "mpnn", "af3", "rmsd", "filter"
    )

    # Round 1: diffusion -> post-diffusion filter -> sequence design -> predict -> filter
    round_1 = [
        RFD3(
            inputs=str(config),
            **({"ckpt_path": checkpoint} if checkpoint else {}),
            n_batches=1,
            batch_size=2,
            inference_sampler={
                "step_scale": 1.5,
                "cfg_scale": 1.0,
            },
            seed_offset=42,
        ),
        Filter(
            name="rfd3_filter",
            filters={
                "rfd3.n_chainbreaks": MetricFilter(max=0),
                "rfd3.non_loop_fraction": MetricFilter(min=0.2),
            },
        ),
        MPNN(
            name="mpnn",
            num_seq_per_pdb=2,
            model_type="ligand_mpnn",
            output_mode="structures",
            args={"number_of_packs_per_design": 1},
        ),
        AF3(diffusion_batch_size=1),
        RMSD(
            name="af3_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
        ),
        Filter(
            name="round_1_filter",
            filters={
                "af3.iptm": MetricFilter(min=0.6),
                "af3.min_pae_interaction": MetricFilter(max=2.5),
                "af3_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=8.0),
            },
        ),
    ]

    # Round 2: re-sequence survivors -> tighter prediction + filter
    round_2 = [
        MPNN(
            name="mpnn_2",
            num_seq_per_pdb=2,
            model_type="ligand_mpnn",
            output_mode="structures",
            args={"number_of_packs_per_design": 1},
        ),
        AF3(name="af3_2", diffusion_batch_size=1),
        RMSD(
            name="af3_2_rmsd",
            alignment_methods=[SELECTIONS.BKBN],
            rmsd_targets=[SELECTIONS.BKBN],
        ),
        Filter(
            name="round_2_filter",
            filters={
                "af3_2.iptm": MetricFilter(min=0.7),
                "af3_2.min_pae_interaction": MetricFilter(max=1.25),
                "af3_2_rmsd.bkbn_align.bkbn_rmsd": MetricFilter(max=3.0),
            },
        ),
    ]

    return round_1, round_2


def main():
    from absl import flags

    flags.DEFINE_string(
        "dna_checkpoint",
        None,
        "Optional RFD3 checkpoint override; defaults to the public installed RFD3 model.",
    )
    flags.DEFINE_string(
        "dna_target",
        str(FILES_DIR / "dna_target.pdb"),
        "Full-atom duplex, chains A/B numbered from 1.",
    )
    flags.DEFINE_string(
        "dna_sequence",
        DNA_SEQUENCE,
        "Sense strand sequence; target coordinates must match.",
    )
    init_cli()
    checkpoint = flags.FLAGS.dna_checkpoint
    if checkpoint and not Path(checkpoint).is_file():
        raise ValueError("--dna_checkpoint must name an existing checkpoint file")
    engine = ExecutionEngine(rundir=SCRIPT_DIR / "rundir")
    config = build_rfd3_config(
        flags.FLAGS.dna_sequence, Path(flags.FLAGS.dna_target), engine.rundir / "inputs"
    )
    first, second = rounds(
        config, str(Path(checkpoint).resolve()) if checkpoint else None
    )
    result = engine.run(first)
    if len(result):
        result = engine.run(second, data=result)
    result.unpack().to_csv(engine.rundir / "results.csv", index=False)
    print(engine)


if __name__ == "__main__":
    main()
