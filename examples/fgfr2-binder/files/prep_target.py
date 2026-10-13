#!/usr/bin/env -S bash -c '"${PIPELINES_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Fetch a complex from the PDB, keep one receptor + one ligand-protein pair, auto-detect
interface hotspots on the receptor, and write a target-only PDB + RFD3 config.

Defaults target PDB 1DJS (FGF1·FGFR2 complex):
  - Picks the largest protein chain (FGFR2) as the design target.
  - Picks the protein chain with the most contacts to it (FGF) as the natural ligand.
  - Computes target residues with sidechain atoms within --radius Å of any natural-ligand atom.
  - Writes <pdb>_complex.pdb (FGFR2 chain B + FGF chain C; for visual inspection) and
    <pdb>_target.pdb (FGFR2 only on chain B; the file RFD3 reads).
  - Writes rfd3_config.json with hotspots on chain B.

The RFD3 binder is diffused on chain A by default, so the receptor is placed on chain B
(matching the convention of examples/peptide-binder).

Run once before the FGFR2 binder pipeline. Re-run with different flags to retarget.
"""

import argparse
import json
import logging
import string
import sys
import urllib.request
from pathlib import Path

import numpy as np
from biotite.structure.io.pdb import PDBFile

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blocks.util_blocks.annotate import (  # noqa: E402
    InterfaceConfig,
    InterfaceQuery,
    _get_closest_residues_by_sidechain_dist,
)
from chaperone.io import parse_to_atom_array  # noqa: E402

logger = logging.getLogger("prep_target")


def fetch_pdb(pdb_id: str, dest: Path) -> Path:
    if dest.exists():
        logger.info("Using cached %s", dest)
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def protein_chain_lengths(atoms) -> dict[str, int]:
    """Return {chain_id: residue_count} counted by CA atoms (protein only)."""
    ca = atoms.atom_name == "CA"
    chain_ids = np.asarray(atoms.chain_id)
    lengths: dict[str, int] = {}
    for cid in np.unique(chain_ids[ca]):
        lengths[str(cid)] = int(np.sum((chain_ids == cid) & ca))
    return lengths


def count_contacts(atoms, chain_a: str, chain_b: str, radius: float) -> int:
    """Number of chain_a CA atoms within `radius` of any chain_b atom."""
    chain_ids = np.asarray(atoms.chain_id)
    a_ca = (chain_ids == chain_a) & (atoms.atom_name == "CA")
    if not a_ca.any():
        return 0
    b_mask = chain_ids == chain_b
    if not b_mask.any():
        return 0
    a_xyz = atoms.coord[a_ca]
    b_xyz = atoms.coord[b_mask]
    d2 = ((a_xyz[:, None, :] - b_xyz[None, :, :]) ** 2).sum(axis=-1)
    return int((d2.min(axis=1) < radius * radius).sum())


def pick_chains(
    atoms,
    receptor_chain: str | None,
    ligand_chain: str | None,
    contact_radius: float,
) -> tuple[str, str]:
    """Choose receptor (largest) and ligand (smallest in contact). Honor overrides."""
    lengths = protein_chain_lengths(atoms)
    if not lengths:
        raise RuntimeError("No protein chains found")
    if receptor_chain is None:
        receptor_chain = max(lengths, key=lambda c: lengths[c])
    if receptor_chain not in lengths:
        raise ValueError(f"Receptor chain {receptor_chain!r} not in {sorted(lengths)}")

    if ligand_chain is None:
        candidates = [c for c in lengths if c != receptor_chain]
        if not candidates:
            raise RuntimeError("No second protein chain to use as ligand")
        # Smaller chains are FGF-like; rank by (contacts desc, length asc).
        ranked = sorted(
            candidates,
            key=lambda c: (
                -count_contacts(atoms, c, receptor_chain, contact_radius),
                lengths[c],
            ),
        )
        ligand_chain = ranked[0]
    if ligand_chain not in lengths:
        raise ValueError(f"Ligand chain {ligand_chain!r} not in {sorted(lengths)}")

    return receptor_chain, ligand_chain


def restrict_hotspots(
    atoms,
    hotspots: list[str],
    natural_ligand_chain: str,
    contact_radius: float,
    cluster_radius: float,
    max_hotspots: int,
    anchor: str | None = None,
) -> list[str]:
    """Rank hotspots by interface depth (# of natural-ligand atoms in contact),
    optionally cluster them around an anchor (auto = top by depth), keep top N."""
    if not hotspots:
        return hotspots
    chain_ids = np.asarray(atoms.chain_id)
    res_ids = atoms.res_id
    atom_names = atoms.atom_name
    backbone = {"N", "CA", "C", "O"}
    is_sidechain = np.array([n.strip() not in backbone for n in atom_names])
    ligand_xyz = atoms.coord[chain_ids == natural_ligand_chain]
    if len(ligand_xyz) == 0:
        return hotspots[:max_hotspots] if max_hotspots > 0 else hotspots

    scored = []
    ca_coords: dict[str, np.ndarray] = {}
    for h in hotspots:
        chain, resnum = h[0], int(h[1:])
        res_mask = (chain_ids == chain) & (res_ids == resnum)
        side_mask = res_mask & is_sidechain
        if not side_mask.any():
            continue
        side_xyz = atoms.coord[side_mask]
        d2 = ((side_xyz[:, None, :] - ligand_xyz[None, :, :]) ** 2).sum(axis=-1)
        depth = int((d2.min(axis=0) < contact_radius * contact_radius).sum())
        ca_mask = res_mask & (atom_names == "CA")
        if ca_mask.any():
            ca_coords[h] = atoms.coord[ca_mask][0]
        scored.append((h, depth))
    if not scored:
        return hotspots
    scored.sort(key=lambda x: (-x[1], x[0]))

    if anchor is not None:
        if anchor not in dict(scored):
            raise ValueError(
                f"--anchor {anchor!r} is not an interface residue "
                f"(candidates: {[h for h, _ in scored]})"
            )
        anchor_depth = dict(scored)[anchor]
        logger.info("Anchor hotspot (manual): %s (depth=%d)", anchor, anchor_depth)
    else:
        anchor, anchor_depth = scored[0]
        logger.info("Anchor hotspot (auto): %s (depth=%d)", anchor, anchor_depth)
    if cluster_radius > 0 and anchor in ca_coords:
        anchor_xyz = ca_coords[anchor]
        scored = [
            (h, d)
            for h, d in scored
            if h in ca_coords
            and np.linalg.norm(ca_coords[h] - anchor_xyz) <= cluster_radius
        ]
    if max_hotspots > 0:
        scored = scored[:max_hotspots]
    kept = sorted(h for h, _ in scored)
    logger.info("After ranking + clustering: %d hotspots", len(kept))
    for h, d in scored:
        logger.info("  %s  depth=%d", h, d)
    return kept


def relabel_chains(atoms, mapping: dict[str, str]):
    """Return a new chain_id array with the given mapping applied."""
    new_ids = np.asarray(atoms.chain_id).copy()
    for old, new in mapping.items():
        new_ids[np.asarray(atoms.chain_id) == old] = new
    atoms.chain_id = new_ids
    if "chain_iid" in atoms.get_annotation_categories():
        atoms.set_annotation("chain_iid", new_ids.copy())
    return atoms


def write_pdb(atoms, dest: Path) -> None:
    pdb = PDBFile()
    atoms.b_factor = np.zeros(len(atoms))
    pdb.set_structure(atoms)
    pdb.write(str(dest))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pdb", default="1DJS", help="PDB ID to fetch (default: 1DJS)")
    p.add_argument(
        "--pdb-file", type=Path, help="Use a local PDB file instead of fetching"
    )
    p.add_argument(
        "--receptor-chain", help="Receptor chain ID; default = largest protein chain"
    )
    p.add_argument(
        "--ligand-chain",
        help="Ligand-protein chain ID; default = smallest in-contact chain",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=4.0,
        help="Hotspot contact cutoff in Å (default: 4.0)",
    )
    p.add_argument(
        "--min-radius",
        type=float,
        default=2.0,
        help="Minimum sidechain distance (default: 2.0)",
    )
    p.add_argument(
        "--max-hotspots",
        type=int,
        default=8,
        help="Keep at most this many hotspots, ranked by contact depth (default: 8). "
        "Use 0 to keep all interface residues.",
    )
    p.add_argument(
        "--cluster-radius",
        type=float,
        default=12.0,
        help="Discard hotspots whose CA is farther than this (Å) from the anchor (default: 12.0). "
        "Use 0 to disable spatial clustering.",
    )
    p.add_argument(
        "--anchor",
        default=None,
        help="Override the auto-picked cluster anchor with a specific residue ID "
        "(e.g. B284). Default: highest-depth interface residue.",
    )
    p.add_argument(
        "--binder-length",
        type=int,
        default=50,
        help="Fixed designed binder length (default: 50). Ignored if "
        "--binder-length-min/--binder-length-max are set.",
    )
    p.add_argument(
        "--binder-length-min",
        type=int,
        default=None,
        help="Lower bound of designed binder length range (default: --binder-length).",
    )
    p.add_argument(
        "--binder-length-max",
        type=int,
        default=None,
        help="Upper bound of designed binder length range (default: --binder-length).",
    )
    p.add_argument(
        "--receptor-contig",
        default=None,
        help="Override the auto-generated receptor portion of the RFD3 contig. "
        "Comma-separated ranges, e.g. 'B227-260,B163-171'. The full receptor "
        "still stays in the cleaned PDB (so AF3 can template on the whole "
        "thing); only RFD3 sees the subset.",
    )
    p.add_argument(
        "--out-prefix", default=None, help="Output basename prefix; default <pdb-lower>"
    )
    p.add_argument(
        "--config-name",
        default="fgfr2_binder",
        help="Top-level key in rfd3_config.json",
    )
    p.add_argument(
        "--no-infer-ori",
        action="store_true",
        help="Skip 'infer_ori_strategy: hotspots' in config",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    out_prefix = args.out_prefix or args.pdb.lower()
    pdb_path = args.pdb_file
    if pdb_path is None:
        pdb_path = SCRIPT_DIR / f"{args.pdb.lower()}.pdb"
        fetch_pdb(args.pdb, pdb_path)

    atoms = parse_to_atom_array(str(pdb_path), remove_hydrogens=True, assign_types=True)
    lengths = protein_chain_lengths(atoms)
    logger.info("Protein chains in %s: %s", pdb_path.name, lengths)

    receptor, ligand = pick_chains(
        atoms, args.receptor_chain, args.ligand_chain, args.radius
    )
    logger.info(
        "Selected receptor=%s (%d res), ligand=%s (%d res)",
        receptor,
        lengths[receptor],
        ligand,
        lengths[ligand],
    )

    keep_mask = np.isin(np.asarray(atoms.chain_id), [receptor, ligand])
    complex_atoms = atoms[keep_mask].copy()

    # Receptor → B (RFD3 binder is diffused on A); FGF → C for visualization.
    chain_map = {receptor: "B", ligand: "C"}
    used = (
        set(string.ascii_uppercase)
        - set(chain_map.values())
        - set(np.unique(np.asarray(complex_atoms.chain_id)))
    )
    tmp_map = {old: used.pop() for old in chain_map}
    complex_atoms = relabel_chains(complex_atoms, tmp_map)
    complex_atoms = relabel_chains(
        complex_atoms, {tmp_map[old]: new for old, new in chain_map.items()}
    )

    chain_ids_now = np.asarray(complex_atoms.chain_id)
    receptor_resids = sorted(
        int(r) for r in np.unique(complex_atoms.res_id[chain_ids_now == "B"])
    )
    if not receptor_resids:
        raise RuntimeError("Receptor chain has no residues after cleaning")
    # 1DJS has unresolved residue gaps; encode contig as the contiguous segments
    # that are actually present in the structure so RFD3 doesn't choke on them.
    receptor_segments: list[tuple[int, int]] = []
    seg_lo = receptor_resids[0]
    prev = receptor_resids[0]
    for r in receptor_resids[1:]:
        if r == prev + 1:
            prev = r
            continue
        receptor_segments.append((seg_lo, prev))
        seg_lo = prev = r
    receptor_segments.append((seg_lo, prev))
    if len(receptor_segments) > 1:
        gaps = [
            (b + 1, a - 1)
            for (_, b), (a, _) in zip(receptor_segments, receptor_segments[1:])
        ]
        logger.info(
            "Receptor has %d gaps: %s",
            len(gaps),
            ", ".join(f"B{a}-B{b}" if a != b else f"B{a}" for a, b in gaps),
        )

    complex_pdb = SCRIPT_DIR / f"{out_prefix}_complex.pdb"
    write_pdb(complex_atoms, complex_pdb)
    logger.info("Wrote %s (target B + natural ligand C, for inspection)", complex_pdb)

    target_atoms = complex_atoms[chain_ids_now == "B"].copy()
    target_pdb = SCRIPT_DIR / f"{out_prefix}_target.pdb"
    write_pdb(target_atoms, target_pdb)
    logger.info("Wrote %s (target only — used by RFD3)", target_pdb)

    hotspots = _get_closest_residues_by_sidechain_dist(
        complex_atoms,
        InterfaceConfig(
            min_contact_radius=args.min_radius,
            contact_radius=args.radius,
            query=InterfaceQuery(chain="C"),
        ),
    )
    hotspots_b = sorted(h for h in hotspots if h.startswith("B"))
    if not hotspots_b:
        raise RuntimeError(
            f"No target hotspots within {args.radius} Å of natural ligand — try a larger --radius."
        )
    logger.info(
        "Interface residues on B within %.1f Å of C: %d", args.radius, len(hotspots_b)
    )
    logger.info("  %s", ",".join(hotspots_b))

    hotspots_b = restrict_hotspots(
        complex_atoms,
        hotspots=hotspots_b,
        natural_ligand_chain="C",
        contact_radius=args.radius,
        cluster_radius=args.cluster_radius,
        max_hotspots=args.max_hotspots,
        anchor=args.anchor,
    )
    if not hotspots_b:
        raise RuntimeError(
            "All hotspots filtered out — relax --max-hotspots or --cluster-radius."
        )

    bind_lo = (
        args.binder_length_min
        if args.binder_length_min is not None
        else args.binder_length
    )
    bind_hi = (
        args.binder_length_max
        if args.binder_length_max is not None
        else args.binder_length
    )
    if bind_lo > bind_hi:
        raise ValueError(
            f"binder-length-min ({bind_lo}) > binder-length-max ({bind_hi})"
        )
    if args.receptor_contig is not None:
        # Auto-split user-supplied ranges at crystal gaps so missing residues
        # don't break RFD3.
        present = set(receptor_resids)
        user_segments: list[tuple[int, int]] = []
        for piece in args.receptor_contig.split(","):
            piece = piece.strip()
            if not piece.startswith("B") or "-" not in piece:
                raise ValueError(
                    f"--receptor-contig: '{piece}' must look like 'B<start>-<end>'"
                )
            lo_s, hi_s = piece[1:].split("-")
            # Accept either 'B241-253' or 'B241-B253'.
            if hi_s.startswith("B"):
                hi_s = hi_s[1:]
            lo, hi = int(lo_s), int(hi_s)
            requested = sorted(r for r in range(lo, hi + 1) if r in present)
            if not requested:
                raise ValueError(
                    f"--receptor-contig {piece}: no residues in this range exist in the receptor"
                )
            # Split into contiguous sub-segments
            sub_lo = requested[0]
            prev = requested[0]
            for r in requested[1:]:
                if r == prev + 1:
                    prev = r
                    continue
                user_segments.append((sub_lo, prev))
                sub_lo = prev = r
            user_segments.append((sub_lo, prev))
            missing = sorted(set(range(lo, hi + 1)) - present)
            if missing:
                logger.info(
                    "  %s: split at crystal gaps (missing %s)",
                    piece,
                    ",".join(f"B{r}" for r in missing),
                )
        receptor_contig = ",".join(f"B{a}-{b}" for a, b in user_segments)
        receptor_residue_count_for_length = sum(b - a + 1 for a, b in user_segments)
        logger.info(
            "Using user-supplied --receptor-contig (%d residues across %d segments): %s",
            receptor_residue_count_for_length,
            len(user_segments),
            receptor_contig,
        )
    else:
        receptor_contig = ",".join(f"B{a}-{b}" for a, b in receptor_segments)
        receptor_residue_count_for_length = sum(b - a + 1 for a, b in receptor_segments)
    contig = f"{bind_lo}-{bind_hi},/0,{receptor_contig}"
    # length is the TOTAL assembled length (binder + fixed motif residues).
    length = f"{bind_lo + receptor_residue_count_for_length}-{bind_hi + receptor_residue_count_for_length}"
    rfd3_config = {
        args.config_name: {
            "input": target_pdb.name,
            "contig": contig,
            "length": length,
            "select_hotspots": ",".join(hotspots_b),
        }
    }
    if not args.no_infer_ori:
        rfd3_config[args.config_name]["infer_ori_strategy"] = "hotspots"

    config_path = SCRIPT_DIR / "rfd3_config.json"
    config_path.write_text(json.dumps(rfd3_config, indent=4))
    logger.info("Wrote %s", config_path)
    logger.info("Contig: %s", contig)


if __name__ == "__main__":
    main()
