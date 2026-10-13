"""PLIP protein-ligand interaction profiling block."""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import Field

from pipelines import InlineMetricBlock
from pipelines.orchestration.job_args import JobArgs

logger = logging.getLogger(__name__)

# Maps metric name → list of PLInteraction attributes to sum.
# Some interaction types are split by direction (e.g. hbonds: protein-donor
# vs ligand-donor). We sum both directions for the aggregate count.
INTERACTION_TYPES = [
    ("hydrogen_bonds", ["hbonds_pdon", "hbonds_ldon"]),
    ("hydrophobic_contacts", ["hydrophobic_contacts"]),
    ("pi_stacking", ["pistacking"]),
    ("pi_cation", ["pication_laro", "pication_paro"]),
    ("salt_bridges", ["saltbridge_lneg", "saltbridge_pneg"]),
    ("water_bridges", ["water_bridges"]),
    ("halogen_bonds", ["halogen_bonds"]),
    ("metal_complexes", ["metal_complexes"]),
]


class PLIPBlock(InlineMetricBlock):
    """Profile protein-ligand interactions using PLIP.

    Analyzes the last structure in each row for non-covalent interactions
    between protein and ligand(s). Returns counts of each interaction type
    (hydrogen bonds, hydrophobic contacts, pi-stacking, pi-cation,
    salt bridges, water bridges, halogen bonds, metal coordination).

    CIF inputs are automatically converted to PDB format via biotite.
    PLIP auto-detects ligands from HETATM records.
    """

    _name = "plip"
    _description = "Profile protein-ligand interactions using PLIP"
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu", rows_per_job=100)

    ligand_ids: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific binding sites by PLIP ID "
        "(format: 'HetID:Chain:Position', e.g. 'LIG:A:301'). "
        "When None, all detected ligands are analyzed.",
    )
    per_residue: bool = Field(
        default=False,
        description="Include per-residue total interaction counts "
        "(e.g. per_res.A42 = 3).",
    )
    include_details: bool = Field(
        default=False,
        description="Include per-ligand interaction breakdown when "
        "multiple binding sites are present.",
    )
    verbose: bool = Field(
        default=False,
        description="When True, output also includes `per_interaction_details`: "
        "for each interaction type, a list of dicts with atom-level info "
        "(ligand_atom, protein_atom, residue, distance, angle).",
    )

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            pdb_string = _get_pdb_string(loaded)
            result = _run_plip_isolated(
                pdb_string,
                ligand_ids=self.ligand_ids,
                per_residue=self.per_residue,
                include_details=self.include_details,
                verbose=self.verbose,
            )
            metrics = result["metrics"]
            if result["ligand_count"] == 0:
                logger.warning("No ligands detected by PLIP for %s", loaded.path)
            elif self.ligand_ids is not None and not result["selected_site_ids"]:
                logger.warning(
                    "None of ligand_ids=%s matched PLIP sites %s for %s",
                    self.ligand_ids,
                    result["site_ids"],
                    loaded.path,
                )

            logger.info(
                "PLIP analysis for %s: %d binding sites, %d total interactions",
                loaded.path,
                metrics.get("num_binding_sites", 0),
                metrics.get("total_interactions", 0),
            )
        except Exception as e:
            logger.error(
                "PLIP analysis failed for %s: %s",
                loaded.path,
                e,
                exc_info=True,
            )
            raise

        return metrics


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


_ISOLATED_PLIP_SCRIPT = """
import json
import sys
from pathlib import Path
from blocks.metrics.plip import _analyze_pdb_file

options = json.loads(Path(sys.argv[2]).read_text())
result = _analyze_pdb_file(Path(sys.argv[1]), **options)
Path(sys.argv[3]).write_text(json.dumps(result))
"""


def _run_plip_isolated(
    pdb_string: str,
    *,
    ligand_ids: list[str] | None,
    per_residue: bool,
    include_details: bool,
    verbose: bool,
) -> dict:
    """Run OpenBabel/PLIP away from RDKit's native libraries.

    OpenBabel and the RDKit build used by the Pipelines image can segfault when
    loaded into one process. The parent performs structure conversion; this
    short-lived child imports only PLIP/OpenBabel and returns JSON metrics.
    """
    with tempfile.TemporaryDirectory(prefix="pipelines-plip-") as temporary:
        workdir = Path(temporary)
        pdb_path = workdir / "input.pdb"
        options_path = workdir / "options.json"
        result_path = workdir / "result.json"
        pdb_path.write_text(pdb_string)
        options_path.write_text(
            json.dumps(
                {
                    "ligand_ids": ligand_ids,
                    "per_residue": per_residue,
                    "include_details": include_details,
                    "verbose": verbose,
                }
            )
        )
        environment = os.environ.copy()
        plip_pythonpath = environment.get("PIPELINES_PLIP_PYTHONPATH")
        if plip_pythonpath:
            environment["PYTHONPATH"] = plip_pythonpath + (
                os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH")
                else ""
            )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _ISOLATED_PLIP_SCRIPT,
                str(pdb_path),
                str(options_path),
                str(result_path),
            ],
            text=True,
            capture_output=True,
            env=environment,
        )
        if completed.returncode:
            detail = (completed.stdout + completed.stderr).strip()
            raise RuntimeError(
                f"isolated PLIP process failed ({completed.returncode}): {detail}"
            )
        if not result_path.is_file():
            raise RuntimeError("isolated PLIP process produced no result")
        return json.loads(result_path.read_text())


def _analyze_pdb_file(
    pdb_path: Path,
    *,
    ligand_ids: list[str] | None,
    per_residue: bool,
    include_details: bool,
    verbose: bool,
) -> dict:
    """Analyze one PDB inside the native-library-isolated child process."""
    from plip.structure.preparation import PDBComplex

    mol = PDBComplex()
    mol.load_pdb(str(pdb_path))
    mol.analyze()
    all_sites = mol.interaction_sets
    sites = all_sites
    if ligand_ids is not None:
        sites = {key: value for key, value in sites.items() if key in ligand_ids}
    metrics = (
        _aggregate_interactions(sites, per_residue, include_details, verbose)
        if mol.ligands
        else _empty_metrics()
    )
    return {
        "metrics": metrics,
        "ligand_count": len(mol.ligands),
        "site_ids": [str(key) for key in all_sites],
        "selected_site_ids": [str(key) for key in sites],
    }


def _get_pdb_string(loaded) -> str:
    """Convert a LoadedStructure to a PDB-format string for PLIP."""
    path = Path(loaded.path)
    if path.suffix == ".pdb" and path.exists():
        return path.read_text()

    # CIF or other format — convert via the shared PDB writer, which handles
    # multi-char chain_iid remapping, NaN coords, duplicate ligand atom names,
    # and over-length residue names. Operate on a copy so loaded.atom_array is
    # not mutated by the in-place renaming inside save_atomarray_to_pdb.
    import tempfile

    from blocks.utils.hbonds import save_atomarray_to_pdb

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:
        save_atomarray_to_pdb(loaded.atom_array.copy(), tmp.name)
        with open(tmp.name) as fh:
            return fh.read()


def _empty_metrics() -> dict:
    """Return a zero-valued metrics dict."""
    metrics = {name: 0 for name, _ in INTERACTION_TYPES}
    metrics["total_interactions"] = 0
    metrics["num_binding_sites"] = 0
    return metrics


def _atomname(obatom) -> str:
    """Extract PDB atom name from a pybel/OpenBabel atom wrapper. Returns '' on failure."""
    if obatom is None:
        return ""
    # PLIP atoms are pybel.Atom — get the PDB atom name via residue.GetAtomID()
    try:
        ob = obatom.OBAtom
        res = ob.GetResidue()
        if res is not None:
            name = res.GetAtomID(ob).strip()
            if name:
                return name
    except AttributeError:
        pass
    # Fallback: OBAtom.GetType() (returns element-derived type like 'O2')
    try:
        return obatom.OBAtom.GetType().strip()
    except AttributeError:
        pass
    try:
        return obatom.GetType().strip()
    except AttributeError:
        return ""


def _residue_label(reschain, resnr) -> str:
    if reschain is None or resnr is None:
        return ""
    return f"{reschain}{resnr}"


def _extract_detail(metric_name: str, ix) -> dict:
    """Build an atom-level detail dict for a single PLIP interaction."""
    base = {
        "protein_residue": _residue_label(
            getattr(ix, "reschain", None), getattr(ix, "resnr", None)
        ),
        "protein_resname": getattr(ix, "restype", None),
        "ligand_residue": _residue_label(
            getattr(ix, "reschain_l", None), getattr(ix, "resnr_l", None)
        ),
        "ligand_resname": getattr(ix, "restype_l", None),
    }
    if metric_name == "hydrogen_bonds":
        prot_is_donor = bool(getattr(ix, "protisdon", False))
        if prot_is_donor:
            base["protein_atom"] = _atomname(ix.d)
            base["ligand_atom"] = _atomname(ix.a)
            base["role"] = "protein_donor"
        else:
            base["protein_atom"] = _atomname(ix.a)
            base["ligand_atom"] = _atomname(ix.d)
            base["role"] = "ligand_donor"
        base["distance"] = float(getattr(ix, "distance_ad", 0.0))
        base["angle"] = float(getattr(ix, "angle", 0.0))
    elif metric_name == "hydrophobic_contacts":
        base["protein_atom"] = _atomname(ix.bsatom)
        base["ligand_atom"] = _atomname(ix.ligatom)
        base["distance"] = float(getattr(ix, "distance", 0.0))
    elif metric_name == "halogen_bonds":
        base["protein_atom"] = (
            _atomname(getattr(ix.acc, "o", None)) if getattr(ix, "acc", None) else ""
        )
        base["ligand_atom"] = (
            _atomname(getattr(ix.don, "x", None)) if getattr(ix, "don", None) else ""
        )
        base["distance"] = float(getattr(ix, "distance", 0.0))
        base["acc_angle"] = float(getattr(ix, "acc_angle", 0.0))
        base["don_angle"] = float(getattr(ix, "don_angle", 0.0))
    elif metric_name == "water_bridges":
        prot_is_donor = bool(getattr(ix, "protisdon", False))
        if prot_is_donor:
            base["protein_atom"] = _atomname(ix.d)
            base["ligand_atom"] = _atomname(ix.a)
            base["role"] = "protein_donor"
        else:
            base["protein_atom"] = _atomname(ix.a)
            base["ligand_atom"] = _atomname(ix.d)
            base["role"] = "ligand_donor"
        base["distance_aw"] = float(getattr(ix, "distance_aw", 0.0))
        base["distance_dw"] = float(getattr(ix, "distance_dw", 0.0))
    elif metric_name == "pi_stacking":
        base["distance"] = float(getattr(ix, "distance", 0.0))
        base["angle"] = float(getattr(ix, "angle", 0.0))
        base["type"] = getattr(ix, "type", None)
    elif metric_name == "pi_cation":
        base["distance"] = float(getattr(ix, "distance", 0.0))
        base["protein_charged"] = bool(getattr(ix, "protcharged", False))
    elif metric_name == "salt_bridges":
        base["distance"] = float(getattr(ix, "distance", 0.0))
        base["protein_positive"] = bool(getattr(ix, "protispos", False))
    elif metric_name == "metal_complexes":
        base["protein_atom"] = _atomname(getattr(ix, "target", None))
        base["distance"] = float(getattr(ix, "distance", 0.0))
        base["coordination_num"] = getattr(ix, "coordination_num", None)
        base["geometry"] = getattr(ix, "geometry", None)
    return base


def _aggregate_interactions(
    sites: dict,
    per_residue: bool,
    include_details: bool,
    verbose: bool = False,
) -> dict:
    """Aggregate interaction counts across all binding sites."""
    totals = {name: 0 for name, _ in INTERACTION_TYPES}
    per_res_counts: dict[str, int] = defaultdict(int)
    details: dict[str, list[dict]] = (
        {name: [] for name, _ in INTERACTION_TYPES} if verbose else {}
    )

    for bsid, iset in sites.items():
        site_counts = {}
        for metric_name, attrs in INTERACTION_TYPES:
            interactions = []
            for attr in attrs:
                interactions.extend(getattr(iset, attr, []))
            count = len(interactions)
            totals[metric_name] += count
            site_counts[metric_name] = count

            if per_residue:
                for ix in interactions:
                    resnr = getattr(ix, "resnr", None)
                    reschain = getattr(ix, "reschain", None)
                    if resnr is not None and reschain is not None:
                        per_res_counts[f"{reschain}{resnr}"] += 1

            if verbose:
                for ix in interactions:
                    try:
                        details[metric_name].append(_extract_detail(metric_name, ix))
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "Failed to extract %s detail for binding site %s: %s",
                            metric_name,
                            bsid,
                            e,
                        )

        if include_details:
            safe_bsid = bsid.replace(":", "_")
            for metric_name, count in site_counts.items():
                totals[f"{safe_bsid}.{metric_name}"] = count

    totals["total_interactions"] = sum(totals[name] for name, _ in INTERACTION_TYPES)
    totals["num_binding_sites"] = len(sites)

    if per_residue:
        for res_key, count in sorted(per_res_counts.items()):
            totals[f"per_res.{res_key}"] = count

    if verbose:
        totals["per_interaction_details"] = details

    return totals
