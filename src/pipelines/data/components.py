# From: https://github.com/RosettaCommons/foundry/blob/production/src/foundry/utils/components.py
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipelines.data import PipelineDataHandler

_aa_like_res_names = np.array(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "MSE",
        "SEC",
        "PYL",
        "UNK",
    ]
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Component name remapping
# ─────────────────────────────────────────────────────────────────────────────

_component_remap: dict[str, str] = {}


def _rewrite_one_cif(path: str, pattern: "re.Pattern", lookup: dict[str, str]) -> None:
    """Apply a single combined regex name-rewrite to one CIF file on disk.

    Shared core for :func:`apply_component_remap` and
    :func:`apply_atom_name_remap` (the two were ~95% identical). A
    ``archive.catcif:entry_id`` tag cannot be rewritten in place, so it is
    skipped with a warning by the caller before reaching here.
    """
    import gzip

    is_gz = path.endswith(".gz")
    opener = gzip.open if is_gz else open
    with opener(path, "rt") as f:
        lines = f.readlines()

    changed = False
    for i, line in enumerate(lines):
        new_line = pattern.sub(lambda m: lookup[m.group(0)], line)
        if new_line is not line and new_line != line:
            lines[i] = new_line
            changed = True

    if changed:
        out_path = path[:-3] if is_gz else path
        with open(out_path, "w") as f:
            f.writelines(lines)


def _partition_catcif_paths(paths: list[str]) -> tuple[list[str], list[str]]:
    """Split paths into (rewritable, catcif) lists; warn about catcif tags.

    Catcif-stored structures (``archive.catcif:entry_id``) cannot be rewritten
    in place via text replacement. They are skipped with a warning so callers
    don't raise FileNotFoundError feeding a tag to ``open()``/``gzip.open()``.
    """
    from pipelines.data.catcif import is_catcif_path

    rewritable = [p for p in paths if p and not is_catcif_path(p)]
    catcif = [p for p in paths if p and is_catcif_path(p)]
    if catcif:
        logger.warning(
            "Skipping in-place name remap for %d catcif-stored structure(s) "
            "(cannot rewrite an archive.catcif:entry_id tag in place; "
            "materialize first if a rewrite is required): %s%s",
            len(catcif),
            catcif[:3],
            " ..." if len(catcif) > 3 else "",
        )
    return rewritable, catcif


def _rewrite_cif_names(
    paths: list[str], pattern: "re.Pattern", lookup: dict[str, str]
) -> None:
    """Parallel-rewrite CIF names across ``paths`` (catcif tags skipped)."""
    from concurrent.futures import ThreadPoolExecutor

    rewritable, _ = _partition_catcif_paths(paths)
    if not rewritable:
        return
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: _rewrite_one_cif(p, pattern, lookup), rewritable))


def set_component_remap(remap: dict[str, str]) -> None:
    """Set global component name remapping.

    Useful when downstream tools rename ligands (e.g. AF3 renames CCD codes
    like "B;N" to "LIG_B"). Set this before running metric blocks that rely
    on motif/ligand metadata from upstream steps.

    Args:
        remap: Mapping from old component names to new names.
               e.g. {"B;N": "LIG_B"}

    Example::

        from pipelines.data.components import set_component_remap
        set_component_remap({"B;N": "LIG_B"})
        engine.run("rmsd", data=engine.compile())
    """
    global _component_remap
    _component_remap = dict(remap)
    logger.info("Component remap set: %s", _component_remap)


def clear_component_remap() -> None:
    """Clear the global component name remapping."""
    global _component_remap
    _component_remap = {}


def apply_component_remap(
    handler: "PipelineDataHandler",
    remap: dict[str, str],
    recompile: bool = False,
) -> None:
    """Rename residues in CIF files on disk for all structures in the handler.

    Uses fast text replacement on ATOM/HETATM lines in the CIF files rather
    than full structure parsing, making it significantly faster for large
    numbers of structures.

    Also sets the global component remap so that ``fetch_mask_from_component``
    can still fall back for structures that were not rewritten (e.g. the
    first/source structure in RMSD which may already have the correct name).

    When ``recompile`` is False (the default), the first file is sampled to
    check whether the remap has already been applied. If none of the old
    names appear, the rewrite is skipped entirely.

    Args:
        handler: PipelineDataHandler whose "last" structures will be rewritten.
        remap: Mapping from current residue name to desired name,
               e.g. ``{"LIG_B": "B;N"}``.
        recompile: If True, always rewrite files even if the remap appears
                   to have been applied already. Defaults to False.

    Example::

        from pipelines.data.components import apply_component_remap
        handler = engine.compile()
        apply_component_remap(handler, {"LIG_B": "B;N"})
        engine.run(block=Blocks.get("rmsd")(name="pd_rmsd"), data=engine.compile())
    """
    import polars as pl
    from pipelines.data.schema import Column

    set_component_remap(remap)
    if not remap:
        return

    # Single combined regex: one pass per line instead of O(patterns)
    pattern = re.compile(
        r"(?<=\s)(" + "|".join(re.escape(old) for old in remap) + r")(?=\s)"
    )
    lookup = dict(remap)

    # Polars-native path extraction (no tqdm, no LoadedStructure objects)
    paths = (
        handler._df.select(
            pl.col(str(Column.STRUCTURES)).list.last().struct.field("path")
        )
        .to_series()
        .to_list()
    )

    rewritable, _ = _partition_catcif_paths(paths)
    if not rewritable:
        return

    if not recompile:
        import gzip

        sample = rewritable[0]
        opener = gzip.open if sample.endswith(".gz") else open
        with opener(sample, "rt") as f:
            content = f.read()
        if not pattern.search(content):
            logger.info(
                "Component remap already applied (cached), skipping %d files",
                len(rewritable),
            )
            return

    _rewrite_cif_names(rewritable, pattern, lookup)

    logger.info("Applied component remap to %d structures", len(rewritable))


# ─────────────────────────────────────────────────────────────────────────────
# Atom-level name remapping (prediction oracle outputs)
# ─────────────────────────────────────────────────────────────────────────────


def apply_atom_name_remap(
    handler: "PipelineDataHandler",
    atom_remap: dict[str, str],
    residue_remap: dict[str, str] | None = None,
    recompile: bool = False,
) -> None:
    """Rename ligand atom names in CIF files on disk.

    Performs fast text replacement on ``_atom_site`` lines in the CIF files,
    replacing predicted atom names with reference atom names. Optionally
    also remaps residue names (equivalent to :func:`apply_component_remap`
    but combined in one pass for efficiency).

    The mapping is typically derived via
    :func:`blocks.networks.prediction.helpers.remap_predicted_ligand`, which
    calls this function internally.

    When ``recompile`` is False (the default), the first file is sampled to
    check whether the remap has already been applied. If none of the old
    names appear, the rewrite is skipped.

    Args:
        handler: PipelineDataHandler whose "last" structures will be rewritten.
        atom_remap: Mapping from predicted atom names to reference atom names,
            e.g. ``{"C40": "C0", "C41": "C1", ...}``.
        residue_remap: Optional mapping from predicted residue names to
            reference names, e.g. ``{"LIG1": "E;T"}``. Applied in the same
            pass as atom names. If provided, also sets the global component
            remap via :func:`set_component_remap`.
        recompile: If True, always rewrite files even if the remap appears
            to have been applied already. Defaults to False.

    Example::

        from blocks.networks.prediction.helpers import remap_predicted_ligand
        # Preferred: standalone function handles everything
        remap_predicted_ligand(handler, "ligand.cif", target_name="E;T")
    """
    import polars as pl
    from pipelines.data.schema import Column

    if residue_remap:
        set_component_remap(residue_remap)

    if not atom_remap:
        return

    # Build combined lookup: atom names + optional residue names
    lookup = dict(atom_remap)
    if residue_remap:
        lookup.update(residue_remap)

    # Sort by longest key first so "C10" is matched before "C1"
    sorted_keys = sorted(lookup.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"(?<=\s)(" + "|".join(re.escape(k) for k in sorted_keys) + r")(?=\s)"
    )

    # Extract paths from handler
    paths = (
        handler._df.select(
            pl.col(str(Column.STRUCTURES)).list.last().struct.field("path")
        )
        .to_series()
        .to_list()
    )

    rewritable, _ = _partition_catcif_paths(paths)
    if not rewritable:
        return

    if not recompile:
        import gzip

        # Use atom names unique to predicted (not present as reference names)
        # to detect whether remap was already applied.  If pred/ref names
        # overlap (e.g. pred C1→ref C23, pred C21→ref C1) a naive check
        # would see the already-correct C1 and think the remap is needed.
        # Only check atom_remap keys — residue names may persist in entity
        # tables outside atom_site lines where the regex doesn't reach.
        pred_only = set(atom_remap.keys()) - set(atom_remap.values())

        if pred_only:
            pred_only_pattern = re.compile(
                r"(?<=\s)("
                + "|".join(
                    re.escape(k) for k in sorted(pred_only, key=len, reverse=True)
                )
                + r")(?=\s)"
            )
            sample = rewritable[0]
            opener = gzip.open if sample.endswith(".gz") else open
            with opener(sample, "rt") as f:
                content = f.read()
            if not pred_only_pattern.search(content):
                logger.info(
                    "Atom name remap already applied (cached), skipping %d files",
                    len(rewritable),
                )
                return

    _rewrite_cif_names(rewritable, pattern, lookup)

    logger.info(
        "Applied atom name remap to %d structures (%d atom names%s)",
        len(rewritable),
        len(atom_remap),
        f", {len(residue_remap)} residue names" if residue_remap else "",
    )


#################################################################################
# Component / contig parsing
#################################################################################


def exists(x):
    return x is not None


TIP_BY_RESTYPE = {
    "TRP": [
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],  # fix both rings
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],  # fixed ring
    "TYR": ["CZ", "OH"],  # keeps ring dihedral flexible
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "GLN": ["CG", "CD", "OE1", "NE2"],
    "GLU": ["CG", "CD", "OE1", "OE2"],
    "CYS": ["CB", "SG"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "VAL": ["CG1", "CG2"],
    "ILE": ["CB", "CG2"],
    "MET": ["SD", "CE"],
    "LYS": ["CE", "NZ"],
    "ARG": ["CD", "NE", "CZ", "NH1", "NH2"],
    "PRO": None,
    "ALA": None,
    "GLY": None,
    "UNK": None,
    "MSK": None,
}


class ComponentValidationError(ValueError):
    """Raised when contig/component inputs cannot be parsed or validated."""

    def __init__(
        self,
        message: str,
        *,
        component: str | None = None,
        details: dict | None = None,
    ):
        self.component = component
        self.details = details or {}
        prefix = f"[component={component}] " if component else ""
        suffix = f" Details: {self.details}" if self.details else ""
        super().__init__(f"{prefix}{message}{suffix}")


def split_contig(x):
    try:
        chain = str(x[0])
        idx = x[1:]
        idx = int(idx)
        if idx < 0:
            raise ComponentValidationError(
                "Residue index must be a non-negative integer.", component=str(x)
            )
    except Exception as e:
        raise ComponentValidationError(
            f"Invalid contig format: '{x}'. Expected format is 'ChainIDResID' (e.g. 'A20').",
            component=str(x),
        ) from e
    return [chain, idx]


def extract_pn_unit_info(contig):
    """
    Convert substring like A20-21 or A20 to separate terms: A, 20, 21.
    """
    pattern = r"([A-Za-z])(\d+)(?:-(\d+))?"

    match = re.match(pattern, contig)
    if match:
        pn_unit_id = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        return pn_unit_id, start, end

    raise ComponentValidationError(
        "Invalid contig format. Expected 'ChainIDStart-Stop' or 'ChainIDIdx'.",
        component=contig,
    )


#################################################################################
# Mask getters
#################################################################################


def get_name_mask(
    source_names: np.ndarray, query_names: str, source_resname: str | None = None
):
    """
    Args:
        source_names: list of all names to match in current token
        query_string: specifier of names to get:
            "ALL" - All atom names in token are matched
            "BKBN - Only backbone atoms (not CB)
            "TIP" - 2 farthest atoms from the backbone are fixed with any
                additional atoms that automatically constrain geometries
                (e.g. 4 atoms for carboxylates/amides). See `constants.py`.
            Comma-separated string - e.g. "N,CA,C,O,CB" for exact queries
            List of names - e.g. ["N", "CA", "C", "O"] for exact queries
        source_resname: residue name is required when specifying just to grab the names for a "TIP"

    Raises error if not all exact atom names are found and unique

    Returns:
        mask of atoms corresponding to token
    """
    if isinstance(query_names, list):
        names = query_names
    elif isinstance(query_names, str):
        if query_names.upper() == "ALL":
            return np.ones(source_names.shape[0], dtype=bool)
        elif query_names.upper() == "BKBN":
            names = ["N", "CA", "C", "O"]
        elif query_names.upper() == "TIP":
            if not exists(source_resname):
                raise ComponentValidationError(
                    "TIP selection requires a residue name.",
                    component=str(source_resname),
                )
            names = TIP_BY_RESTYPE[source_resname]
            if not exists(names):
                raise ComponentValidationError(
                    "Residue does not define TIP atoms; use ALL, BKBN, or explicit names.",
                    component=str(source_resname),
                )
        elif query_names == "":
            names = []
        else:
            names = query_names.split(",")
    else:
        raise ComponentValidationError(
            "query_names must be a string or list of strings.",
            details={"got_type": str(type(query_names))},
        )

    if any(n == "" for n in names):
        raise ComponentValidationError(
            f"Empty atom name found in selection '{query_names}'.",
            component=str(source_resname),
        )
    mask = np.isin(source_names, names)

    if len(names) == 0:
        return mask

    if not len(set(names)) == len(names):
        raise ComponentValidationError(
            f"Atom names in '{query_names}' must be unique.",
            details={"duplicates": names},
        )
    if not mask.any():
        raise ComponentValidationError(
            f"Could not find requested atoms '{query_names}' in atom array.",
            details={"source_names": np.asarray(source_names).tolist()},
        )
    if mask.sum() != len(names):
        logger.warning(
            "Not all atoms found in atom array. Are you expecting multiple residues/ligands with the same names? "
            + "If not, check your input pdb file. "
            + "Atom array requested to contain names {}. Got: {}. Requested {}".format(
                query_names,
                np.asarray(source_names).tolist(),
                np.asarray(names).tolist(),
            )
        )
    if mask.sum() % len(names) != 0:
        # for the case where source_names are originated from multiple residues with the same names
        # (e.g. two ORO ligands in the input pdb: {ligand: "ORO", fixed_atoms: {ORO:"N3,C2,C4,N1"}})
        raise ComponentValidationError(
            "Number of atoms must be a multiple of the requested names.",
            details={
                "query": query_names,
                "source_names": np.asarray(source_names).tolist(),
                "requested": np.asarray(names).tolist(),
            },
        )

    return mask


def fetch_mask_from_idx(contig_str, *, atom_array):
    """
    contig_str: A11
    returns:
        mask of atoms within contig (e.g. residue 11 in chain A)
    """
    chain, res_id = split_contig(contig_str)
    mask = (atom_array.chain_id == chain) & (atom_array.res_id == res_id)
    if not np.any(mask):
        raise ComponentValidationError(
            f"Residue {chain}{res_id} not found in atom array.",
            component=f"{chain}{res_id}",
        )
    return mask


def fetch_mask_from_name(name, *, atom_array):
    """
    name: LIG_NAME
    returns:
        mask of atoms corresponding to non-protein
    """
    mask = atom_array.res_name == name
    if not np.any(mask):
        non_protein_res_names = np.unique(
            atom_array.res_name[~np.isin(atom_array.res_name, _aa_like_res_names)]
        )
        raise ComponentValidationError(
            "Component not found in input atom array.",
            component=name,
            details={"available_non_protein": non_protein_res_names.tolist()},
        )
    return mask


def fetch_mask_from_component(component, *, atom_array):
    """
    Catch-all function for fetching a component by non-protein name or contig.

    Tries the original component name first. If that fails and a remap exists
    (via :func:`set_component_remap`), tries the remapped name as a fallback.
    This allows the same remap to work across structures where the ligand has
    different names (e.g. "B;N" in RFD3 vs "LIG_B" in AF3).

    component: A11 or LIG_NAME
    returns:
        mask of atoms corresponding to component
    """
    try:
        mask = fetch_mask_from_name(component, atom_array=atom_array)
        return mask
    except ComponentValidationError:
        pass

    try:
        mask = fetch_mask_from_idx(component, atom_array=atom_array)
        return mask
    except ComponentValidationError:
        pass

    # Fallback: try remapped name
    remapped = _component_remap.get(component)
    if remapped is not None and remapped != component:
        logger.info("Trying remapped component: '%s' -> '%s'", component, remapped)
        try:
            return fetch_mask_from_name(remapped, atom_array=atom_array)
        except ComponentValidationError:
            pass
        try:
            return fetch_mask_from_idx(remapped, atom_array=atom_array)
        except ComponentValidationError:
            pass

    raise ComponentValidationError(
        "Component not found in input atom array.",
        component=component,
        details={
            "available_non_protein": np.unique(
                atom_array.res_name[~np.isin(atom_array.res_name, _aa_like_res_names)]
            ).tolist(),
            "remap": remapped,
        },
    )
