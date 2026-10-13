"""I/O utilities for pipelines."""

import contextlib
import logging
from pathlib import Path

import numpy as np
from biotite import structure as struc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_bonds(atom_array):
    """Return ``atom_array`` with a bond list, adding an empty one if missing.

    atomworks >= 3 CIF writers (``to_cif_file`` / ``to_cif_string``) read
    ``atom_array.bonds`` unconditionally, so arrays built by hand or loaded
    from bond-less formats (e.g. PDB via biotite) must carry at least an empty
    :class:`~biotite.structure.BondList`. Arrays that already have bonds are
    returned unchanged; otherwise a copy is returned and the input is untouched.
    """
    if atom_array.bonds is not None:
        return atom_array
    out = atom_array.copy()
    out.bonds = struc.BondList(out.array_length())
    return out


def _sanitize_auth_seq_id(atom_array):
    """Replace non-integer auth_seq_id values (e.g. '.') with 0 in-place."""
    if not hasattr(atom_array, "auth_seq_id"):
        return
    auth = atom_array.auth_seq_id
    if auth.dtype.kind in ("U", "O", "S"):  # string-like dtypes

        def _safe(x):
            try:
                return int(str(x))
            except (ValueError, TypeError):
                return 0

        atom_array.set_annotation(
            "auth_seq_id",
            np.array([_safe(v) for v in auth], dtype=np.int64),
        )


# (module, function) pairs through which atomworks consumes ``auth_seq_id``.
# atomworks >= 3 reads it in ``update_nonpoly_seq_ids`` (called from
# ``io._pipeline``); older releases read it in ``io_utils._add_bonds``.
_AUTH_SEQ_ID_CONSUMERS = (
    ("atomworks.io.transforms.atom_array", "update_nonpoly_seq_ids"),
    ("atomworks.io.utils.io_utils", "_add_bonds"),
)


@contextlib.contextmanager
def _sanitized_auth_seq_ids():
    """Make atomworks tolerate non-integer ``auth_seq_id`` values (e.g. ``.``).

    Wraps the first atomworks function in :data:`_AUTH_SEQ_ID_CONSUMERS` that
    exists so the annotation is sanitised just before it is parsed as an int.
    Warns (rather than silently skipping) when none of them exist, since such
    CIFs would then fall through to the ``load_any`` fallback.
    """
    import importlib

    for module_name, attr in _AUTH_SEQ_ID_CONSUMERS:
        module = importlib.import_module(module_name)
        original = getattr(module, attr, None)
        if original is not None:
            break
    else:
        logger.warning(
            "No atomworks hook found to sanitize auth_seq_id (%s); CIFs with "
            "'.' author sequence ids may fail to parse.",
            ", ".join(f"{m}.{a}" for m, a in _AUTH_SEQ_ID_CONSUMERS),
        )
        yield
        return

    def patched(atom_array, *args, **kwargs):
        _sanitize_auth_seq_id(atom_array)
        return original(atom_array, *args, **kwargs)

    setattr(module, attr, patched)
    try:
        yield
    finally:
        setattr(module, attr, original)


# 3-letter to 1-letter amino acid mapping
RES_NAME_TO_ONE_LETTER = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

# Nucleic acids are identified by residue name rather than an anchor atom:
# ``get_residue_starts()`` returns each nucleotide's first atom, which is often
# phosphate rather than C3'.  Keeping this map separate from the amino-acid map
# also prevents one-letter RNA residue names from being mistaken for proteins.
NUCLEIC_RES_NAME_TO_ONE_LETTER = {
    **{base: base for base in "ACGU"},
    **{f"D{base}": base for base in "ACGT"},
}

_AA_LIKE_RES_NAMES = np.array([*RES_NAME_TO_ONE_LETTER, "MSE", "SEC", "PYL"])
_RNA_LIKE_RES_NAMES = np.array(["A", "C", "G", "U", "I"])
_DNA_LIKE_RES_NAMES = np.array(["DA", "DC", "DG", "DT", "DI"])
_STANDARD_AA = np.array(list(RES_NAME_TO_ONE_LETTER))

# Atomworks' ChainType IntEnum values. Keeping the numeric annotation avoids
# importing Atomworks (and therefore Torch) merely to load an ordinary CIF.
_CHAIN_DNA = 3
_CHAIN_POLYPEPTIDE_L = 6
_CHAIN_RNA = 7
_CHAIN_NON_POLYMER = 8


class AssignTypes:
    """
    Assigns types to the atoms in the atom array using af3 sequence encoding scheme.
    """

    def check_input(self, data):
        assert "atom_array" in data, "Input data must contain 'atom_array'."

    def forward(self, data):
        data["atom_array"] = assign_types_(data["atom_array"])
        return data


def assign_types_(atom_array):
    res_names = atom_array.res_name
    is_protein = np.isin(res_names, _AA_LIKE_RES_NAMES)
    is_residue = np.isin(res_names, _STANDARD_AA)
    is_rna = np.isin(res_names, _RNA_LIKE_RES_NAMES)
    is_dna = np.isin(res_names, _DNA_LIKE_RES_NAMES)
    is_het = ~(is_residue | is_rna | is_dna).astype(bool)
    is_ligand = ~(is_protein | is_rna | is_dna).astype(bool)

    # Set annotations
    atom_array.set_annotation("is_protein", is_protein)
    atom_array.set_annotation("is_rna", is_rna)
    atom_array.set_annotation("is_dna", is_dna)
    atom_array.set_annotation("is_ligand", is_ligand)
    atom_array.set_annotation("is_het", is_het)
    atom_array.set_annotation("is_residue", is_residue)

    return atom_array


def get_sequences(atom_array):
    sequences = {}
    from biotite.structure import get_residue_starts

    starts = get_residue_starts(atom_array)
    tokens = atom_array[starts]

    for chain_id in np.unique(tokens.chain_id):
        tokens_in_chain = tokens[tokens.chain_id == chain_id]
        ca_mask = tokens_in_chain.atom_name == "CA"
        seq_tokens = tokens_in_chain[ca_mask]
        if len(seq_tokens) == 0:
            n_mask = tokens_in_chain.atom_name == "N"
            seq_tokens = tokens_in_chain[n_mask]
        mapping = RES_NAME_TO_ONE_LETTER
        if len(seq_tokens) == 0:
            nucleic_mask = np.isin(
                tokens_in_chain.res_name, list(NUCLEIC_RES_NAME_TO_ONE_LETTER)
            )
            seq_tokens = tokens_in_chain[nucleic_mask]
            mapping = NUCLEIC_RES_NAME_TO_ONE_LETTER
        if len(seq_tokens) == 0:
            sequences[chain_id] = None
            continue
        res_names = seq_tokens.res_name
        seq = "".join([mapping.get(res_name, "X") for res_name in res_names])
        sequences[chain_id] = seq
    return sequences


def pdb_mirror_path(pdb_id: str, pdb_mirror: str | Path) -> Path:
    """Return the canonical sharded PDB-mirror path for an entry."""
    pdb_id = pdb_id.lower()
    path = Path(pdb_mirror) / pdb_id[1:3] / f"{pdb_id}.cif.gz"
    if not path.is_file():
        raise FileNotFoundError(f"PDB entry {pdb_id!r} is not mirrored at {path}")
    return path


def clean_atom_array(atom_array):
    """Remove waters, crystallization solvents, and hydrogens."""
    from atomworks.constants import CRYSTALLIZATION_AIDS, WATER_LIKE_CCDS

    solvents = tuple(WATER_LIKE_CCDS) + tuple(CRYSTALLIZATION_AIDS)
    keep = ~np.isin(atom_array.res_name, solvents)
    keep &= np.char.upper(np.asarray(atom_array.element, dtype=str)) != "H"
    return atom_array[keep]


_clean_atom_array = clean_atom_array


def disambiguate_assembly_chain_ids(atom_array):
    """Assign unique chain IDs to repeated symmetry copies in an assembly."""
    from atomworks.io.utils.chain import create_chain_id_generator

    if "chain_iid" not in atom_array.get_annotation_categories():
        return atom_array

    pairs = list(
        dict.fromkeys(
            zip(
                (str(value) for value in atom_array.chain_id),
                (str(value) for value in atom_array.chain_iid),
                strict=True,
            )
        )
    )
    if len({chain_id for chain_id, _ in pairs}) == len(pairs):
        return atom_array

    out = atom_array.copy()
    generator = create_chain_id_generator(
        unavailable_chain_ids={str(value) for value in atom_array.chain_id}
    )
    seen: set[str] = set()
    for chain_id, chain_iid in pairs:
        if chain_id not in seen:
            seen.add(chain_id)
            continue
        mask = (atom_array.chain_id == chain_id) & (atom_array.chain_iid == chain_iid)
        out.chain_id[mask] = next(generator)
    return out


_disambiguate_assembly_chain_ids = disambiguate_assembly_chain_ids


def parse_pdb_assembly(
    pdb_id: str,
    assembly_id: str,
    pdb_mirror: str | Path,
    *,
    clean_atom_array: bool = True,
    disambiguate_assembly_chain_ids: bool = True,
):
    """Strictly load and optionally normalize one biological assembly."""
    from atomworks.io import parse
    from atomworks.io.config import ParseConfig

    result = parse(
        pdb_mirror_path(pdb_id, pdb_mirror),
        config=ParseConfig(
            build_assembly=(assembly_id,),
            convert_mse_to_met=True,
            remove_waters=False,
            remove_ccds=[],
            hydrogen_policy="keep",
        ),
    )
    if assembly_id not in result["assemblies"]:
        available = list(result["assemblies"])
        raise KeyError(
            f"assembly {assembly_id!r} is unavailable for {pdb_id}; "
            f"available assemblies: {available}"
        )
    atom_array = result["assemblies"][assembly_id][0]
    if clean_atom_array:
        atom_array = _clean_atom_array(atom_array)
    if disambiguate_assembly_chain_ids:
        atom_array = _disambiguate_assembly_chain_ids(atom_array)
    return atom_array


def parse_to_atom_array(
    p,
    remove_hydrogens=True,
    remove_nonprotein=False,
    return_stack=False,
    assign_types=False,
    **parser_kwargs,
):
    """
    Central load function used throughout this repo to handle edge cases. Ideally this wouldn't require unique kwargs but here we are.

    Args:
        cache_dir: Directory for caching parsed atom arrays. Falls back to settings.CACHE_DIR if None.
        use_cache: If True, enables caching. If None (default), uses settings.ATOM_CACHE_ENABLED.
    """
    from pipelines import get_settings

    settings = get_settings()

    # Determine if caching should be enabled
    if settings.ENABLE_ATOMWORKS_CACHE:
        parser_kwargs["save_to_cache"] = True
        parser_kwargs["cache_dir"] = settings.CACHE_DIR

    # This setting is intended for structures from the PDB and will remove motif info
    parser_kwargs.setdefault("add_missing_atoms", False)
    if parser_kwargs["add_missing_atoms"]:
        print(
            "WARNING: add_missing_atoms is True, which is not recommended. This setting is intended for structures from the PDB and will remove motif annotations."
        )

    # Catcif tag: dereference to a real CIF path before calling parse().
    # Can be removed once atomworks bundles catcif support in parse() itself.
    from pipelines.data.catcif import is_catcif_path

    if is_catcif_path(str(p)):
        import catcif_tools

        cif_str = catcif_tools.get_structure(str(p))
        import tempfile as _tempfile

        with _tempfile.NamedTemporaryFile(
            suffix=".cif", delete=False, mode="w"
        ) as _tmp:
            _tmp.write(cif_str)
            p = _tmp.name

    data = _parse_structure(p, parser_kwargs)

    if "atom_array" not in data:
        assembly_ids = list(data["assemblies"].keys())
        data["atom_array"] = data["assemblies"][assembly_ids[0]]

    parsed = data["atom_array"]
    arrays = list(parsed) if isinstance(parsed, struc.AtomArrayStack) else [parsed]
    atom_arrays = []
    for atom_array in arrays:
        if remove_hydrogens:
            atom_array = atom_array[atom_array.element != "H"]
        if remove_nonprotein:
            atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        if assign_types:
            atom_array = assign_types_(atom_array)
        if "chain_type" not in atom_array.get_annotation_categories():
            is_aa = np.isin(atom_array.res_name, _AA_LIKE_RES_NAMES)
            is_rna = np.isin(atom_array.res_name, _RNA_LIKE_RES_NAMES)
            is_dna = np.isin(atom_array.res_name, _DNA_LIKE_RES_NAMES)

            chain_type = np.full(len(atom_array), _CHAIN_NON_POLYMER)
            chain_type[is_aa] = _CHAIN_POLYPEPTIDE_L
            chain_type[is_dna] = _CHAIN_DNA
            chain_type[is_rna] = _CHAIN_RNA
            atom_array.set_annotation("chain_type", chain_type)

            if "is_polymer" not in atom_array.get_annotation_categories():
                atom_array.set_annotation("is_polymer", is_aa | is_rna | is_dna)
        atom_array = atom_array[atom_array.atom_name != "OXT"]
        atom_arrays.append(atom_array)

    if return_stack:
        return struc.stack(atom_arrays)
    else:
        return atom_arrays[0]


# Fields for the fallback parse: read the coordinates as written, with no
# repair, solvent removal or annotation. This replaces atomworks' deprecated
# load_any() and spells out its "minimal" preset so a preset change upstream
# cannot silently alter what the fallback returns.
_RAW_PARSE = {
    "add_missing_atoms": False,
    "fix_arginines": False,
    "add_id_and_entity_annotations": False,
    "long_bond_policy": "keep",
    "remove_waters": False,
    "remove_ccds": (),
}


def _parse_structure(path, parser_kwargs):
    """Use Biotite for ordinary local files; load Atomworks only as fallback."""
    path = Path(path)
    ignored = {"remove_ccds", "add_missing_atoms", "save_to_cache", "cache_dir"}
    unsupported = set(parser_kwargs) - ignored
    is_cif = path.suffix.lower() == ".cif"
    is_gzipped_cif = path.name.lower().endswith(".cif.gz")
    if not unsupported and (is_cif or is_gzipped_cif):
        from biotite.structure.io.pdbx import CIFFile, get_structure

        if is_gzipped_cif:
            import gzip

            with gzip.open(path, "rt") as handle:
                cif_file = CIFFile.read(handle)
        else:
            cif_file = CIFFile.read(path)
        return {
            "atom_array": get_structure(
                cif_file,
                model=1,
                extra_fields=["b_factor"],
                include_bonds=True,
            )
        }
    if not unsupported and path.suffix.lower() == ".pdb":
        from biotite.structure.io.pdb import PDBFile, get_structure

        return {
            "atom_array": get_structure(
                PDBFile.read(path), model=1, extra_fields=["b_factor"]
            )
        }

    from atomworks.io import parse
    from atomworks.io.config import ParseConfig

    # Built outside the try: an unknown option is a caller error, not a parse
    # failure to be papered over by the raw fallback.
    config = ParseConfig(**parser_kwargs)
    with _sanitized_auth_seq_ids():
        try:
            return parse(path, config=config)
        except Exception as exc:
            logger.warning(
                "atomworks parse() failed for %s: %s — falling back to a raw parse",
                path,
                exc,
            )
            return {
                "atom_array": parse(path, config=ParseConfig(**_RAW_PARSE))["asym_unit"]
            }
