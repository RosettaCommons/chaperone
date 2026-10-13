"""DSSR nucleotide-pair and DSSR/SNAP interface metrics.

Wraps the ``x3dna-dssr`` and ``x3dna-snap`` binaries from 3DNA v2.3 to
annotate per-design protein/DNA hbond + interaction counts on AF3-style
predicted complexes.

For each row, this block:

1. Runs ``x3dna-dssr`` on the *last* structure to produce ``<base>-dssr.json``
   (DNA hbond + base-pair annotations).
2. Runs ``x3dna-snap`` on the same input to produce ``<base>-snap.txt``
   (protein-DNA interaction counts).
3. Parses both outputs into 11 metrics and attaches them under
   ``dssr_snap_metrics.*`` on the row.

The binaries are statically-linked Linux x86_64 executables; no container
is required. Bin paths are read from the ``X3DNA_DIR`` and ``X3DNA_BIN_DIR``
settings (configured in ``.env.ipd`` for IPD digs); override per-run via
``--dssr_snap_metrics.x3dna_dssr_bin=...`` if needed.

Reference parsing logic ported from
``tests/test_data/sample_notebooks/raktim_dna_monomer_binder.ipynb``
(notebook step 7: "Compute biophysical metrics" / "Create dataframe").

``dssr_motif_base_pairing`` is the motif-aware companion block. It preserves
DSSR's complete pair vector and classifications for RNA/DNA structure scoring
and never starts SNAP.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import Field

from chaperone import InlineMetricBlock, JobArgs, get_setting_field

logger = logging.getLogger(__name__)


# ── Constants (ported from notebook) ────────────────────────────────────

# Major-groove (Hoogsteen-edge) donor/acceptor atoms.
# RNA residues (A/G/C/U) mirror their DNA counterparts atom-for-atom: the base
# chemistry is identical and only the sugar differs, so without these keys every
# protein-RNA base-edge hbond would be misfiled as ``Backbone_hbond``.
MAJOR: dict[str, set[str]] = {
    "DA": {"N6", "N7"},
    "DG": {"O6", "N7"},
    "DC": {"N4"},
    "DT": {"O4"},
    "A": {"N6", "N7"},
    "G": {"O6", "N7"},
    "C": {"N4"},
    "U": {"O4"},
}

# Minor-groove (sugar-edge) donor/acceptor atoms.
# The ribose 2'-OH (``O2'``) is deliberately *not* counted here: it is a sugar
# atom, not a base edge, and is left in the backbone/``other`` bucket so
# ``Minor_hbond`` means the same thing for RNA as it does for DNA.
MINOR: dict[str, set[str]] = {
    "DA": {"N3"},
    "DG": {"N2", "N3"},
    "DC": {"O2"},
    "DT": {"O2"},
    "A": {"N3"},
    "G": {"N2", "N3"},
    "C": {"O2"},
    "U": {"O2"},
}

# 20 standard amino acid 3-letter codes (matches cifutils.constants.STANDARD_AA)
STANDARD_AA: set[str] = {
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
}

# Output column labels in the same order they appear in the
# notebook's ``output[name] = [...]`` list.
METRIC_LABELS: tuple[str, ...] = (
    "Total_hbond",
    "Major_hbond",
    "Minor_hbond",
    "Backbone_hbond",
    "hbond_support",
    "hbond_support_no_alpha",
    "hbond_support_no_backbone",
    "nt_aa",
    "bp_aa",
    "pseudo_pairs",
    "pseudo_stacks",
)


def _default_bin(name: str) -> str | None:
    """Return ``$X3DNA_BIN_DIR/<name>`` if the setting is configured, else None."""
    # Must not raise: this is a Field default_factory, so an unset setting would
    # make the block unconstructable on any machine without 3DNA installed.
    bin_dir = get_setting_field("X3DNA_BIN_DIR", None)
    return str(Path(bin_dir) / name) if bin_dir else None


# ── Parsers ─────────────────────────────────────────────────────────────


def _load_dssr_json(path: Path) -> dict:
    """Load a DSSR JSON file, with the notebook's truncated-output fallback."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: notebook observed truncated output where the trailing
        # ',"auxfiles":[' opens an unclosed array. Strip it and re-close.
        patched = text.replace(',"auxfiles":[', "}}")
        return json.loads(patched)


def parse_dssr_nt_id(identifier: str) -> dict[str, str | int]:
    """Parse a DSSR ``--idstr=long`` nucleotide identifier.

    DSSR emits identifiers such as ``1..A.G.20.`` (model, segment, chain,
    residue name, residue number).  Returning both the raw identifier and the
    compact ``A10`` spelling keeps pair records lossless while making them easy
    to join to Chaperone's ``diffused_index_map`` convention.
    """
    parts = str(identifier).rstrip(".").split(".")
    if len(parts) != 5:
        raise ValueError(
            f"Unsupported DSSR long nucleotide id {identifier!r}; expected "
            "'<model>.<segment>.<chain>.<resname>.<resnum>.'"
        )
    model, segment, chain, residue_name, residue_number = parts
    match = re.fullmatch(r"(-?\d+)(?:\^(.+))?", residue_number)
    if match is None or not chain or not residue_name:
        raise ValueError(f"Malformed DSSR long nucleotide id {identifier!r}")
    res_id = int(match.group(1))
    insertion_code = match.group(2) or ""
    canonical = {
        "DA": "A",
        "DG": "G",
        "DC": "C",
        "DT": "T",
    }.get(residue_name.upper(), residue_name.upper())
    return {
        "dssr_id": str(identifier),
        "model": model,
        "segment": segment,
        "chain": chain,
        "res_name": residue_name,
        "base": canonical,
        "res_id": res_id,
        "insertion_code": insertion_code,
        "residue_id": f"{chain}{res_id}{insertion_code}",
    }


def parse_dssr_pairs_data(data: dict) -> list[dict]:
    """Return lossless, joinable pair annotations from loaded DSSR JSON.

    Every returned record preserves DSSR's trusted ``name``, ``Saenger``,
    ``LW`` and ``DSSR`` classifications.  Records are deliberately not reduced
    to one partner per residue: DSSR can annotate one nucleotide in multiple
    pairs, and downstream metrics must see all of them.
    """
    records = []
    for item in data.get("pairs", []):
        if "nt1" not in item or "nt2" not in item:
            raise ValueError(f"DSSR pair record is missing nt1/nt2: {item!r}")
        records.append(
            {
                "index": item.get("index"),
                "nt1": parse_dssr_nt_id(item["nt1"]),
                "nt2": parse_dssr_nt_id(item["nt2"]),
                "bp": item.get("bp", ""),
                "name": item.get("name", ""),
                "Saenger": item.get("Saenger", ""),
                "LW": item.get("LW", ""),
                "DSSR": item.get("DSSR", ""),
            }
        )
    return records


def parse_dssr_pairs(path: Path) -> list[dict]:
    """Parse every base pair from a DSSR JSON file, preserving classifications."""
    return parse_dssr_pairs_data(_load_dssr_json(Path(path)))


def _dssr_env(
    x3dna_dssr_bin: str | Path,
    x3dna_home: str | Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    if x3dna_home:
        env["X3DNA"] = str(x3dna_home)
    bin_dir = str(Path(x3dna_dssr_bin).parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def run_dssr_pairs(
    input_path: str | Path,
    work_dir: str | Path,
    *,
    x3dna_dssr_bin: str | Path | None = None,
    x3dna_home: str | Path | None = None,
    base: str = "structure",
) -> list[dict]:
    """Run only ``x3dna-dssr`` and return classified base-pair records.

    This is the reusable pair-only path for RNA/DNA structure metrics.  It does
    not invoke SNAP, whose protein-nucleic-acid interaction analysis is
    unrelated to nucleotide pairing.
    """
    binary = Path(x3dna_dssr_bin or _default_bin("x3dna-dssr") or "")
    if not str(binary) or not binary.is_file():
        raise FileNotFoundError(
            "x3dna-dssr binary was not found; configure X3DNA_BIN_DIR or pass "
            "x3dna_dssr_bin"
        )
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"DSSR input structure does not exist: {input_path}")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    safe_base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    json_name = f"{safe_base}-dssr.json"
    json_out = work_dir / json_name
    # DSSR 1.7.8 can segfault when --o/--prefix contain long absolute paths.
    # The subprocess already runs inside work_dir, so keep those arguments
    # relative while retaining an absolute path for existence checks/parsing.
    prefix = safe_base
    command = [
        str(binary),
        f"--i={input_path}",
        f"--o={json_name}",
        "--json",
        "--idstr=long",
        f"--prefix={prefix}",
    ]
    env = _dssr_env(binary, x3dna_home)
    proc = subprocess.run(
        command,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        [str(binary), "--cleanup", f"--prefix={prefix}"],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not json_out.exists():
        raise RuntimeError(
            f"x3dna-dssr failed for {input_path}: rc={proc.returncode}\n"
            f"stderr: {proc.stderr[-500:]}"
        )
    return parse_dssr_pairs(json_out)


def _reverse_pair_classification(record: dict) -> dict:
    """Orient a pair classification from ``nt2 -> nt1`` instead of nt1 -> nt2."""
    oriented = dict(record)
    bp = str(oriented.get("bp", ""))
    if "-" in bp:
        left, right = bp.split("-", 1)
        oriented["bp"] = f"{right}-{left}"
    lw = str(oriented.get("LW", ""))
    if len(lw) == 3 and lw[0] in "ct":
        oriented["LW"] = f"{lw[0]}{lw[2]}{lw[1]}"
    dssr = str(oriented.get("DSSR", ""))
    match = re.fullmatch(r"([ct])(.+)-(.+)", dssr)
    if match:
        oriented["DSSR"] = f"{match.group(1)}{match.group(3)}-{match.group(2)}"
    return oriented


def dssr_pairs_by_residue(pairs: list[dict]) -> dict[str, list[dict]]:
    """Index all DSSR annotations by endpoint, retaining multiple partners."""
    indexed: dict[str, list[dict]] = {}
    classification_keys = ("bp", "name", "Saenger", "LW", "DSSR")
    for pair in pairs:
        for endpoint, partner, reverse in (
            (pair["nt1"], pair["nt2"], False),
            (pair["nt2"], pair["nt1"], True),
        ):
            classification = {key: pair.get(key, "") for key in classification_keys}
            if reverse:
                classification = _reverse_pair_classification(classification)
            annotation = {
                "pair_index": pair.get("index"),
                "partner": partner,
                **classification,
            }
            indexed.setdefault(str(endpoint["residue_id"]), []).append(annotation)
    return indexed


def motif_base_pairing_metrics(
    predicted_pairs: list[dict],
    native_pairs: list[dict],
    diffused_index_map: dict[str, str] | str,
    motif_reference_ids: list[str] | tuple[str, ...],
) -> dict:
    """Compare DSSR pair annotations for a mapped motif against its native.

    Pairing is a per-residue property: a residue counts once even when DSSR
    assigns multiple partners. Native partner type and classification matches
    accept *any* native annotation for that residue, so multiple trusted DSSR
    partners are retained rather than one being selected arbitrarily.
    """
    if isinstance(diffused_index_map, str):
        diffused_index_map = json.loads(diffused_index_map)
    index_map = {str(key): str(value) for key, value in diffused_index_map.items()}
    missing_map = [key for key in motif_reference_ids if key not in index_map]
    if missing_map:
        raise ValueError(f"diffused_index_map is missing motif residues {missing_map}")

    predicted = dssr_pairs_by_residue(predicted_pairs)
    native = dssr_pairs_by_residue(native_pairs)
    missing_native = [key for key in motif_reference_ids if not native.get(key)]
    if missing_native:
        raise ValueError(f"native DSSR annotation has no pairs for {missing_native}")

    mapped_motif = {index_map[key] for key in motif_reference_ids}
    metrics: dict[str, object] = {
        "n_pairs_total": len(predicted_pairs),
        "pair_details": {},
        "native_pair_details": {},
    }
    n_paired = 0
    n_native_type = 0
    n_in_motif = 0
    n_lw_match = 0
    n_classification_match = 0
    classification_keys = ("name", "Saenger", "LW", "DSSR")

    for reference_id in motif_reference_ids:
        design_id = index_map[reference_id]
        annotations = predicted.get(design_id, [])
        native_annotations = native[reference_id]
        native_partner_types = {
            str(annotation["partner"]["base"]) for annotation in native_annotations
        }
        paired = bool(annotations)
        native_type = any(
            str(annotation["partner"]["base"]) in native_partner_types
            for annotation in annotations
        )
        in_motif = any(
            str(annotation["partner"]["residue_id"]) in mapped_motif
            for annotation in annotations
        )
        lw_match = any(
            predicted_annotation["LW"] == native_annotation["LW"]
            and predicted_annotation["partner"]["base"]
            == native_annotation["partner"]["base"]
            for predicted_annotation in annotations
            for native_annotation in native_annotations
        )
        classification_match = any(
            predicted_annotation["partner"]["base"]
            == native_annotation["partner"]["base"]
            and all(
                predicted_annotation[key] == native_annotation[key]
                for key in classification_keys
            )
            for predicted_annotation in annotations
            for native_annotation in native_annotations
        )

        metrics[f"paired_{reference_id}"] = int(paired)
        metrics[f"n_partners_{reference_id}"] = len(annotations)
        metrics[f"partner_native_type_{reference_id}"] = int(native_type)
        metrics[f"partner_in_motif_{reference_id}"] = int(in_motif)
        metrics[f"native_lw_match_{reference_id}"] = int(lw_match)
        metrics[f"native_classification_match_{reference_id}"] = int(
            classification_match
        )
        # CSV-safe detailed vector. Keep the nested form below as well for
        # direct PipelineDataHandler consumers, while this deterministic JSON
        # scalar can be selected into a final per-design score table.
        metrics[f"pair_classifications_{reference_id}"] = json.dumps(
            [
                {
                    "partner_id": annotation["partner"]["residue_id"],
                    "partner_base": annotation["partner"]["base"],
                    "name": annotation["name"],
                    "Saenger": annotation["Saenger"],
                    "LW": annotation["LW"],
                    "DSSR": annotation["DSSR"],
                }
                for annotation in annotations
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        metrics["pair_details"][reference_id] = annotations
        metrics["native_pair_details"][reference_id] = native_annotations

        n_paired += int(paired)
        n_native_type += int(native_type)
        n_in_motif += int(in_motif)
        n_lw_match += int(lw_match)
        n_classification_match += int(classification_match)

    n_motif = len(motif_reference_ids)
    metrics.update(
        {
            "n_paired": n_paired,
            "fraction_paired": n_paired / n_motif,
            "n_partner_native_type": n_native_type,
            "n_partner_in_motif": n_in_motif,
            "n_native_lw_match": n_lw_match,
            "fraction_native_lw_match": n_lw_match / n_motif,
            "n_native_classification_match": n_classification_match,
            "fraction_native_classification_match": (n_classification_match / n_motif),
        }
    )
    return metrics


def parse_dssr_snap_outputs(
    dssr_json_path: Path,
    snap_txt_path: Path | None,
) -> dict[str, float | int]:
    """Port of notebook lines ~1750-1900: produce 11 metrics from DSSR+SNAP.

    Returns a dict keyed by ``METRIC_LABELS``. SNAP-derived metrics
    (``nt_aa``, ``bp_aa``, ``pseudo_pairs``, ``pseudo_stacks``) are NaN
    when the SNAP file is missing/unreadable, matching notebook fallback.
    """
    data = _load_dssr_json(dssr_json_path)

    # --- DSSR hbonds: groove classification ---------------------------------
    aa_hbond_partners: list[str] = []
    major = 0
    minor = 0
    other = 0
    for item in data.get("hbonds", []):
        if item.get("residue_pair") in ("nt:nt", "aa:aa"):
            continue
        id1 = item["atom1_id"]
        id2 = item["atom2_id"]

        res1 = id1.split(".")[3]
        res2 = id2.split(".")[3]

        atom1 = id1.split("@")[0]
        atom2 = id2.split("@")[0]

        # The non-AA partner identifies the DNA atom we classify.
        if res1 not in STANDARD_AA:
            atom = atom1
            res = res1
            aa_res = ".".join(id2.split(".")[2:])
        else:
            atom = atom2
            res = res2
            aa_res = ".".join(id1.split(".")[2:])

        aa_hbond_partners.append(aa_res)

        if res in MAJOR and atom in MAJOR[res]:
            major += 1
        elif res in MINOR and atom in MINOR[res]:
            minor += 1
        else:
            # Note: notebook treats "other" as the Backbone_hbond column.
            other += 1
    total = major + minor + other

    # --- SNAP-derived counts -----------------------------------------------
    if snap_txt_path is not None and snap_txt_path.exists():
        try:
            nt_aa = 0
            bp_aa = 0
            pseudo_pairs = 0
            pseudo_stacks = 0
            for line in snap_txt_path.read_text().splitlines():
                if "nucleotide/amino-acid interactions" in line:
                    nt_aa += int(line.split(" ")[2])
                if "base-pair/amino-acid interactions" in line:
                    bp_aa += int(line.split(" ")[2])
                if "base/amino-acid pseudo pairs" in line:
                    pseudo_pairs += int(line.split(" ")[2])
                if "base/amino-acid pseudo stacks" in line:
                    pseudo_stacks += int(line.split(" ")[2])
        except Exception as e:  # noqa: BLE001 - notebook parity (broad except)
            logger.warning("Failed to parse SNAP file %s: %s", snap_txt_path, e)
            nt_aa = bp_aa = pseudo_pairs = pseudo_stacks = float("nan")
    else:
        nt_aa = bp_aa = pseudo_pairs = pseudo_stacks = float("nan")

    # --- Supporting (intra-protein) hbond counts ----------------------------
    aa_hbond_partner_set = set(aa_hbond_partners)
    supporting = 0
    supporting_no_alpha = 0
    supporting_no_backbone = 0
    for item in data.get("hbonds", []):
        if item.get("residue_pair") != "aa:aa":
            continue
        id1 = item["atom1_id"]
        id2 = item["atom2_id"]
        atom1 = id1.split("@")[0]
        atom2 = id2.split("@")[0]
        try:
            res1 = int(id1.split(".")[-2])
            res2 = int(id2.split(".")[-2])
        except ValueError:
            # Some atom_ids (e.g. ligand atoms) lack a numeric residue
            # index; skip them rather than raising — matches notebook
            # tolerance (the original would have crashed but never hits
            # such records on protein-only aa:aa pairs).
            continue
        id1_short = ".".join(id1.split(".")[2:])
        id2_short = ".".join(id2.split(".")[2:])
        if id1_short in aa_hbond_partner_set or id2_short in aa_hbond_partner_set:
            supporting += 1
            is_backbone_pair = (atom1, atom2) in (("N", "O"), ("O", "N"))
            if not (is_backbone_pair and abs(res2 - res1) == 4):
                supporting_no_alpha += 1
            if not is_backbone_pair:
                supporting_no_backbone += 1

    return {
        "Total_hbond": total,
        "Major_hbond": major,
        "Minor_hbond": minor,
        "Backbone_hbond": other,  # NB: column label is Backbone, value is `other`
        "hbond_support": supporting,
        "hbond_support_no_alpha": supporting_no_alpha,
        "hbond_support_no_backbone": supporting_no_backbone,
        "nt_aa": nt_aa,
        "bp_aa": bp_aa,
        "pseudo_pairs": pseudo_pairs,
        "pseudo_stacks": pseudo_stacks,
    }


# ── Block ───────────────────────────────────────────────────────────────


class DSSRSnapMetricsBlock(InlineMetricBlock):
    """Run DSSR + SNAP on each row's last structure and attach 11 metrics.

    Adds the following keys under ``dssr_snap_metrics.*`` to every row:

    DSSR-derived (always populated):

    * ``Total_hbond``: protein-DNA hbond count (excludes nt:nt / aa:aa)
    * ``Major_hbond``: hbonds to DNA major-groove atoms
    * ``Minor_hbond``: hbonds to DNA minor-groove atoms
    * ``Backbone_hbond``: hbonds to DNA backbone (i.e. neither MAJOR nor MINOR)
    * ``hbond_support``: aa:aa hbonds where one partner already hbonds DNA
    * ``hbond_support_no_alpha``: ``hbond_support`` minus i,i+4 backbone hbonds
    * ``hbond_support_no_backbone``: ``hbond_support`` minus all backbone hbonds

    SNAP-derived (NaN if SNAP fails for that row):

    * ``nt_aa``: nucleotide / amino-acid interactions
    * ``bp_aa``: base-pair / amino-acid interactions
    * ``pseudo_pairs``: base / amino-acid pseudo pairs
    * ``pseudo_stacks``: base / amino-acid pseudo stacks

    Notes:
        * Inputs are taken from ``which="last"`` (typically AF3 predictions).
        * Binaries are static x86_64 ELF, so no container is needed; bin paths
          are configurable via Pydantic fields.
        * If the DSSR JSON cannot be produced/parsed, the row is failed —
          NaN-poisoning here usually indicates a real upstream issue
          (bad CIF, missing DNA chain, malformed atom IDs).
    """

    _name = "dssr_snap_metrics"
    _display_name = "DSSR/SNAP Metrics"
    _description = (
        "Per-design biophysical metrics for protein-DNA complexes via "
        "x3dna-dssr and x3dna-snap (3DNA v2.3)."
    )
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu")

    x3dna_dssr_bin: str | None = Field(
        default_factory=lambda: _default_bin("x3dna-dssr"),
        description=(
            "Path to the x3dna-dssr binary. Defaults to "
            "``$X3DNA_BIN_DIR/x3dna-dssr`` from the environment "
            "(set via .env.ipd on IPD digs)."
        ),
    )
    x3dna_snap_bin: str | None = Field(
        default_factory=lambda: _default_bin("x3dna-snap"),
        description=(
            "Path to the x3dna-snap binary. Defaults to "
            "``$X3DNA_BIN_DIR/x3dna-snap`` from the environment."
        ),
    )
    x3dna_home: str | None = Field(
        default_factory=lambda: get_setting_field("X3DNA_DIR", None),
        description=(
            "Value to export as $X3DNA when invoking the binaries (the "
            "3DNA install root containing config/, lib/, etc.). Defaults "
            "to the ``X3DNA_DIR`` setting."
        ),
    )
    skip_snap: bool = Field(
        default=False,
        description=(
            "If True, skip SNAP entirely (fast path) — SNAP-derived "
            "metrics will be NaN."
        ),
    )

    # ── helpers ─────────────────────────────────────────────────────────

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.x3dna_home:
            env["X3DNA"] = self.x3dna_home
        if self.x3dna_dssr_bin:
            bin_dir = str(Path(self.x3dna_dssr_bin).parent)
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        return env

    def _run_dssr(self, input_path: Path, work_dir: Path, base: str) -> Path:
        """Invoke x3dna-dssr; return the produced JSON path."""
        json_out = work_dir / f"{base}-dssr.json"
        prefix = work_dir / base
        cmd = [
            self.x3dna_dssr_bin,
            f"--i={input_path}",
            f"--o={json_out}",
            "--json",
            "--idstr=long",
            f"--prefix={prefix}",
        ]
        proc = subprocess.run(  # noqa: S603 - trusted inputs
            cmd, cwd=work_dir, env=self._env(), capture_output=True, text=True
        )
        # Always run cleanup, even on failure, to avoid leaving auxfiles
        subprocess.run(
            [self.x3dna_dssr_bin, "--cleanup", f"--prefix={prefix}"],
            cwd=work_dir,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not json_out.exists():
            raise RuntimeError(
                f"x3dna-dssr failed for {input_path}: rc={proc.returncode}\n"
                f"stderr: {proc.stderr[-500:]}"
            )
        return json_out

    def _run_snap(self, input_path: Path, work_dir: Path, base: str) -> Path | None:
        """Invoke x3dna-snap; return the produced txt path or None on failure."""
        # Mirror the reference shell script: SNAP requires the input to
        # be co-located with where it writes auxfiles, so copy it into
        # work_dir first.
        local_input = work_dir / input_path.name
        if not local_input.exists():
            shutil.copy2(input_path, local_input)

        txt_out = work_dir / f"{base}-snap.txt"
        prefix = work_dir / base
        base_cmd = [
            self.x3dna_snap_bin,
            f"--i={local_input}",
            f"--o={txt_out}",
            "--json",
            "--idstr=long",
            f"--prefix={prefix}",
        ]
        proc = subprocess.run(
            base_cmd, cwd=work_dir, env=self._env(), capture_output=True, text=True
        )
        # Cleanup pass per the reference script.
        subprocess.run(
            base_cmd + ["--cleanup"],
            cwd=work_dir,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not txt_out.exists():
            logger.warning(
                "x3dna-snap failed for %s: rc=%d stderr=%s",
                input_path,
                proc.returncode,
                proc.stderr[-500:],
            )
            return None
        return txt_out

    # ── main entry point ────────────────────────────────────────────────

    # load=False: we only need paths; structure parsing is delegated to DSSR.
    def get_data_kwargs(self):
        return {"which": "last", "load": False}

    def setup(self, handler, sandbox_dir, chunk_id):
        sandbox_dir = Path(sandbox_dir)
        work_root = sandbox_dir / "dssr_snap"
        work_root.mkdir(parents=True, exist_ok=True)

        # Sanity check binaries up front so users get a clear error before
        # we start chewing through rows.
        for attr in ("x3dna_dssr_bin", "x3dna_snap_bin"):
            bin_path = getattr(self, attr)
            if not bin_path:
                raise RuntimeError(
                    f"{self._name}: {attr} is not set. Configure X3DNA_BIN_DIR "
                    f"in .env.ipd (or .env.local), or pass --{self._name}.{attr}=...."
                )
            if not Path(bin_path).exists():
                raise FileNotFoundError(
                    f"{self._name}: required binary not found at {bin_path}. "
                    f"Check X3DNA_BIN_DIR in your env, or override via "
                    f"--{self._name}.{attr}=...."
                )

        return {"work_root": work_root}

    def compute_metrics(self, loaded, *, context=None, index=0):
        work_root = context["work_root"]
        # Resolve before use: the binaries are invoked with ``cwd=row_dir``, so a
        # relative structure path would be looked up in the wrong directory.
        input_path = Path(loaded.path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                f"{self._name}: input structure does not exist: {input_path}"
            )

        # Per-row sandbox to keep auxfiles from colliding across rows.
        base = f"{index:05d}_{input_path.stem.replace('.', '_')}"
        row_dir = work_root / base
        row_dir.mkdir(parents=True, exist_ok=True)

        try:
            dssr_json = self._run_dssr(input_path, row_dir, base)
        except Exception as e:
            logger.error("DSSR failed for %s: %s", input_path, e, exc_info=True)
            raise

        snap_txt: Path | None = None
        if not self.skip_snap:
            snap_txt = self._run_snap(input_path, row_dir, base)

        metrics = parse_dssr_snap_outputs(dssr_json, snap_txt)
        # Replace numpy NaN with Python float('nan') for JSON serializability.
        metrics = {
            k: (float("nan") if isinstance(v, float) and np.isnan(v) else v)
            for k, v in metrics.items()
        }

        logger.info(
            "DSSR/SNAP metrics for %s: total=%s major=%s minor=%s nt_aa=%s",
            input_path,
            metrics["Total_hbond"],
            metrics["Major_hbond"],
            metrics["Minor_hbond"],
            metrics["nt_aa"],
        )
        return metrics


class DSSRMotifBasePairingBlock(InlineMetricBlock):
    """DSSR pairing and native classification retention for a mapped motif.

    The block reads the motif correspondence from ``diffused_index_map``, runs
    DSSR on the native reference once and on each predicted structure, and
    reports a per-residue pairing vector plus aggregate retention diagnostics.
    Full predicted/native DSSR annotations are retained in ``pair_details`` and
    ``native_pair_details``; no residue is reduced to an arbitrary first
    partner. SNAP is neither configured nor invoked.
    """

    _name = "dssr_motif_base_pairing"
    _display_name = "DSSR motif base pairing"
    _description = "Mapped motif base pairing and native DSSR classification retention."
    _creates_new_step = False
    _resources: ClassVar[JobArgs] = JobArgs(partition="cpu")

    x3dna_dssr_bin: str | None = Field(
        default_factory=lambda: _default_bin("x3dna-dssr"),
        description=("Path to x3dna-dssr. Defaults to $X3DNA_BIN_DIR/x3dna-dssr."),
    )
    x3dna_home: str | None = Field(
        default_factory=lambda: get_setting_field("X3DNA_DIR", None),
        description="Value exported as $X3DNA while invoking x3dna-dssr.",
    )
    reference: str = Field(
        description="Native RNA/DNA structure used to derive partner annotations."
    )
    residues: list[str] = Field(
        description="Reference motif residue ids to report, e.g. ['A10', 'B4']."
    )
    continue_on_error: bool = Field(
        default=False,
        description=(
            "Return per-row status/error diagnostics instead of aborting the "
            "metric block when predicted-structure DSSR processing fails. "
            "Reference setup errors remain fatal."
        ),
    )
    which: list[int] = Field(
        default=[0, -1],
        description=(
            "Structures carrying diffused_index_map and predicted coordinates."
        ),
    )

    def get_data_kwargs(self):
        return {"which": self.which, "load": False}

    def setup(self, handler, sandbox_dir, chunk_id):
        if not self.x3dna_dssr_bin:
            raise RuntimeError(
                f"{self._name}: x3dna_dssr_bin is not set. Configure "
                "X3DNA_BIN_DIR or pass the block field explicitly."
            )
        if not Path(self.x3dna_dssr_bin).is_file():
            raise FileNotFoundError(
                f"{self._name}: required binary not found at {self.x3dna_dssr_bin}."
            )
        if not self.residues:
            raise ValueError(f"{self._name}: residues must not be empty")
        reference = Path(self.reference).resolve()
        if not reference.is_file():
            raise FileNotFoundError(f"{self._name}: reference not found: {reference}")
        work_root = Path(sandbox_dir) / "dssr_motif_base_pairing"
        work_root.mkdir(parents=True, exist_ok=True)
        native_pairs = run_dssr_pairs(
            reference,
            work_root / "reference",
            x3dna_dssr_bin=self.x3dna_dssr_bin,
            x3dna_home=self.x3dna_home,
            base=f"reference_{reference.stem}",
        )
        native_index = dssr_pairs_by_residue(native_pairs)
        missing = [
            residue for residue in self.residues if not native_index.get(residue)
        ]
        if missing:
            raise ValueError(
                f"{self._name}: reference DSSR has no pairs for motif residues "
                f"{missing}"
            )
        return {"work_root": work_root, "native_pairs": native_pairs}

    def compute_metrics(self, loaded, *, context=None, index=0):
        try:
            source, prediction = loaded
            raw_index_map = source.metadata.get("diffused_index_map")
            if not raw_index_map:
                raise ValueError(
                    f"{self._name}: {source.path} has no diffused_index_map metadata"
                )
            input_path = Path(prediction.path).resolve()
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"{self._name}: input structure does not exist: {input_path}"
                )
            base = f"{index:05d}_{input_path.stem.replace('.', '_')}"
            pairs = run_dssr_pairs(
                input_path,
                context["work_root"] / base,
                x3dna_dssr_bin=self.x3dna_dssr_bin,
                x3dna_home=self.x3dna_home,
                base=base,
            )
            logger.info("DSSR found %d nucleotide pairs for %s", len(pairs), input_path)
            metrics = motif_base_pairing_metrics(
                pairs,
                context["native_pairs"],
                raw_index_map,
                self.residues,
            )
        except Exception as exc:
            if not self.continue_on_error:
                raise
            logger.exception("DSSR motif pairing failed for row %d: %s", index, exc)
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        return {
            **metrics,
            "status": "ok",
            "error_type": "",
            "error": "",
        }
