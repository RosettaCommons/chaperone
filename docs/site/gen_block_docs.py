"""Generate block reference documentation from Pydantic models.

Run standalone:
    python docs/site/gen_block_docs.py

Or as part of a MkDocs build via the mkdocs-gen-files plugin (if configured).

Produces a single ``blocks-reference.md`` with tiered parameter tables for every
discovered block, organized by category (Networks, Metrics, Utilities).
"""

from __future__ import annotations

import sys
import inspect
from pathlib import Path
from typing import get_args, get_origin, Union
import types

# Ensure src/ is on the path so blocks and chaperone are importable
_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from blocks import Blocks  # noqa: E402
from chaperone.block_config.base import get_tier, TIER_PRIMARY  # noqa: E402
from chaperone.orchestration.job_args import JobArgs  # noqa: E402


# ---------------------------------------------------------------
# Paper / reference URLs for blocks
# ---------------------------------------------------------------
BLOCK_REFERENCES: dict[str, dict[str, str]] = {
    # Networks
    "rfoptimization": {
        "paper": "https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2",
        "title": "RFOptimization: Guiding Design Optimization with All-Atom Structure Prediction",
    },
    "proteina_complexa": {
        "paper": "https://arxiv.org/abs/2603.27950",
        "title": "Proteina-Complexa: Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute",
    },
    "rfd3": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2",
        "code": "https://github.com/RosettaCommons/foundry",
        "title": "Butcher, Krishna et al. (2025) — De novo Design of All-atom Biomolecular Interactions with RFdiffusion3",
    },
    "rfd3na": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2",
        "code": "https://github.com/RosettaCommons/foundry/tree/production/models/rfd3na",
        "title": "Butcher, Krishna et al. (2025) — RFdiffusion3NA multipolymer design",
    },
    "rfd2": {
        "code": "https://github.com/RosettaCommons/RFdiffusion2",
        "title": "RFdiffusion2",
    },
    "af3": {
        "paper": "https://doi.org/10.1038/s41586-024-07487-w",
        "code": "https://github.com/google-deepmind/alphafold3",
        "title": "Abramson et al. (2024) — Accurate structure prediction of biomolecular interactions with AlphaFold 3",
    },
    "mpnn": {
        "paper": "https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1",
        "code": "https://github.com/dauparas/LigandMPNN",
        "title": "Dauparas et al. (2023) — Atomic context-conditioned protein sequence design using LigandMPNN",
    },
    "boltz2": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1",
        "code": "https://github.com/jwohlwend/boltz",
        "title": "Passaro, Corso, Wohlwend et al. (2025) — Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction",
    },
    "boltzgen": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1",
        "code": "https://github.com/HannesStark/boltzgen",
        "title": "Stark et al. (2025) — BoltzGen: Toward Universal Binder Design",
    },
    "bindcraft": {
        "paper": "https://www.biorxiv.org/content/10.1101/2024.09.30.615802",
        "code": "https://github.com/martinpacesa/BindCraft",
        "title": "Pacesa et al. (2024) — BindCraft: one-shot design of functional protein binders",
    },
    "e1": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1",
        "code": "https://github.com/Profluent-AI/E1",
        "title": "Jain et al. (2025) — E1: Retrieval-Augmented Protein Encoder Models",
    },
    "esm2": {
        "paper": "https://doi.org/10.1126/science.ade2574",
        "code": "https://github.com/facebookresearch/esm",
        "title": "Lin et al. (2023) — Evolutionary-scale prediction of atomic-level protein structure with a language model",
    },
    "halludesign": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.11.08.686881v1",
        "code": "https://github.com/MinchaoFang/HalluDesign",
        "title": "Fang et al. (2025) — HalluDesign: Protein Optimization and de novo Design via Iterative Structure Hallucination and Sequence design",
    },
    "protein_hunter": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.10.10.681530v1",
        "code": "https://github.com/yehlincho/Protein-Hunter",
        "title": "Cho et al. (2025) — Protein Hunter: exploiting structure hallucination within diffusion for protein design",
    },
    "rf3": {
        "paper": "https://www.biorxiv.org/content/10.1101/2025.08.14.670328v1",
        "code": "https://github.com/RosettaCommons/foundry",
        "title": "Krishna et al. (2025) — Accelerating Biomolecular Modeling with AtomWorks and RF3",
    },
    "foundry_mpnn": {
        "paper": "https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1",
        "code": "https://github.com/RosettaCommons/foundry",
        "title": "Foundry Protein/LigandMPNN implementation; paper describes the underlying LigandMPNN method",
    },
    # Metrics
    "rmsd": {
        "title": "Root-mean-square deviation between aligned structures",
    },
    "peppr": {
        "code": "https://github.com/aivant/peppr",
        "title": "PePPr — Package for Evaluation of Predicted Poses",
    },
    "foldseek": {
        "paper": "https://doi.org/10.1038/s41587-023-01773-0",
        "code": "https://github.com/steineggerlab/foldseek",
        "title": "van Kempen et al. (2024) — Fast and accurate protein structure search with Foldseek",
    },
    "ligand_ddg": {
        "title": "Rosetta DDG binding energy calculation",
        "code": "https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/filter_pages/DdgFilter",
    },
}


def _categorize_block(name: str, cls) -> str:
    """Categorize a block by its module path."""
    module = cls.__module__
    if ".metrics." in module:
        return "metrics"
    if ".networks." in module:
        return "networks"
    return "utilities"


CATEGORY_LABELS = {
    "networks": (
        "Network Blocks",
        "Blocks that run neural network inference — structure prediction, design, sequence models.",
    ),
    "metrics": (
        "Metric Blocks",
        "Blocks that compute metrics on existing structures — RMSD, contacts, energetics, clustering.",
    ),
    "utilities": (
        "Utility Blocks",
        "Blocks for data manipulation — ingestion, filtering, format conversion, annotation.",
    ),
}

CATEGORY_ORDER = ["networks", "metrics", "utilities"]

# Folding oracles, newest first. Alphabetical order buries the current model
# among its predecessors and reads as if they were interchangeable, so these
# are listed by release instead — see ORACLE_GUIDANCE.
ORACLE_RECENCY = [
    "esmfold2",
    "openfold3",
    "boltz2",
    "boltz2_steered",
    "rf3",
    "protenix",
    "chai1",
    "af3",
    "af3_with_target_template",
    "colabfold",
]

ORACLE_GUIDANCE = (
    "Choose a predictor whose supported molecules, inputs, and validation evidence "
    "fit the task. Listing a block here does not establish scientific equivalence "
    "or a successful fresh-cluster installation. See [Core Examples](core-examples.md) "
    "for benchmark scope and [Validation](validation.md) for prediction comparisons."
)

# Definitions belong beside parameter tables; implementation links below expose
# exact output keys and selection behavior for every discovered block.
METRIC_NOTES = {
    "rmsd": "RMSD is the root mean square atom displacement in Å after fitting the selected alignment atoms. Output names such as `bkbn_align.motif_allatom_rmsd` identify the fit selection and the measured selection. Lower means closer geometry; compare only matched atoms and selections.",
    "rasa": "Per-atom relative solvent accessibility is computed as solvent-accessible area divided by an element-specific reference area. `ligand_rasa` is the **sum** of these fractions, not a fraction bounded by one; `mean_ligand_atom_rasa` is their mean. Smaller values indicate burial. Keep probe radius, atom groups, and backbone-only settings fixed when comparing designs.",
    "ligand_ddg": "`ddg_norepack` is Rosetta's bound-minus-separated score across jump 1, in Rosetta energy units; more negative values favor the bound state under that score function. The bundled protocol uses `beta_nov16`, minimizes before scoring, and disables repacking in the Ddg filter (`repack=0`, one repeat). It is not an experimental binding free energy or kcal/mol estimate. Check that jump 1 separates the intended partners. See the [exact protocol](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/scripts/ddg_protocol.xml) and [flags](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/scripts/flags_rescore_genpot); overriding either changes the interpretation.",
    "ligand_ddg_soft": "`ddg_soft` is the mean bound-minus-separated Rosetta score after packing with the hard beta_nov16 function and scoring with beta_nov16_soft. With at least three replicates the best and worst are dropped; `ddg_soft_std` reports the retained replicate spread. Values are Rosetta energy units, not measured binding free energies; do not compare numerically with the hard-score `ligand_ddg` protocol.",
    "ppi_ddg": "Protein–protein binding score computed with FastRelax and a bound-versus-separated ddG calculation. These are Rosetta energy units, not experimental affinities. Partner selections, relaxation, and score-function settings must match for comparisons; consult the implementation for the returned energy terms.",
    "ipsae": "Interface predicted Score from Aligned Errors summarizes inter-chain confidence from a predictor's PAE matrix using the configured distance/PAE cutoffs. Higher confidence does not prove binding. Missing PAE cannot be treated as a passing score. See the [upstream implementation](https://github.com/flyark42/ipsae).",
    "peppr": "PEPPR computes the requested structure and interface comparison metrics against a reference; there is no single universal PEPPR score. Interpret each returned metric using its upstream definition and retain the atom/chain mapping and reference used for the calculation.",
    "hbond_metrics": "HBPLUS identifies hydrogen bonds between ligand/motif and surrounding protein atoms. Outputs count the selected interactions, optionally by ligand atom group. Counts depend on the input protonation and HBPLUS geometry rules and are not hydrogen-bond energies.",
    "aromatic_neighbors": "Counts aromatic residues near selected ligand atoms under the configured distance cutoff. This is a geometric neighborhood count, not a pi-stacking energy or proof of a favorable interaction.",
    "foldseek": "Groups structures by structural similarity at the requested Foldseek thresholds. Cluster counts describe diversity under that clustering protocol, rather than a per-design correctness or activity score.",
    "plip": "PLIP classifies protein–ligand contacts into interaction types such as hydrogen bonds, hydrophobic contacts, and salt bridges. Returned counts and optional details depend on PLIP's geometric and chemical rules; they are not binding energies. See [PLIP](https://github.com/pharmai/plip).",
    "tm_score": "TM-score compares a structure with an external reference fold using a length-normalized structural similarity score. Higher values indicate greater similarity; reference choice, aligned chains, and normalization determine comparability.",
    "tm_score_rna": "RNA structural similarity against a reference, using the block's selected nucleotide anchor atoms and alignment/normalization settings. Higher TM-score means closer overall fold; it does not certify base-pair or catalytic geometry.",
    "rna_connectivity": "Detects nucleic-acid backbone breaks from the O3′ of residue i to the P atom of residue i+1 using the configured distance criterion. Missing atoms and residue mapping affect which links can be evaluated.",
    "rna_ss_f1_mcc": "F1 and Matthews correlation coefficient compare predicted/reference RNA secondary-structure pair assignments. The assigned base pairs and matching scheme define positives and negatives; inspect unresolved residues before comparing results.",
    "rna_thermo": "ViennaRNA estimates `p_motif = exp(-(G_constrained - G_unconstrained) / RT)`, the ensemble probability of the required motif base pairs under its thermodynamic model. Pseudoknots are excluded and reported with partial status; the model lacks an Mg²⁺ term. Use it to compare designs within a target, not as an absolute measure of full-complex folding.",
    "chirality_check": "Compares assigned per-atom CIP stereochemical codes with a reference ligand. A matching code requires a resolvable atom correspondence; missing or unassigned centers should not be assumed correct.",
    "secondary_structure": "Reports secondary-structure composition and loop/confidence disorder summaries globally or around a ligand pocket. Confidence interpretation requires the input B-factor field to carry the intended pLDDT values.",
    "rotamer_metrics": "Reports sidechain rotamer quality against the selected library/model. Percentile and probability fields are different quantities; compare using the same backend and residue selection.",
    "pocket_coulomb": "An approximate charge-based pocket interaction score using the configured charge and distance treatment. Its scale depends on those settings and is not an experimental affinity.",
    "metal3d": "A learned three-dimensional probability field for transition-metal binding sites. Predicted site confidence and position describe the model's output, not experimentally confirmed metal occupancy.",
    "symmetry": "Measures deviation of a homo-oligomeric assembly from cyclic point-group symmetry. Atom selection and subunit correspondence define the geometric comparison; symmetry alone does not establish stability.",
    "ligand_pocket": "Combines contact molecular surface, heavy-atom contact coverage, pocket shell/enclosure, and polar-atom neighborhood counts for the selected ligand. Surface uses Å² and distance cutoffs use Å; contact counts and coverage summarize geometry, not interaction energy.",
    "ligand_clash": "`n_clashes` counts selected protein–ligand heavy-atom pairs separated by less than `cutoff` Å; `minimum_distance` and `closest_pair` identify the nearest contact. This simple distance test does not distinguish intentional covalent bonds.",
    "clash": "Counts backbone-heavy-atom pairs (N/CA/C/O) closer than the cutoff in Å, excluding atoms in the same residue and consecutive residues of one chain. This is a geometric clash screen, not a full steric energy calculation.",
    "reference_motif_rmsd": "Motif atom RMSD in Å against the original input reference, with correspondence resolved through `diffused_index_map`. Smaller values indicate closer motif geometry; mapping coverage and atom selection matter.",
    "motif_geometry": "Reports named distances in Å and angles between selected motif atoms addressed using reference residue identifiers. Compare each quantity against the intended geometry rather than treating all distances as smaller-is-better.",
    "contact_ms": "Distance-weighted contact molecular surface in Å²: the target surface buried by the selected binder chain under the py_contact_ms algorithm. It measures interface contact extent, not binding energy.",
    "dssr_snap_metrics": "Counts protein–DNA hydrogen bonds classified as major-groove, minor-groove, or backbone contacts, plus hydrogen-bond support and SNAP interaction categories. SNAP-derived fields are missing if SNAP fails; missing values are not zero interactions. See the implementation for all 11 keys.",
    "dssr_motif_base_pairing": "Compares DSSR base-pair assignments and native pairing classifications for motif residues mapped by `diffused_index_map`. Per-residue pairing and aggregate retention diagnostics preserve the full native/predicted pair annotations; SNAP is not used.",
    "motif_confidence": "Summarizes predictor pLDDT and PAE separately for the mapped motif and scaffold. Higher pLDDT and lower PAE indicate model confidence, not experimentally validated accuracy; the upstream confidence scale and residue mapping must be preserved.",
    "mpnn_score": "Copies per-sequence `overall_confidence`, `seq_rec`, and optional `ligand_confidence` from MPNN outputs. Confidence is an upstream model ranking signal and sequence recovery measures agreement with the input sequence; neither establishes folding or binding success. Backend definitions can differ.",
    "sequence_identity_gate": "Checks identity against a reference sequence collection and reports the nearest match/gate verdict. Identity thresholds describe sequence similarity, not structural or functional novelty by themselves.",
}



def _oracle_sort_key(item: tuple[str, type]) -> tuple[int, str]:
    """Sort folding oracles by release, everything else alphabetically after."""
    name = item[0]
    if name in ORACLE_RECENCY:
        return (ORACLE_RECENCY.index(name), "")
    return (len(ORACLE_RECENCY), name)


# Blocks to exclude from documentation (test/internal blocks)
EXCLUDED_BLOCKS = {"copy", "copy2"}

# Standard fields shared across many blocks — excluded from per-block tables
# and summarized once at the bottom of the page
STANDARD_FIELDS = {"cloud", "command_template", "store"}


def _type_str(annotation) -> str:
    """Render a type annotation as a readable string."""
    if annotation is None:
        return "—"
    origin = get_origin(annotation)
    is_union = origin is Union or isinstance(annotation, types.UnionType)
    if is_union:
        args = get_args(annotation)
        parts = [_type_str(a) for a in args if a is not type(None)]
        s = " | ".join(parts)
        if type(None) in args:
            return f"Optional[{s}]" if len(parts) == 1 else f"{s} | None"
        return s
    if origin is list:
        inner = get_args(annotation)
        if inner:
            return f"list[{_type_str(inner[0])}]"
        return "list"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _default_str(field) -> str:
    """Render the default value as a readable string."""
    from pydantic_core._pydantic_core import PydanticUndefinedType

    if field.default_factory is not None:
        # Evaluating factories leaks the documentation builder's home/site paths
        # and may require optional binaries. These defaults resolve at runtime.
        return "`<computed at runtime>`"
    if isinstance(field.default, PydanticUndefinedType):
        return "**required**"
    val = field.default
    if isinstance(val, (str, Path)):
        val = str(val).replace(str(_src.parent), "<checkout>")
        val = val.replace(str(Path.home()), "~")
    if val is None:
        return "`None`"
    if isinstance(val, str) and len(val) > 60:
        return f"`{val[:57]}...`"
    return f"`{val!r}`" if isinstance(val, str) else f"`{val}`"


def _resources_str(resources: JobArgs | None) -> str:
    """Format resource requirements as a compact string."""
    if resources is None:
        return "Not specified"
    parts = []
    if resources.partition:
        parts.append(f"partition=`{resources.partition}`")
    if resources.gpu_class:
        parts.append(f"gpu_class=`{resources.gpu_class}`")
    if resources.gres:
        parts.append(f"gres=`{resources.gres}`")
    if resources.mem_gb:
        parts.append(f"mem=`{resources.mem_gb}GB`")
    if resources.time_limit:
        parts.append(f"time=`{resources.time_limit}`")
    return ", ".join(parts) if parts else "Not specified"


def _reference_lines(name: str, cls) -> list[str]:
    """Generate reference/citation lines for a block."""
    ref = BLOCK_REFERENCES.get(name, {})
    lines = []
    title = ref.get("title", "")
    paper = ref.get("paper")
    spec = getattr(cls, "_script_spec", None)
    code = getattr(spec, "source_url", None) or ref.get("code")
    if title:
        lines.append(f"> {title}")
    link_parts = []
    if paper:
        link_parts.append(f"[Paper]({paper})")
    if code:
        link_parts.append(f"[Upstream code / documentation]({code})")
    source = inspect.getsourcefile(cls)
    if source:
        relative = Path(source).resolve().relative_to(_src.parent)
        link_parts.append(
            f"[Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/{relative.as_posix()})"
        )
    if link_parts:
        lines.append(f"> {' · '.join(link_parts)}")
    lines.append("")
    return lines


def _render_block(name: str, cls) -> list[str]:
    """Render a single block's documentation section."""
    lines = []
    docstring = (cls.__doc__ or getattr(cls, "_description", "")).strip().split("\n")[0]
    resources = getattr(cls, "_resources", None)
    image = cls._get_image() if hasattr(cls, "_get_image") else None
    try:
        requires_input = cls._requires_input
    except Exception:
        requires_input = "dynamic"

    lines.append(f"### {name}")
    lines.append("")
    if docstring:
        lines.append(f"*{docstring}*")
        lines.append("")

    # Reference / paper link
    lines.extend(_reference_lines(name, cls))
    if name == "rfoptimization":
        lines.extend([
            "**Inputs and outputs:** [Configuration guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/rfoptimization/input.md). Refines an existing complex with designed protein chain A. The final MPNN redesign has incomplete side chains at mutations; pack/refold it before evaluating that sequence. Intermediate prediction confidence is retained in the upstream records, not attached as a score for the final redesigned structure.",
            "",
        ])
    if name == "proteina_complexa":
        lines.extend([
            "**Inputs and outputs:** [Binder configuration guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/proteina_complexa/input.md). Single-pass all-atom protein binder generation from a target PDB, target selection and hotspots. Outputs include target and designed binder. Run a downstream structure predictor to evaluate binding geometry and confidence; generation alone is not a binding-quality result.",
            "",
        ])
    if name == "foundry_mpnn":
        lines.extend([
            "**Validation status:** Wrapper/integration coverage does not establish scientific equivalence to the `mpnn` block (LigandMPNN). The [PPI benchmark scripts](https://github.com/RosettaCommons/chaperone/tree/main/examples/benchmarks/ppi) provide a starting point for a controlled comparison; no equivalence result is asserted here.",
            "",
        ])
    if name == "mpnn":
        lines.extend([
            "**Backend:** `mpnn` runs LigandMPNN (`dauparas/LigandMPNN`) from the `ligandmpnn` image, with every official checkpoint baked in; install it with `chaperone install ligandmpnn`. No settings are needed. `model_type` selects the model (`protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`, `per_residue_label_membrane_mpnn` or `global_label_membrane_mpnn`) and `noise_level` selects the checkpoint version; the [container guide](containers.md) lists the valid combinations.",
            "",
            "**Packed structures:** Use `output_mode=\"structures\"` and, for one structure per sampled sequence, `args={\"number_of_packs_per_design\": 1}`. The worked examples set these explicitly. The block default remains compact sequence output. With packed output and `args.repack_everything=0`, Chaperone restores observed coordinates of fixed protein residues after packing: the upstream packer can reconstruct and move these atoms even when their sequences are fixed. Newly completed atoms remain packed. Set `args.repack_everything=1` to allow fixed residues to be repacked, or `preserve_fixed_coordinates=False` to retain native packer reconstructions. Sequence fixing and coordinate preservation are distinct from subsequent refolding, which can move the structure.",
            "",
        ])
    example_notes = {
        "rpxdock": "Input geometry must use the C3 symmetry axis as its z axis and retain the asymmetric unit's radial displacement. Outputs retain motif correspondence, input lineage and each model's RPX score. Changing scoring residue selection requires checking score thresholds; the score alone does not validate a complete cage or experimental assembly.",
        "nanoparticle_prepare": "Requires RFdiffusion motif correspondence. Emits an asymmetric unit with motif metadata and preserved radial placement; rows exceeding the clash limit are removed. Invalid symmetry or missing correspondence fails explicitly.",
        "select_chains": "Missing chains raise an error. The output is a new CIF; motif residues, atom selections and diffusion correspondence are restricted to retained chains. This supports a true protein-only prediction branch when a complex also contains a protein or peptide partner, which ligand removal alone would retain.",
    }
    if name in example_notes:
        lines.extend([example_notes[name], ""])
    if name in METRIC_NOTES:
        lines.extend(["**Outputs and interpretation:** " + METRIC_NOTES[name], ""])

    # Compact metadata line
    meta_parts = [f"Class: `{cls.__qualname__}`"]
    meta_parts.append(f"Input: `{requires_input}`")
    meta_parts.append(f"Resources: {_resources_str(resources)}")
    if image:
        img_name = getattr(image, "name", "—")
        meta_parts.append(f"Container: `{img_name}`")
    lines.append(" · ".join(meta_parts))
    lines.append("")

    # Separate primary and secondary fields, excluding standard fields
    fields = [
        (fname, f)
        for fname, f in cls.model_fields.items()
        if fname != "name"
        and fname not in STANDARD_FIELDS
        and not (isinstance(f.json_schema_extra, dict) and f.json_schema_extra.get("internal"))
    ]
    primary = [(n, f) for n, f in fields if get_tier(f) == TIER_PRIMARY]
    secondary = [(n, f) for n, f in fields if get_tier(f) != TIER_PRIMARY]

    if primary or secondary:
        lines.append("| Parameter | Type | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")

        # Primary parameters first (orange-colored names)
        for fname, field in primary:
            typ = _type_str(field.annotation).replace("|", "\\|")
            default = _default_str(field)
            fdesc = (field.description or "—").replace("\n", " ").strip()
            if len(fdesc) > 120:
                fdesc = fdesc[:117] + "..."
            lines.append(
                f'| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`{fname}`</span> '
                f"| `{typ}` | {default} | {fdesc} |"
            )

        # Separator between primary and secondary
        if primary and secondary:
            lines.append(
                '| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |'
            )

        # Secondary parameters
        for fname, field in secondary:
            typ = _type_str(field.annotation).replace("|", "\\|")
            default = _default_str(field)
            fdesc = (field.description or "—").replace("\n", " ").strip()
            if len(fdesc) > 120:
                fdesc = fdesc[:117] + "..."
            lines.append(f"| `{fname}` | `{typ}` | {default} | {fdesc} |")

        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def generate_block_docs() -> str:
    """Generate markdown documentation for all discovered blocks."""
    lines = [
        "# Block Reference",
        "",
        "Auto-generated parameter reference for all pipeline blocks.",
        "Highlighted parameters are *primary* — they directly affect output count, quality, or runtime.",
        "Computed defaults resolve from settings or model logic at runtime; no documentation builder's site configuration is embedded here.",
        "",
        "Resources list what a block requests: `partition` is the class `cpu` or `gpu`, and `gpu_class` is the GPU size (`small` or `large`). Each site maps these to a real partition and `--gres` through its `CHAPERONE_*` settings; see [Slurm Site Configuration](syntax.md#slurm-site-configuration).",
        "",
        "---",
        "",
    ]

    # Group blocks by category (excluding test blocks)
    categories: dict[str, list[tuple[str, type]]] = {cat: [] for cat in CATEGORY_ORDER}
    # ``Blocks`` is lazy; iterating ``items()`` before a scan can be empty.
    # Accessing keys first makes standalone documentation generation reliable.
    for name in sorted(Blocks.keys()):
        cls = Blocks[name]
        if name in EXCLUDED_BLOCKS:
            continue
        cat = _categorize_block(name, cls)
        categories[cat].append((name, cls))

    # Oracles lead the networks section in release order; the rest stay
    # alphabetical behind them.
    categories["networks"].sort(key=_oracle_sort_key)

    # Table of contents
    for cat in CATEGORY_ORDER:
        label, _ = CATEGORY_LABELS[cat]
        blocks = categories[cat]
        if not blocks:
            continue
        lines.append(f"**{label}**\n")
        if cat == "networks":
            lines.append(f"{ORACLE_GUIDANCE}\n")
        for name, cls in blocks:
            desc = getattr(cls, "_description", "")
            ref = BLOCK_REFERENCES.get(name, {})
            paper = ref.get("paper")
            paper_link = f" [[paper]({paper})]" if paper else ""
            lines.append(f"- [{name}](#{name}) — {desc}{paper_link}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Render each category
    for cat in CATEGORY_ORDER:
        label, subtitle = CATEGORY_LABELS[cat]
        blocks = categories[cat]
        if not blocks:
            continue

        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"*{subtitle}*")
        lines.append("")

        for name, cls in blocks:
            lines.extend(_render_block(name, cls))

    # Standard parameters section (shared across many blocks)
    lines.append("## Standard Parameters")
    lines.append("")
    lines.append(
        "The following parameters are shared across most blocks and are omitted "
        "from the per-block tables above for brevity."
    )
    lines.append("")
    lines.append("| Parameter | Description |")
    lines.append("|-----------|-------------|")
    lines.append(
        "| `store` | Store structure file contents in a Delta Lake table for "
        "portable/archival runs. Default: `False`. Available on all blocks. |"
    )
    lines.append(
        "| `command_template` | Override the container/binary command template. "
        "Pass a list of strings to run against a local install instead of the "
        "default container. Available on compiled blocks. |"
    )
    lines.append(
        "| `cloud` | Set `True` to request Modal execution for a compiled block. "
        "Default: `False`; the engine backend can also select Modal. See [cloud setup](cloud.md). |"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    """Generate and write block reference docs."""
    output = Path(__file__).resolve().parent / "blocks-reference.md"
    content = generate_block_docs()
    output.write_text(content)
    print(f"Generated block reference: {output} ({len(content)} chars)")


if __name__ == "__main__":
    main()
