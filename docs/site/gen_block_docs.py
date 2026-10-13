"""Generate block reference documentation from Pydantic models.

Run standalone:
    python docs/site/gen_block_docs.py

Or as part of a MkDocs build via the mkdocs-gen-files plugin (if configured).

Produces a single ``blocks-reference.md`` with tiered parameter tables for every
discovered block, organized by category (Networks, Metrics, Utilities).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args, get_origin, Union
import types

# Ensure src/ is on the path so blocks and pipelines are importable
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
        "paper": "https://www.science.org/doi/10.1126/science.adl2528",
        "code": "https://github.com/baker-laboratory/rf_diffusion_all_atom",
        "title": "Krishna et al. (2024) — Generalized biomolecular modeling and design with RoseTTAFold All-Atom",
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
        "title": "Dauparas et al. (2023) — LigandMPNN (Foundry variant)",
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
    "rf3_dev",
    "protenix",
    "chai1",
    "af3",
    "af3_with_target_template",
    "colabfold",
]

ORACLE_GUIDANCE = (
    "Folding oracles are listed newest first. **When in doubt, use the latest "
    "model — [esmfold2](#esmfold2).** The older oracles remain available for "
    "reproducing published protocols or when a specific model is required."
)


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
        try:
            val = field.default_factory()
            return f"`{val!r}`"
        except Exception:
            return "`<factory>`"
    if isinstance(field.default, PydanticUndefinedType):
        return "**required**"
    val = field.default
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
    if resources.gres:
        parts.append(f"gres=`{resources.gres}`")
    if resources.mem_gb:
        parts.append(f"mem=`{resources.mem_gb}GB`")
    if resources.time_limit:
        parts.append(f"time=`{resources.time_limit}`")
    return ", ".join(parts) if parts else "Not specified"


def _reference_lines(name: str) -> list[str]:
    """Generate reference/citation lines for a block."""
    ref = BLOCK_REFERENCES.get(name)
    if not ref:
        return []
    lines = []
    title = ref.get("title", "")
    paper = ref.get("paper")
    code = ref.get("code")
    if title:
        lines.append(f"> {title}")
    link_parts = []
    if paper:
        link_parts.append(f"[Paper]({paper})")
    if code:
        link_parts.append(f"[Code]({code})")
    if link_parts:
        lines.append(f"> {' · '.join(link_parts)}")
    lines.append("")
    return lines


def _render_block(name: str, cls) -> list[str]:
    """Render a single block's documentation section."""
    lines = []
    docstring = (cls.__doc__ or "").strip().split("\n")[0]
    resources = getattr(cls, "_resources", None)
    image = getattr(cls, "_image", None)
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
    lines.extend(_reference_lines(name))

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
        if fname != "name" and fname not in STANDARD_FIELDS
    ]
    primary = [(n, f) for n, f in fields if get_tier(f) == TIER_PRIMARY]
    secondary = [(n, f) for n, f in fields if get_tier(f) != TIER_PRIMARY]

    if primary or secondary:
        lines.append("| Parameter | Type | Default | Description |")
        lines.append("|-----------|------|---------|-------------|")

        # Primary parameters first (orange-colored names)
        for fname, field in primary:
            typ = _type_str(field.annotation)
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
            typ = _type_str(field.annotation)
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
        "Parameters marked with **★** are *primary* — they directly affect output count, quality, or runtime.",
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
        "| `cloud` | Cloud execution configuration for running on Modal. "
        "Default: `None` (local/SLURM). Available on compiled blocks. |"
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
