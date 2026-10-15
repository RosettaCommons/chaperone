"""Rosetta DDG (ligand binding energy) block.

Runs rosetta_scripts with a DDG protocol to compute
the ddg_norepack metric for protein-ligand complexes.

Ligand identity and residue number are auto-detected from the
structure's metadata and atom array. Override fields are available
for explicit control.
"""

import logging
import os
import shlex
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import Field

from chaperone import CompiledBlockFunction, JobArgs, PipelineDataHandler
from chaperone.data.schema import make_row
from chaperone import get_setting_field

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_DEFAULT_PROTOCOL = str(_SCRIPTS_DIR / "ddg_protocol.xml")
_DEFAULT_FLAGS = str(_SCRIPTS_DIR / "flags_rescore_genpot")


def _detect_ligand_res_number(atom_array, ligand_resname: str) -> int | None:
    """Find the residue number for a ligand by its res_name in the atom array.

    Returns the res_id of the first atom matching ``ligand_resname``,
    or ``None`` if no match is found.
    """
    mask = atom_array.res_name == ligand_resname
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return None
    return int(atom_array.res_id[indices[0]])


class LigandDDGBlock(CompiledBlockFunction):
    """Compute Rosetta DDG (binding energy) for protein-ligand complexes.

    Runs rosetta_scripts with a DDG protocol on each input structure
    and extracts the ddg_norepack metric from the output score files.
    CIF inputs are automatically converted to PDB in create_command.

    Auto-detection (per structure):
    - **ligand name**: read from ``metadata["ligand"]``; first entry is used.
    - **ligand_res_number**: detected by finding atoms matching the ligand's
      ``res_name`` in the atom array.
    - **param_file**: resolved via ``blocks.utils.params.resolve_param_file``.

    Use ``override_*`` fields to bypass auto-detection.
    """

    _name: ClassVar[str] = "ligand_ddg"
    _display_name: ClassVar[str] = "Ligand DDG"
    _description: ClassVar[str] = "Compute Rosetta DDG binding energy"
    _resources: ClassVar[JobArgs] = JobArgs(
        partition="cpu",
        mem_gb=8,
        rows_per_job=10,
    )

    def get_command_template(self) -> list[str]:
        return [self.rosetta_bin]

    rosetta_bin: str = Field(
        default_factory=lambda: get_setting_field("ROSETTA_BIN", check_bin=True),
        description="Path to rosetta_scripts binary",
    )
    protocol: str = Field(
        default=_DEFAULT_PROTOCOL,
        description="Path to Rosetta XML protocol file",
    )
    flags: str = Field(
        default=_DEFAULT_FLAGS,
        description="Path to Rosetta flags file",
    )
    dalphaball_bin: str | None = Field(
        default_factory=lambda: get_setting_field("DALPHABALL_BIN", default=None),
        description="DAlphaBall executable required by the bundled Holes filter; also searches PATH.",
    )
    override_param_file: str | None = Field(
        default=None,
        description="Override path to ligand .params file for -extra_res. "
        "Bypasses auto-resolution from ligand name.",
    )
    override_ligand_names: list[str] | None = Field(
        default=None,
        description="Override ligand 3-letter codes (e.g. ['FAD', 'IAI']). "
        "Bypasses auto-detection from structure metadata. "
        "The first matching name found in each structure's atom array is used.",
    )
    ligand_map: dict[str, str] | None = Field(
        default=None,
        description="Rename ligand residues in the atom array before detection "
        "and PDB writing. Maps old res_name to new res_name, e.g. "
        "{'LIG': 'FAD'}. Applied in-place before all other resolution.",
    )

    def _resolve_per_structure_info(self, data) -> dict:
        """Resolve ligand name, param file, and res number for one structure.

        Uses override fields first, falling back to auto-detection from
        the LoadedStructure's metadata and atom_array.

        Returns dict with keys: ligand_name, param_file, ligand_res_number.
        Any value may be None if detection failed.
        """
        from blocks.utils.params import resolve_param_file

        info: dict = {
            "ligand_name": None,
            "param_file": self.override_param_file,
            "ligand_res_number": None,
        }

        # --- Determine ligand name ---
        # When ligand_map is set, use its values as candidate names
        candidate_names = self.override_ligand_names
        if candidate_names is None and self.ligand_map:
            candidate_names = list(self.ligand_map.values())

        if candidate_names is not None:
            # Use first candidate name that matches atoms in this structure
            if data.atom_array is not None:
                for name in candidate_names:
                    if _detect_ligand_res_number(data.atom_array, name) is not None:
                        info["ligand_name"] = name
                        break
                if info["ligand_name"] is None:
                    # None matched — use first candidate as fallback
                    info["ligand_name"] = candidate_names[0]
                    logger.warning(
                        "No candidate ligand names %s matched atoms in %s. "
                        "Using '%s' as fallback.",
                        candidate_names,
                        data.path,
                        info["ligand_name"],
                    )
            else:
                info["ligand_name"] = candidate_names[0]
        else:
            # Auto-detect from metadata
            meta_ligands = data.metadata.get("ligand")
            if meta_ligands and isinstance(meta_ligands, list):
                info["ligand_name"] = meta_ligands[0]
            elif meta_ligands and isinstance(meta_ligands, str):
                info["ligand_name"] = meta_ligands

        lig = info["ligand_name"]

        # --- Resolve param file ---
        if info["param_file"] is None and lig is not None:
            try:
                info["param_file"] = resolve_param_file(lig)
            except KeyError:
                logger.warning(
                    "No params file found for ligand '%s' in %s. "
                    "Rosetta will run without -extra_res.",
                    lig,
                    data.path,
                )

        # --- Detect ligand_res_number from atom array ---
        if lig is not None:
            if data.atom_array is not None:
                res_num = _detect_ligand_res_number(data.atom_array, lig)
                if res_num is not None:
                    info["ligand_res_number"] = res_num
                else:
                    logger.error(
                        "Could not detect ligand_res_number: no atoms with "
                        "res_name='%s' found in %s. Skipping this structure.",
                        lig,
                        data.path,
                    )
            else:
                logger.error(
                    "Cannot auto-detect ligand_res_number without a loaded "
                    "atom array for %s.",
                    data.path,
                )

        return info

    def _dalphaball_command(self) -> list[str]:
        # Custom protocols without Holes do not acquire an unrelated dependency.
        uses_holes = next(ET.parse(self.protocol).iter("Holes"), None) is not None
        if not uses_holes:
            return []
        configured = self.dalphaball_bin
        if not configured:
            tokens = shlex.split(Path(self.flags).read_text(), comments=True)
            for index, token in enumerate(tokens):
                if token == "-holes:dalphaball" and index + 1 < len(tokens):
                    configured = tokens[index + 1]
                elif token.startswith("-holes:dalphaball="):
                    configured = token.split("=", 1)[1]
        binary = shutil.which(configured or "DAlphaBall.gcc")
        if binary is None and configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                binary = str(candidate.resolve())
        if binary is None:
            raise FileNotFoundError(
                "The ligand_ddg protocol uses Rosetta's Holes filter. Install DAlphaBall "
                "and set DALPHABALL_BIN or dalphaball_bin to its executable path."
            )
        return ["-holes:dalphaball", binary]

    def create_command(self, handler, sandbox_dir, chunk_id):
        """Convert inputs to PDB, detect ligand info, and build DDG script."""
        import biotite.structure.io.pdb as pdb_io

        dalphaball_args = self._dalphaball_command()
        sandbox_dir = Path(sandbox_dir)
        pdb_dir = sandbox_dir / "pdb_inputs"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        output_dir = sandbox_dir / "scorefiles"
        output_dir.mkdir(parents=True, exist_ok=True)

        script_lines = []

        for data in handler.get_data(which="last", load=True):
            p = Path(data.path)
            pdb_path = pdb_dir / f"{data.basename}.pdb"

            # Apply ligand_map renaming before PDB write and detection
            if self.ligand_map and data.atom_array is not None:
                for old_name, new_name in self.ligand_map.items():
                    mask = data.atom_array.res_name == old_name
                    if np.any(mask):
                        data.atom_array.res_name[mask] = new_name
                        logger.debug(
                            "Renamed ligand %s -> %s in %s",
                            old_name,
                            new_name,
                            data.basename,
                        )

            if p.suffix == ".pdb" and not self.ligand_map:
                if not pdb_path.exists():
                    self.stage_or_copy(p, pdb_path)
            else:
                pdb_file = pdb_io.PDBFile()
                pdb_file.set_structure(data.atom_array)
                pdb_file.write(str(pdb_path))
                logger.debug("Converted %s -> %s", p, pdb_path)

            info = self._resolve_per_structure_info(data)
            lig_res = info.get("ligand_res_number")
            param = info.get("param_file") or self.override_param_file

            if lig_res is None:
                logger.error(
                    "No ligand_res_number for %s — skipping DDG calculation.",
                    data.basename,
                )
                script_lines.append(
                    f"echo 'SKIPPED (no ligand_res_number): {data.basename}'"
                )
                continue

            # Materialize archived structures before execution.
            protocol = str(self.stage_file(self.protocol, sandbox_dir))
            flags = str(self.stage_file(self.flags, sandbox_dir))
            if param:
                param = str(self.stage_file(param, sandbox_dir))

            scorefile = str(output_dir / f"{data.basename}_ddg.sc")
            argv = [
                self.rosetta_bin, "-parser:protocol", protocol, f"@{flags}",
                "-s", str(pdb_path), "-out:file:score_only",
                "-out:file:scorefile", scorefile,
                "-parser:script_vars", f"ligand_res_number={lig_res}",
                *dalphaball_args,
            ]
            if param:
                argv += ["-extra_res", param]
            cmd = shlex.join(argv)

            script_lines.append(f"echo 'Running DDG: {data.basename}'")
            script_lines.append(cmd + " || echo 'FAILED: " + data.basename + "'")

        if not script_lines:
            raise ValueError("No valid structures to process in create_command.")

        return ["bash", "-c", "\n".join(script_lines)]

    def compile(self, sandbox_dir):
        """Parse DDG score files and return handler with metrics + path_handle for backlinking."""
        sandbox_dir = Path(sandbox_dir)
        output_dir = sandbox_dir / "scorefiles"

        rows = []
        for scorefile in sorted(output_dir.glob("*_ddg.sc")):
            ddg = _parse_scorefile(scorefile)
            rows.append(
                make_row(
                    name=self.name,
                    path=None,
                    path_handle=scorefile,
                    metrics={
                        self.name: {"ddg_norepack": ddg, "scorefile": str(scorefile)}
                    },
                )
            )
        if not rows:
            raise ValueError("No score files found in compile.")

        return PipelineDataHandler(rows)


def _parse_scorefile(scorefile: Path) -> float | None:
    """Parse ddg_norepack value from a Rosetta .sc scorefile."""
    import pandas as pd

    try:
        # Find the first SCORE: line (header)
        header_idx = 0
        with open(scorefile) as f:
            for i, line in enumerate(f):
                if line.startswith("SCORE:"):
                    header_idx = i
                    break

        df = pd.read_csv(
            scorefile,
            sep=r"\s+",
            skiprows=header_idx,
            comment="#",
        )

        if "ddg_norepack" in df.columns:
            return float(df["ddg_norepack"].iloc[0])
        else:
            logger.warning(
                "'ddg_norepack' not in %s. Columns: %s",
                scorefile,
                list(df.columns),
            )
    except Exception as e:
        logger.error("Failed to parse %s: %s", scorefile, e)

    return None
