---
name: debugging-inputs
description: Troubleshoot common input preparation issues across pipeline blocks — chain ID mismatches, ligand numbering, residue ID conflicts, and structure parsing failures. Use when users report RMSD errors, AF3 input mismatches, or unexpected chain assignments.
allowed-tools: ["Read", "Grep", "Glob", "Bash"]
---

# Debugging Pipeline Inputs

Reference for diagnosing and fixing input preparation issues that cause failures or silent errors across pipeline blocks. These issues typically surface as RMSD pairing failures, AF3 crashes, or incorrect metric values.

## Example Questions This Skill Can Answer

- "My ligand RMSD is NaN even though AF3 ran successfully"
- "RMSD says atom count mismatch between source and design"
- "AF3 input JSON has wrong chain IDs for ligands"
- "RFD3 output has all ligands on one chain"
- "Chain IDs don't match between my RFD3 and AF3 outputs"
- "res_id mismatch between source and prediction"

---

## Known Issues

### 1. Multi-ligand theozymes: all ligands on one chain

**Symptom**: RMSD `select_mask_by_name_pairing` falls back to `(res_name, atom_name)` matching with a warning: "No common atoms with exact (chain_id, res_id, atom_name) match."

**Root cause**: Input theozyme PDB files have multiple ligand components (e.g. ADP, MG, 3PG) all on the same chain (typically chain A or B). RFD3 preserves input chain IDs via `extract_ligand_array` + `struc.concatenate` in foundry's `input_parsing.py`. AF3 then splits each ligand into a separate chain (B, C, D…) for its input format. The RFD3 source and AF3 output end up with different chain assignments for the same ligands.

**Fix**: Ensure each ligand component in the input theozyme PDB sits on its own chain. Use this script to split them:

```python
import biotite.structure.io.pdb as pdb
from biotite.structure import get_residue_starts
import numpy as np

STANDARD_AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
               'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

f = pdb.PDBFile.read("theozyme.pdb")
atoms = f.get_structure(model=1, include_bonds=False)
chain_ids = atoms.chain_id.copy()
used_chains = set(np.unique(chain_ids))

for c in np.unique(atoms.chain_id):
    mask = atoms.chain_id == c
    chain_atoms = atoms[mask]
    starts = get_residue_starts(chain_atoms)
    res_names = [str(chain_atoms.res_name[s]) for s in starts]

    # Find ligand residue indices on this chain
    lig_indices = [i for i, r in enumerate(res_names) if r not in STANDARD_AA and r != 'ORI']
    if len(lig_indices) <= 1:
        continue

    global_indices = np.where(mask)[0]
    for lig_idx in lig_indices[1:]:  # first keeps original chain
        s = starts[lig_idx]
        e = starts[lig_idx + 1] if lig_idx + 1 < len(starts) else len(chain_atoms)
        new_chain = next(ch for ch in 'BCDEFGHIJKLMNOPQRSTUVWXYZ' if ch not in used_chains)
        used_chains.add(new_chain)
        for idx in range(s, e):
            chain_ids[global_indices[idx]] = new_chain

atoms.chain_id = chain_ids
out = pdb.PDBFile()
out.set_structure(atoms)
out.write("theozyme_fixed.pdb")
```

**Verification**: After fixing, each ligand should be on its own chain:
```bash
python -c "
import biotite.structure.io.pdb as pdb
from biotite.structure import get_residue_starts
import numpy as np
f = pdb.PDBFile.read('theozyme_fixed.pdb')
a = f.get_structure(model=1, include_bonds=False)
for c in np.unique(a.chain_id):
    m = a.chain_id == c
    rn = np.unique(a.res_name[m])
    print(f'Chain {c}: {list(rn)}')
"
```

---

### 2. Residue ID renumbering by AF3

**Symptom**: RMSD `(chain_id, res_id, atom_name)` pairing finds zero common atoms for ligands, even though chain IDs match. Falls back to `(res_name, atom_name)`.

**Root cause**: AF3 always outputs `res_id=1` for each ligand chain, regardless of the input residue numbering. RFD3 outputs use offset residue IDs (e.g. `res_id=55`) from `_append_ligand` which does: `ligand_array.res_id = ligand_array.res_id - min + max(protein_res_id) + 1`.

**Status**: This is a known AF3 output convention. The `(res_name, atom_name)` fallback in `_get_paired_masks` handles it correctly. Newer foundry versions may normalize residue IDs in RFD3 outputs.

**Impact**: None — the fallback produces correct RMSD values. The warning can be safely ignored.

---

### 3. AF3 input prep: chain ID reassignment

**Symptom**: AF3 input JSON has ligand chain IDs (B, C, D) that don't match the source structure.

**Root cause**: The `create_af3_json_from_design` function in `src/blocks/networks/prediction/helpers.py` historically assigned the next available letter as the chain ID for each ligand, ignoring the source chain ID.

**Fix (applied)**: The function now preserves the source chain ID from the atom array when available. Falls back to next available letter only if the source chain is already used. See the `source_chain = atom_array.chain_id[start]` logic in `helpers.py`.

---

### 4. PDB parsing failures: "Atom IDs are not strictly increasing"

**Symptom**: `biotite.InvalidFileError: Atom IDs are not strictly increasing` when loading certain PDB files.

**Root cause**: Some PDB files (especially from external databases like MCSA) have non-standard atom numbering or CONECT records with IDs exceeding the atom count.

**Workaround**: Load with `include_bonds=False`:
```python
f = pdb.PDBFile.read(path)
atoms = f.get_structure(model=1, include_bonds=False)
```
Or use `parse_to_atom_array` from `pipelines.io` which handles this automatically.

---

## Diagnostic Checklist

When debugging input issues, check these in order:

1. **Chain IDs in source theozyme**: Does each ligand component have its own chain?
   ```python
   from pipelines.io import parse_to_atom_array
   atoms = parse_to_atom_array("theozyme.pdb")
   for c in np.unique(atoms.chain_id):
       mask = atoms.chain_id == c
       print(f"Chain {c}: {np.unique(atoms.res_name[mask])}")
   ```

2. **Chain IDs in RFD3 output**: Do they match the theozyme?
   ```python
   atoms = parse_to_atom_array("rundir/rfd3/chunk_0/design.cif.gz")
   # Compare ligand chains with theozyme
   ```

3. **AF3 input JSON**: Do ligand chain IDs match the RFD3 output?
   ```bash
   python -m json.tool rundir/af3/chunk_0/inputs/design.json | grep -A2 '"ligand"'
   ```

4. **AF3 output**: Do chain IDs and res_ids match expectations?
   ```python
   atoms = parse_to_atom_array("rundir/af3/chunk_0/outputs/design/design_model.cif")
   # Compare with RFD3 output
   ```

5. **RMSD pairing**: Check if fallback was triggered
   ```bash
   grep "Falling back" rundir/logs/*.log
   ```
