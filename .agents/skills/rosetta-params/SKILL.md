---
name: rosetta-params
description: Expert knowledge on generating and handling Rosetta small molecule .params files - ligand setup, rotamer generation, charge assignment, and troubleshooting. Use when users need to create, modify, or debug Rosetta params files for ligands.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit"]
---

# Rosetta Small Molecule Params Files

Comprehensive reference for generating, modifying, and troubleshooting Rosetta `.params` files for small molecule ligands.

Install the tools you need and use their public commands or explicitly configured
paths. Set `ROSETTA_PATH` to a Rosetta checkout containing `main/`. Optional helper
variables (`XTB`, `CHEMCRAFT`, `DUMP_ROTAMERS`, `ADD_VXS`, `RIGID_BODY_ROTAMERS`)
refer to your own installations; check availability before using a recipe.

## Example Questions This Skill Can Answer

- "How do I generate a params file for my ligand?"
- "How do I add rotamers to my params file?"
- "What's the difference between genpot and ref2015 params?"
- "My ligand isn't being recognized by Rosetta"
- "How do I fix charges on my ligand?"
- "How do I generate conformers for my ligand?"
- "What do the CHI lines mean in a params file?"
- "How do I set up a new ligand for RIFgen/RIFdock?"

---

## Overview

A `.params` file tells Rosetta how to understand a non-standard residue (ligand). Standard residues (amino acids, DNA, RNA) have built-in params at `$ROSETTA_PATH/main/database/chemical/`.

For each ligand, you technically only need **one params file** unless the ligand undergoes conformational changes that significantly alter charge/proton distribution.

### Chaperone Integration

Params files used with the `ligand_ddg` block are stored in `src/blocks/utils/params/` and auto-discovered at runtime. To add a new ligand:

1. Generate the `.params` file (see steps below)
2. Place it in `src/blocks/utils/params/<LIGAND_CODE>.params`
3. It will be automatically available via `ligand_name="<LIGAND_CODE>"`

Override the params directory with `CHAPERONE_LIGAND_PARAMS_DIR` env var.

---

## Step-by-Step: Generating a Params File

### Step 1: Generate Starting 3D Structure

Options (pick one):
- **Draw in VMD**: `vmd`
- **Draw in Chemcraft**: `$CHEMCRAFT` — copy XYZ from "Coords" tab
- **From ChemDraw**: Save as `.sdf`, convert to SMILES, generate 3D with RDKit
- **From PDB**: Save ligand coordinates directly (likely needs hydrogen/charge fixing in Step 2)

### Step 2: Fix Structure (if needed)

**Add hydrogens** — use OpenBabel or PyMOL:
```bash
obabel -i pdb ligand.pdb -o mol2 -O ligand.mol2
```
Note: OpenBabel does NOT handle Rosetta score lines — remove them first.

If OpenBabel fails, use VMD: fix valence and hydrogens, save as mol2.

**Add partial charges** — use antechamber:
```bash
antechamber -i input.mol2 -fi mol2 -o charged.mol2 -fo mol2 -c bcc -pf y -dr n -at sybyl -nc <CHARGE>
```

Or minimize geometry with XTB (semiempirical QM):
```bash
$XTB input.xyz --opt --chrg <CHARGE> --gbsa h2o -v > xtb.out
# Optimized geometry in xtbopt.xyz
```

### Step 3: Generate Rotamers/Conformers

**For ligands available from PDB** — use CSD conformer generator:
```bash
conformer_generator charged.mol2 conformers_out.mol2 -nc <NUM_CONFORMERS>
```

**For custom ligands**:
- Save all rotamers into a single PDB, separated by `TER` lines (required by Rosetta)
- Convert to mol2: `cat rot*.pdb > all_conformers.pdb`

### Step 4: Score and Select Rotamers

Use Adam's AIMNet to score and select the best rotamers.

### Step 5: Generate the Params File

**Basic ref2015 params** (under PyRosetta environment):
```bash
python $ROSETTA_PATH/main/source/scripts/python/public/molfile_to_params.py \
    --keep-names ligand.mol2 -n LIG
```

**With extra rotamers**:
```bash
python2 $ROSETTA_PATH/main/source/scripts/python/public/molfile_to_params.py \
    --keep-names --clobber --extra_torsion_output --centroid ligand.mol2 -n LIG
```

**For genpot (generic potential)**:
```bash
python $ROSETTA_PATH/main/source/scripts/python/public/generic_potential/mol2genparams.py \
    -s ligand.mol2
```

### Step 6: Verify the Params File

- Check CHI lines for reasonable movable chi angles
- Verify `NBR_RADIUS` (increase if too small — default may miss neighboring atom interactions)
- Dump and visually check rotamers:
```bash
python $DUMP_ROTAMERS --pdb_in ligand.pdb --params_in ligand.params
```

### Step 7 (optional): Rigid Body Rotamer Sampling

**Do NOT use if your score function has `cart_bonded` or other Cartesian terms during design.**

```bash
python $ADD_VXS
python $RIGID_BODY_ROTAMERS
```

### Step 8 (optional): Add Rotamer Library

Concatenate all rotamers into one PDB and reference it in the params file via the `PDB_ROTAMERS` line. The PDB and `.params` file must be in the same folder.

**Note: `PDB_ROTAMERS` does NOT work with genpot params files.**

---

## Generating Both genpot and ref2015 Params

```bash
# 1. Generate individual rotamers (ref2015)
python $ROSETTA_PATH/main/source/scripts/python/public/molfile_to_params.py \
    ligand.sdf -n LIG

# 2. Convert each rotamer PDB to mol2
for i in *.pdb; do obabel -i pdb $i -o mol2 -O ${i%.pdb}.mol2; done

# 3. Generate genpot params for all
conda activate common_use
for i in *.mol2; do
    python $ROSETTA_PATH/main/source/scripts/python/public/generic_potential/mol2genparams.py -s $i
done
# Rename to distinguish
for i in *.params; do mv $i ${i%.params}_genpot.params; done
for i in *_*_0001.pdb; do mv $i ${i%_0001.pdb}_genpot.pdb; done

# 4. Generate ref2015 params (for RIFgen/RIFdock)
mkdir good_to_use && mv *_genpot* good_to_use/ && cd good_to_use/
for i in *_genpot.mol2; do
    python $ROSETTA_PATH/main/source/scripts/python/public/molfile_to_params.py $i --keep-names -n LIG
done
mv LG.params LIG_ref2015.params

# 5. Post-processing
# a) Increase NBR_RADIUS (e.g. 9.15 -> 15.15)
# b) Double-check CHI angles

# 6. Dump rotamers with PROTON_CHI angles
for i in *_00*_genpot.pdb; do
    python $DUMP_ROTAMERS \
        --pdb_in $i --params_in LIG_0001_genpot.params
done
# Fix ligand names
for i in LIG_*_genpot_1_*.pdb; do sed -i 's/LG1/LIG/g' $i; done
```

---

## Params File Format — Line-by-Line

```
NAME KXH                          # 3-letter ligand code
IO_STRING KXH Z                   # name + single-letter code (Z for unknown)
TYPE LIGAND                       # LIGAND, DNA, RNA, etc.
AA UNK                            # amino acid type (UNK for ligands)

ATOM P2   PG3  X     0.810        # ATOM <name> <rosetta_type> <mm_type> <charge>
ATOM C10  aroC X    -0.050        # ... more atoms ...

BOND_TYPE C10  N5   1             # BOND_TYPE <atom1> <atom2> <bond_order>

NBR_ATOM O10                      # Neighbor atom (near geometric center)
NBR_RADIUS 17.96                  # Radius of gyration. Set generously — too
                                  # small misses energy calculations with neighbors.

ICOOR_INTERNAL O10 0.0 0.0 0.0 O10 C11 P1
                                  # Internal coordinates:
                                  # child phi theta distance parent angle_atom torsion_atom

CHI 1  C13  O12  P2  O9           # CHI <number> <atom1> <atom2> <atom3> <atom4>
                                  # atom4 is moved by this chi angle

PROTON_CHI 7 SAMPLES 5 -107 -84 -67 -45 -22 EXTRA 2 8 4
                                  # Optional: explicit chi sampling
                                  # SAMPLES N val1..valN = discrete chi values
                                  # EXTRA N val1..valN = +/- perturbations per sample

PDB_ROTAMERS ligand_rotamers.pdb  # Optional: external rotamer library
                                  # Must be in same folder as .params
                                  # Does NOT work with genpot!
```

### CHI / PROTON_CHI Ordering Rules

CHI angles with `PROTON_CHI` definitions **must come first**. If a `PROTON_CHI` references a CHI that appears after a CHI without `PROTON_CHI`, Rosetta will set it to 0.0.

Correct:
```
CHI 1  C15  C18  C19  C20
CHI 2  C18  C19  C20  C21
CHI 3  C19  C20  C21  O4
PROTON_CHI 1 SAMPLES 3 -120 -90 -58 EXTRA 1 5
PROTON_CHI 2 SAMPLES 1 180 EXTRA 1 7
```

Wrong (CHI 3 will be forced to 0.0):
```
CHI 1  ...
CHI 2  ...
CHI 3  ...
PROTON_CHI 1 SAMPLES 3 -120 -90 -58 EXTRA 1 5
PROTON_CHI 3 SAMPLES 2 80 180 EXTRA 2 20 10
```

**Warning**: Defining `PROTON_CHI` for many rotatable bonds causes combinatorial explosion. For FastDesign workflows, only define `PROTON_CHI` for bonds with distinct discrete chi values. FastDesign does its own chi sampling/minimization.

---

## Generating a Ligand from Scratch

1. **Create 3D structure** in VMD or Chemcraft
2. **Rename atoms** — RIFgen/RIFdock don't recognize standard backbone names (CA, CB, N, O). Rename to C1, C2, O1, O2, H1, H2, N1, N2, etc. Change residue name to a 3-letter code, residue number to 1.
3. **Relax the ligand** before loading:
   ```bash
   "$ROSETTA_BIN" \
       -parser:protocol LA_relax_and_score_ligand.xml \
       -in:file:s changed.pdb -beta \
       @flags_rescore_natives \
       -suffix _relax \
       -extra_res ligand_genpot.params
   ```
4. The resulting PDB is suitable for rotamer generation

---

## Common Issues

| Problem | Solution |
|---------|----------|
| Rosetta doesn't recognize ligand | Check `TYPE LIGAND` is set; ensure `-extra_res ligand.params` flag is passed |
| NBR_RADIUS too small | Increase it (e.g. to 15+). Too small = missing energy terms with neighbors |
| RIFgen caches wrong ligand | Use different `cache_dir` for each rotamer set — RIFgen only caches by ligand name |
| Rotamers look unreasonable | Dump with `dump_rotamers.py` and visually inspect |
| PROTON_CHI ignored | Ensure CHI lines with PROTON_CHI come before CHI lines without |
| `PDB_ROTAMERS` not working | Only works with ref2015 params, NOT genpot |
| OpenBabel fails on Rosetta PDB | Remove Rosetta score lines from the PDB first |

---

## Configurable tools

User-installed helpers have no fixed location; the variables below are placeholders.

| Tool | Path |
|------|------|
| OpenBabel | `obabel` |
| VMD | `vmd` |
| Chemcraft | `$CHEMCRAFT` |
| XTB | `$XTB` |
| CSD Conformer Gen | `conformer_generator` |
| molfile_to_params | `$ROSETTA_PATH/main/source/scripts/python/public/molfile_to_params.py` |
| mol2genparams | `$ROSETTA_PATH/main/source/scripts/python/public/generic_potential/mol2genparams.py` |
| dump_rotamers | `$DUMP_ROTAMERS` |
| rigid_body_rotamers | `$RIGID_BODY_ROTAMERS` |
| add_VXs_to_params | `$ADD_VXS` |
| Rosetta database | `$ROSETTA_PATH/main/database/chemical/` |
