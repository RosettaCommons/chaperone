# Evaluation task: de novo biotin binders

REQUIRED_N = {{REQUIRED_N}}
Submit exactly **{{REQUIRED_N}} de novo mini-protein designs** that bind a
modified biotin ligand.

You are given a ligand-only structure of **linker-modified biotin** — the natural
biotin scaffold with an extended aliphatic linker
(`inputs/biotin_ligand_only.pdb`). It has a rigid bicyclic core and flexible tail.

## Design details

- **Ligand name:** `B;N`
- **SMILES:** `N1C(=O)N[C@@H]2[C@@H](SC[C@H]12)CCCCC(=O)NCCCCC`

The ligand's atoms split into two groups (names as in
`inputs/biotin_ligand_only.pdb`):

- **Rigid core:** `N1, N2, S1, O3, C2, C3, C4, C5, C6, C7, C8, C9,
  C10`. Core-restricted contacts and aromatic neighbors are measured against
  these atoms after canonical ligand-name remapping.
- **Linker (must stay exposed):** `C14, C15, C16` — the distal end of the
  aliphatic tail. At least one of these must remain solvent-exposed.


## Design criteria

1. **Linker exposure** — ≥1 of `C14, C15, C16` has relative SASA ≥ 0.10.
2. **Polar satisfaction** — ≥3 protein–ligand hydrogen bonds.
3. **Core contact** — core-restricted contact molecular surface `contact_ms ≥ 140`.
4. **Aromatic pocket contact** — ≥1 unique PHE/TYR/TRP/HIS residue within 6 Å
   of the rigid biotin core. A contact to the flexible linker does not satisfy
   this gate.
5. **High interface confidence** — fold quality and the protein-ligand interaction are checked by an independent held-out structure-prediction oracle.
    Make sure to create a high confident interaction region to the best of your abilities.
6. **Distinct designs** — every submitted sequence must be unique and use its
   own independently generated de novo backbone. Every pair of submitted
   sequences must have global Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using
   affine gap penalties −10/−0.5.



## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The scorer
  generates deterministic row IDs when it is omitted.
- Every pair of designs must have global Needleman–Wunsch/BLOSUM62 identity
  ≤ 70%, using affine gap penalties −10/−0.5.
- Do not add structures, coordinates, paths, the ligand, lineage records,
  telemetry, or any other column to `designs.csv`.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, or a pairwise identity above 70%.
