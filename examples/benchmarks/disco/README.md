# DISCO — Ligand-Conditioned Protein Co-Design Benchmark

[DISCO](https://github.com/DISCO-design/DISCO) co-designs protein sequence + structure conditioned on a ligand, validated by AF3 refolding.

**Pipeline**: `disco -> af3 -> rmsd`

**Task**: 200-residue heme B binder. A design is co-designable if backbone RMSD < 2 A after refolding.

## 1) Code-level features

- Input JSON uses AF3 Server format: masked protein (`-`) + ligand (SDF/SMILES/CCD)
- DISCO labels ligands as `l01` in output — AF3 needs `specified_ccd={"l01": "HEM"}` mapping
- See `src/blocks/networks/design/disco/input.md` for full input format docs

## 2) Pilot-level experiments

- **Length sweep**: 150/200/250 residues (add jobs to input JSON)
- **Preset**: `diverse` for variety, `designable` for higher refolding rates
- **Effort**: `fast` (~4x cheaper) vs `max` (full quality, recommended for conditional)
- **Seeds**: increase `num_seeds` for more samples (100+ for statistics)

## 3) Campaign-level criteria

- Co-designability rate (backbone + ligand RMSD < 2 A) across 100+ seeds
- Structural diversity (pairwise TM-score)
- Ligand placement fidelity after refolding
