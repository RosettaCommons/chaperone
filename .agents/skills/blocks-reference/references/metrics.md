# Metric and energy blocks

Use cheap geometry and confidence checks before energy calculations. Combine orthogonal hard gates instead of one weighted score.

| Block | Best Use, Strengths, & Limitations |
|---|---|
| `clash` | Cheap protein-backbone clash gate. Does not detect side-chain or ligand clashes. |
| `ligand_clash` | Cheap severe protein-ligand penetration gate. Reports the closest contact, not whole-pocket strain. |
| `ligand_pocket` | Ligand contact, enclosure, shell-depth, and distribution geometry. Requires explicit `ligand_resnames=[...]`; pass `core_atom_names` and `tail_polar_atom_names` when those subsets matter. Does not measure affinity. |
| `aromatic_neighbors` | Count nearby aromatic residues. Fast and interpretable, but ignores orientation and energy. |
| `contact_ms` | Binder-target contact molecular surface. Measures interface extent, not binding quality. |
| `rasa` | Ligand burial by SASA/RASA. Burial alone can reward a chemically poor pose. |
| `hbond_metrics` | HBPLUS protein-ligand H bonds. Sensitive to protonation and atom names; counts are not bond strength. |
| `plip` | Detailed ligand interaction types. Useful breadth, but ligand perception and heuristic cutoffs require validation. |
| `pocket_coulomb` | Fast charged-pocket complementarity ranking. Omits solvent, dielectric, and full energetics. |
| `chirality_check` | Compare ligand CIP stereochemistry to a reference. RDKit failures or no detected stereocenters currently pass silently; prevalidate before a hard gate. |
| `secondary_structure` | Global and pocket secondary structure plus disorder. Useful triage, not proof of topology. |
| `symmetry` | Diagnose cyclic subunit, axis, angle, and screw deviations. Requires the correct point group and chain mapping. |
| `dssr_snap_metrics` | Protein-DNA groove, stacking, H-bond, and interface metrics. DNA-specific. |
| `metal3d` | Predict transition-metal site probability and location. Specialized probability, not proof of coordination. |
| `rmsd` | Global, motif, ligand, or selected-atom deviation. Sensitive to alignment, pairing, chains, and reference choice. |
| `peppr` | Broad RMSD, TM, lDDT, interface, and DockQ comparison. Comprehensive but requires a valid paired reference. |
| `foldseek` | Fast structural clustering and diversity. Approximate and threshold sensitive. |
| `ipsae` | PAE-derived interface confidence from AF3/RF3 sidecars. Confidence is not affinity. |
| `mpnn_score` | Cheap MPNN confidence ranking. Model self-likelihood is not an independent oracle. |
| `rotamer_metrics` | Local side-chain plausibility. Library dependent and does not establish interface energetics. |
| `tmol_interface` | Single-point ligand-interface ddG with term breakdown. Fast energy view, but score-function and preparation dependent. |
| `tmol_minimize` | Local Cartesian all-atom minimization. Not a predictor; unconstrained relaxation can move required geometry. |
| `ligand_ddg` | Rosetta ligand binding-energy estimate. Expensive and sensitive to params, protonation, pose, and score function. |
| `ligand_ddg_soft` | Replicated soft-repulsive ligand ddG. More robust, but costly and can tolerate unrealistic contacts. |
| `ppi_ddg` | Relaxed protein-interface ddG. Expensive and starting-pose/score-function dependent. |

## Selection shortcuts

- Early validity: `clash`, `ligand_clash`, and required `rmsd` gates.
- Ligand pocket: `ligand_pocket`, `rasa`, `hbond_metrics`, then `plip` if needed.
- Diversity: `foldseek`; use `rmsd` for defined reference geometry.
- Interface confidence: `ipsae` for compatible AF3/RF3 outputs.
- Energy ranking: geometry first, then the task-specific tmol or Rosetta block.
