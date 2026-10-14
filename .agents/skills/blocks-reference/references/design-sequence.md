# Generation and sequence blocks

Choose generation blocks by design scope, then use an MPNN block for fixed-backbone sequence design.

| Block | Best Use, Strengths, & Limitations |
|---|---|
| `rfd3` | Default de novo, motif, and ligand-aware generation. Flexible and modern; still requires downstream ligand-mpnn for more reliable sequence generation |
| `rfd3_partial` | Diversify a promising structure locally. Useful for when one structure is particularly promising and we want to get more samples similar to it while increasing diversity a bit. However outputs are correlated descendants. |
| `rfd2` | Reproduce RFD2 or use its atomic-design behavior. Prefer RFD3 unless a protocol or matched pilot favors RFD2. |
| `rfd1` | Classic backbone, motif, binder, or symmetric design. Mature, but less suitable for rich ligand chemistry. Very outdated, would avoid unless explicitly told |
| `bindcraft` | End-to-end protein-binder hallucination, ProteinMPNN redesign, AF2 validation, and PyRosetta filtering against an input target structure. Expensive and protein-target-specific; use a high-memory NVIDIA GPU and calibrate its AF2-derived ranking with an independent final oracle. |
| `disco` | Joint sequence-structure design around proteins, ligands, DNA, or RNA. Multimodal, but newer and SUPER SLOW 10 min per design not locally calibrated. |
| `boltzgen` | End-to-end design from a specification. Integrated but expensive, less modular, and subject to the Boltz calibration warning, also not as well aligned with oracles outside of boltz family. |
| `protein_hunter` | Specialized binder hallucination against biomolecular targets, generally really hard to control the design outcomes and very overfitted to boltz which doens't transfer to other oracles. Integrated iteration, but Boltz-dependent and costly. |
| `boltz2_steered` | Suppress unwanted binder helicity in Protein Hunter. Specialized and slow; verify improvement with the final scorer. |
| `foundry_mpnn` | MPNN from the Foundry image shared with RFD3 and RF3. Default inverse folding after `rfd3`. Clean, inexpensive wrapper; limited to fixed-backbone sequence design. |
| `mpnn` | LigandMPNN from its own image (`chaperone install ligandmpnn`). `model_type` picks protein, ligand, soluble or membrane MPNN; `noise_level` picks the checkpoint version. Flexible, but likelihood is not fold/function evidence. |
| `esm2` | Sequence embeddings and similarity features with selectable model sizes. Fast, but indirect, gears more towards natural rather than de novo sequence, and checkpoint-size dependent. |

## Selection shortcuts

- New de novo campaign: `rfd3` then `foundry_mpnn`.
- Local backbone exploration: `rfd3_partial`.
- Reproduce a legacy RFdiffusion protocol: `rfd1` or `rfd2`.
- End-to-end AF2-guided protein binder generation: `bindcraft` on a trimmed target PDB, followed by an independent oracle.
- Sequence representation only: `e1` or `esm2`.
- Use Boltz-based design only after a bounded matched calibration.

`abMPNN-PoE` is mentioned in project guidance but is not separately registered. Confirm its wrapper before promising it.
