# Utility blocks

Use these blocks to preserve schema, lineage, caching, and storage instead of manual file wrangling.

| Block | Best Use, Strengths, & Limitations |
|---|---|
| `ingest` | Load directories, CSVs, dataframes, sequences, or compiled outputs. Validate schema and identifiers. |
| `annotate` | Add interface, H-bond, and ligand annotations. Depends on correct chain and ligand interpretation. |
| `select_motif` | Store residues for downstream motif fixing. Test the selector because mistakes propagate silently. |
| `filter` | Vectorized metric or expression filtering with lineage. Match missing-value behavior to `score.py`. |
| `select_diverse` | Select by sequence identity and optional structural similarity. Greedy and cutoff/order dependent; retain lineage caps. |
| `transform` | Apply a reusable Atomworks transform. Verify schema, annotations, and atom mapping afterward. |
| `to_cif` | Normalize structures to CIF. Cannot recover missing chemistry. |
| `to_pdb` | Convert for PDB-only consumers. May lose identifiers or complex chemistry; prefer CIF otherwise. |
| `remap_ligand` | Restore canonical ligand atom/residue names after prediction. Requires a correct reference and cannot fix a wrong pose. |
| `mutate_residues` | Create named point mutants with lineage. Outputs are related variants and may need repacking/refolding. |
| `restore_sidechains` | Restore selected motif rotamers. Can create clashes; validate or repack afterward. |
| `selective_sidechain_repack` | Repack chosen side chains while fixing backbone and ligand. Cannot repair a bad backbone or pose. |
| `rotamer_sweep` | Enumerate small functional chi-angle hypotheses. Combinatorial and not a whole-protein packer. |
| `materialize` | Write selected stored structures for file-based consumers. Clean up temporary loose files afterward. |
| `order_collection` | Collect sequences, prior-order status, and prediction metrics. Does not run `score.py`. |
| `package_submission` | Validate and write a sequence-only submission. Cannot carry structural or lineage deliverables. |

## Selection shortcuts

- Start data: `ingest`; attach motif/interface metadata with `annotate` or `select_motif`.
- Filter and diversify: `filter` then `select_diverse`.
- Repair ligand names after folding: `remap_ligand`.
- Materialize only survivors: `materialize`.
- Final sequence deliverable: `order_collection` then `package_submission` when appropriate.

`copy` and `copy2` are test fixtures, not scientific generation or diversity blocks.
