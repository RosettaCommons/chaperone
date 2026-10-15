---
name: blocks-reference
description: Choose and compare Chaperone blocks by scientific task, modality, speed, cost, evidence quality, and known limitations. Use when an agent needs a tool recommendation, a design-funnel block, an oracle or metric choice, an alternative to a slow block, or a quick inventory of available design, sequence, prediction, analysis, energy, and workflow blocks.
---

# Blocks Reference

Choose the least expensive block that directly measures the campaign quantity. Treat this skill as qualitative selection guidance; retrieve exact parameters from `docs/site/blocks-reference.md` or the block source immediately before implementation.

## Select a block

1. State the required quantity and modality: backbone generation, inverse folding, monomer fold, multicomponent structure, ligand geometry, interface confidence, energy, diversity, or data movement.
2. Read only the matching reference:
   - Generation, co-design, inverse folding, and language models: [design-sequence.md](references/design-sequence.md)
   - Structure-prediction oracles: [prediction.md](references/prediction.md)
   - Geometry, confidence, energy, and diversity metrics: [metrics.md](references/metrics.md)
   - Ingest, filtering, annotation, structure repair, selection, and packaging: [utilities.md](references/utilities.md)
3. Prefer a direct, cheap measurement before a costly or indirect oracle. Use cheap orthogonal gates before expensive folding or energy calculations.
4. Preserve every final metric or named oracle required by `score.py`. Do not silently replace it with a proxy.
5. If local evidence is absent, label quality, throughput, and transferability as unknown and run the smallest representative calibration.
6. Report the recommendation, rejected alternatives, decisive tradeoff, and required calibration or failure check.

## Default choices

- Use `rfd3` for de novo backbone or atom-aware generation. Use `rfd3_partial` only for local diversification; its outputs are correlated descendants, not independent de novo designs.
- `mpnn` runs LigandMPNN from its own `ligandmpnn` image: `model_type` (protein, ligand, soluble or membrane MPNN) and `noise_level` select the model and checkpoint. `foundry_mpnn` is a separate block that runs MPNN from the Foundry image used by RFD3 and RF3.
- Use `esmfold2` for cheap single-sequence monomer triage.
- Use `rf3` for ligand-aware or multicomponent prediction when it satisfies the scorer. Use `af3` when AF3 itself, custom chemistry, or AF3-equivalent final evaluation is required.
- Avoid the Boltz and Chai families by default in this environment: they are comparatively slow here and have not been reliable screening oracles. Use one only for a missing capability and after a bounded, matched calibration shows useful signal per cost.
- Use geometric/contact metrics before Rosetta or tmol energies. Energy scores are model- and preparation-dependent and do not establish binding alone.

## Guardrails

- Compare tools on the same inputs, fixed scorer, oracle/model versions, and denominator. Do not infer superiority from unmatched runs.
- Check execution failures, missing metrics, and diversity alongside conditional pass rate.
- Validate ligand names, chain IDs, covalent bonds, protonation, atom mapping, and confidence sidecars before trusting ligand or interface metrics.
- Separate hard gates from ranking signals. Self-confidence, language-model likelihood, docking energy, and one contact count are usually ranking signals unless independently calibrated.
- Do not call multiple samples, seeds, point mutants, or partial-diffusion descendants independent diversity.
- Consult `/debugging-inputs` for chain, ligand, residue-numbering, or parsing failures and `/rosetta-params` for ligand parameterization.

## Verify availability and details

Use `Blocks.keys()` or the AST-backed registry in `src/blocks/__init__.py` as the source of truth for availability. Use `docs/site/blocks-reference.md` for generated parameter summaries, then inspect only the selected block's source when behavior remains ambiguous. The `copy` and `copy2` blocks are test fixtures and are intentionally excluded from the user-facing references.

When adding or removing a public block, update the relevant table in this skill. Flag a block that exists in source but is missing from the generated docs instead of assuming it is unsupported.
