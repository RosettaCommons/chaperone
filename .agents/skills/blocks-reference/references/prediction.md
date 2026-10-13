# Structure-prediction blocks

Choose the oracle by required modality and the metrics in `score.py`, then calibrate its throughput and failure modes on representative inputs. Prediction confidence is not experimental binding or activity.
The following blocks are sorted by runtime order from fastest to slowest. Rember that all these oracles might have different ways of renaming their atoms and that is why we need to use remap ligand after folding if there are atom name specific downstream metrics.

| Block                   | Best Use, Strengths, & Limitations |
|-------------------------|-------------------------------------|
| `esmfold2`              | Fast, cheap triage for monomer, protein-ligand, or PPI. Single-sequence, no-MSA, high throughput. Lower accuracy than heavy oracles; may rename atoms. |
| `rf3`                   | Default for ligand-aware & multicomponent prediction. Direct biomolecular output, balanced/fast, supports ligand templates. Requires careful ligand setup; new, so calibrate locally. |
| `protenix`              | For multicomponent prediction needing model size choices or training-free guidance. Handles proteins/ligands/nucleic acids and constraints, but is complex to run and less locally validated. |
| `af3`                   | Most reliable for final scoring, evaluation of broad complexes, modifications, or covalent chemistry. Robust input model, flexible templates, but strict input/convention requirements. |
| `rf3_dev`               | For advanced RF3 (manual templates, pocket templating, classifier-free guidance). Extra controls; more complex and slower due to advanced features. |
| `af3_with_target_template` | Use to hold a receptor fixed with a known template when testing partner design. Controls template bias, but not equivalent to untemplated prediction. Must declare this bias. |
| `openfold3`             | Open-source AF3-alternative for complexes, ligands, nucleic acids, templates, MSA. Less studied locally; not a primary default. |
| `colabfold`             | Mature AF2-style monomer refolding or protocol reproduction. Works offline, supports templates/MSA; mostly for monomers, and remote templates not for de novo. |
| `boltz2`                | For Boltz/affinity scoring, not routine screening. Broad input support. Slow and unreliable for direct binding prediction; not recommended as default. |
| `chai1`                 | Alternate for multimodal prediction in comparisons. Handles proteins, ligands, nucleic acids; unreliable and expensive—use only after careful calibration. |

## Selection shortcuts

- Quick triage: `esmfold2`.
- Ligand or multicomponent screen: `rf3`.
- Final gating verification:`af3`.
- Template-conditioned receptor test: `af3_with_target_template` or, for RF3-specific guidance, `rf3_dev`.
- Reproduce an AF2/ColabFold protocol: `colabfold`.
- Use `openfold3`, `protenix`, `boltz2`, or `chai1` only for a required capability or a bounded comparison with measured incremental value.
