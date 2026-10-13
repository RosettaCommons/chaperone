# `enzyme-glycosidase` held-out scorer

`score.py` is the grading pipeline. It never trusts a self-reported number: it
re-folds every submitted sequence with AF3 together with the scorer-supplied
substrate and re-measures everything.

```bash
./score.py --designs=/abs/path/to/submission.csv --nouse_local_gpus
```

## Contract, before any GPU is touched

`SequenceSubmissionSpec` requires 96 rows with `sequence` and
`motif_index_map`. The map must name all six reference positions
`Y77, E86, Y88, R122, Q136, E177`, each landing on a distinct residue of
`sequence`, and the residue at that index must be the right amino acid
(`Y, E, Y, R, Q, E`). A violation writes a contract-failure report and stops.

Those six IDs were read off `inputs/theozyme.pdb`, not guessed: chain A of the
theozyme contains exactly `TYR 77`, `GLU 86`, `TYR 88`, `ARG 122`, `GLN 136`,
`GLU 177` (side-chain atoms only — the theozyme carries no backbone). They are
also exactly the six positions the RFdiffusion3 generation config
(`inputs/config.json`) marks with `unindex` and `select_fixed_atoms`, so the
submission contract and the generation task describe the same motif.

## Oracle

AF3 with `glycosidase_holo_template.json`. The substrate `L:1` is a custom
4-methylumbelliferyl β-glycoside with no CCD code, so its chemistry has to
reach AF3 through the template's `userCCD` block; it cannot be rebuilt from an
input structure alone. The template puts it on chain **B**. The scorer also
appends theozyme chain B to each design's input structure via
`load_context_chains`, so the input the oracle is handed is self-describing
about what is being folded; the geometry that AF3 actually uses comes from the
template.

Everything runs as one chunk on one GPU (`rows_per_job` is set above the row
count); the downstream metric blocks run locally on CPU.

## Gates

| Gate | Metric | Threshold | Rationale |
|---|---|---|---|
| `motif_fidelity` | `motif_rmsd` | ≤ 1.0 Å | All-atom RMSD of the six mapped side chains against the theozyme, atoms paired by name. Tighter than the protease benchmark's 1.5 Å: the theozyme is side-chains-only and a two-residue catalytic core is an easier placement than a metal site with three ligands. |
| **`dyad_geometry`** | `dyad_distance` | **5.9 – 8.9 Å** | The benchmark's defining measurement. See below. |
| `plddt` | `holo.plddt` | ≥ 85 | A designed enzyme that AF3 is unsure about is not a starting point. |
| `substrate_pose` | `substrate_rmsd` | ≤ 2.0 Å | Substrate heavy-atom RMSD after superimposing on the mapped motif. |
| `substrate_confidence` | `holo.ligand_plddt` | ≥ 70 | Ligand pLDDT is systematically lower than protein pLDDT; 70 is a confident small-molecule placement. |
| `pocket_burial` | `holo_rasa.mean_ligand_atom_rasa` | ≤ 0.15 | The substrate must sit in a pocket, not on a surface patch. Mean over all 22 heavy atoms, so the solvent-exposed coumarin leaving group is allowed to be partly accessible. |
| `packing` | `holo_contact_ms` (A vs B) | ≥ 200 Å² | Buried molecular surface between the enzyme and the substrate. Lower than a protein–protein gate because the substrate has only 22 heavy atoms. |
| **soft** `hbonds` | `holo_plip.hydrogen_bonds` | ≥ 3 | The sugar has four free hydroxyls plus a ring oxygen; three H-bonds is the minimum that reads as recognition rather than a hydrophobic slot. Soft because PLIP's geometric criteria are strict and a good pocket can miss it. |

## The dyad gate

A glycoside hydrolase is defined by the *separation* of its two catalytic
carboxylates, not by their presence. In a **retaining** (double-displacement)
enzyme the nucleophile attacks the anomeric carbon directly and the acid/base
protonates the leaving group from ≈5.5 Å away; in an **inverting** enzyme the
two carboxylates sit ≈10 Å apart to make room for the attacking water. Motif
RMSD cannot see this: a design can satisfy a six-residue superposition on
average while the dyad itself opens or collapses.

`dyad_distance` is the distance between the carboxylate carbons — `CD` for Glu,
`CG` for Asp; verified against `inputs/theozyme.pdb`, whose `GLU` residues carry
`CB CG CD OE1 OE2` — of the residues the submission maps to `E86` and `E177`.

**Which mechanism does the committed theozyme encode?** Measured directly on
`inputs/theozyme.pdb`:

| Measurement | Value |
|---|---|
| `E86:CD` ↔ `E177:CD` | **7.387 Å** |
| closest `E86:OE*` ↔ `E177:OE*` | 5.872 Å |
| `E86` nearest carboxylate O → substrate anomeric carbon `C4` | 2.477 Å |
| `E177` nearest carboxylate O → glycosidic oxygen `O1` | 2.810 Å |

The functional assignment is unambiguous and **retaining**: `E86` is poised
2.5 Å under the anomeric carbon for in-line attack (the source example
corroborates this — its PTM branch covalently attaches the substrate to
`GLU A86` to model the glycosyl-enzyme intermediate), and `E177` hydrogen-bonds
the glycosidic oxygen as the general acid. Both carboxylates are on the same
face of the sugar, ≈5.9 Å apart at their closest oxygens — the textbook
retaining arrangement — not the ≈10 Å inverting one.

**Deviation from the plan spec.** `evals/PLAN.md` proposes a 4.5–6.5 Å window
for a retaining enzyme. The committed theozyme's carboxylate-*carbon*
separation is 7.39 Å, which that window excludes — a scorer whose headline gate
rejects its own reference geometry measures nothing. The gate is therefore
recentred on the measured reference: **7.387 ± 1.5 Å → 5.9–8.9 Å**. The ±1.5 Å
tolerance is set to be consistent with the 1.0 Å all-atom motif-RMSD gate (the
error in a CD–CD *separation* is bounded by roughly twice the per-atom
deviation). The window still excludes the ≈10 Å inverting arrangement, so the
gate remains mechanism-discriminating, and it still rejects a collapsed dyad
below 5.9 Å. The canonical 5.5 Å retaining figure is best matched by this
theozyme's closest-oxygen distance (5.87 Å); the carbon–carbon figure runs
larger because the two carboxylate planes are rotated away from each other.

## Why `substrate_rmsd` is bespoke rather than the `rmsd` block

`evals/PLAN.md` specifies `rmsd.bkbn_align.ligand_rmsd`. That block aligns the
first and last structures of a row's lineage on their backbones — and neither
side has one here. A sequence-only submission carries no coordinates (the
scorer's own input structure is a coordinate-free CA placeholder), and the
theozyme is a side-chain-only template with no backbone atoms at all. The only
frame the design and the reference share is the catalytic motif.

`substrate_rmsd` therefore superimposes the design on the theozyme using
exactly the atom pairing `pipelines.evals.motif_rmsd` uses — mapped residues,
atoms paired by name — and then measures the folded substrate against the
reference substrate, paired by atom name (AF3 emits the `userCCD` `atom_id`s
verbatim, so the 22 heavy atoms pair 1:1; the scorer raises if any is missing).
This asks the sharper question anyway: does the substrate sit where the
catalytic machinery needs it, *relative to that machinery*? The 2.0 Å threshold
from the plan is kept.

No `remap_ligand` step is needed: the reference and the fold use the same
`userCCD` atom names.

## Failure policy

`require_finite_metrics` runs over every gated column before any criterion is
evaluated, and the per-structure measurements are computed row by row so a
failure names the offending design. A missing metric or a NaN is an upstream
bug and aborts the scorer; it is never treated as a failing design, and no row
is ever dropped.

## Calibration status

**Uncalibrated.** The thresholds above are the plan's starting spec, adjusted
only where the committed inputs made the spec unimplementable (the dyad window
and the substrate-RMSD alignment frame, both documented above). Nothing was
loosened to improve the reference pipeline's score.

### What the reference run actually scores

`example/submission.csv` (96 designs from RFD3 → filter → LigandMPNN) scores
**0 / 96**. The full per-gate breakdown, from `example/scored/score.json`:

| Gate | Passing | Observed min / median / max |
|---|---|---|
| `motif_fidelity` | **0** | 1.25 / 2.76 / 15.91 Å |
| `dyad_geometry` | 44 | 3.79 / 8.14 / 41.27 Å |
| `plddt` | 54 | 38.0 / 86.3 / 95.7 |
| `substrate_pose` | **0** | 2.25 / 7.86 / 35.44 Å |
| `substrate_confidence` | 58 | 27.7 / 73.2 / 89.8 |
| `pocket_burial` | 59 | 0.000 / 0.080 / 0.685 |
| `packing` | 47 | 81.7 / 199.3 / 237.3 Å² |
| `hbonds` (soft) | 71 | 0 / 4 / 9 |

Six of the eight gates discriminate usefully; 17 designs clear 5 of the 7 hard
gates and every design clears at least none-to-five. Two gates are hard zeros
and must be treated as **not yet calibrated**:

1. **`motif_fidelity` (0/96, best 1.25 Å).** The 1.0 Å all-atom gate is
   stricter than what the generation task itself constrains: `inputs/config.json`
   pins only the *tip* atoms of `E86`, `R122`, `Q136` and `E177`
   (`select_fixed_atoms: TIP`), leaving CB/CG free, and the scorer then measures
   every shared side-chain atom after an independent refold. Either the gate
   should move to a tip-atom subset, or the threshold should be set from a
   campaign that actually optimises full side-chain placement — not lowered to
   fit this reference.
2. **`substrate_pose` (0/96, best 2.25 Å).** This gate is not independent of the
   first: both use the same motif superposition, so a motif that is 2.8 Å off
   drags the substrate with it (observed correlation *r* = 0.54). It cannot
   meaningfully pass while `motif_fidelity` fails, and should be re-read only
   after the motif gate is calibrated.

The `dyad_geometry` split is informative in its own right: of the 96 designs,
12 collapse the dyad below 5.9 Å, 44 land in the retaining window, and 40 open
past 8.9 Å. That is exactly the discrimination this benchmark exists to make,
and it is uncorrelated with simply folding well.

Treat `example/scored/` as the floor a submission should beat, not as a
calibrated pass rate.
