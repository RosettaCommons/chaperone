# Scoring rubric: nanoparticle-c3 eval

Held-out scorer for the C3 cyclic homo-oligomer eval. It takes a **CSV of 96
protomer sequences**, replicates each one into **three identical chains A/B/C**,
folds the trimer with AF3, and gates on the geometry of the *prediction*.

Two things make this benchmark different from every other one in the suite, and
both are deliberate.

## 1. The symmetry gates are the headline

Nothing else in `evals/` measures **global geometry**. Every other benchmark
scores a *local* property — an interface area, a clash count, a contact PAE, an
interface energy — and a design can satisfy all of them while being globally
wrong. Here the primary question is architectural: are the three protomers
actually one fold, related by one 120° rotation about one axis?

This is a real test rather than a tautology because **nothing in the scoring
path is symmetry-constrained**. The submission is a sequence. AF3 folds three
copies of it with no symmetry restraint applied, and is free to return a
lopsided dimer-plus-monomer, a domain-swapped tangle, or a helical stack. The
`symmetry` block then fits the actual inter-protomer transforms. A backbone
generator that produced a perfect C3 backbone earns nothing here if the sequence
it was given does not re-encode that assembly.

| Metric | Threshold | Rationale |
|---|---|---|
| `symmetry.angle_deviation_deg` | ≤ 5.0° | The fitted rotation between protomers must be 120°. 5° is roughly the spread seen between AF3 replicas of a genuinely symmetric assembly; beyond it the ring is measurably open or over-closed. |
| `symmetry.subunit_rmsd` | ≤ 1.0 Å | Backbone RMSD of each protomer onto the first. A homo-oligomer whose chains adopt different folds is not a homo-oligomer, whatever its interface scores say. |

`axis_rmsd` (collinearity of the fitted rotation axes) is **reported but not
gated**: on a three-protomer assembly a tilted axis also drives
`angle_deviation_deg` up (10.2° on the block's synthetic tilted-axis fixture),
so gating it as well would double-count one failure mode.

`max_screw_translation` (rise along the axis) is **reported but not gated**, and
this one deserves an explicit caveat rather than a hand-wave. It is a genuine
blind spot of the two gates *in principle*: a helical stack of three identical
protomers, each rotated exactly 120° but translated along the axis, scores
`angle_deviation_deg = 0` and `subunit_rmsd = 0` and would pass both. The
`symmetry` block's own `test_screw_translation_is_detected` fixture is exactly
that construction.

It is not gated because on the reference run it never fires independently.
Across all 128 predictions: 29 had `max_screw_translation > 2 Å`, and **every
one of them already failed `angle_deviation_deg` or `subunit_rmsd`**. Among the
76 predictions that passed both gates, the largest screw translation was
**0.455 Å**. Adding a third gate would therefore have rejected nothing while
adding a threshold nobody had calibrated. Revisit this if a future submission
shows high screw translation alongside clean angle and RMSD — that combination
is the signature of a designed helix being passed off as a ring.

## 2. `iptm ≥ 0.85` is higher than any binder case in the suite

`ppi-pdl1` uses no ipTM gate at all and leans on interface PAE; `dna-binder`'s
soft ipTM gate is 0.60; `antibody-cdr` deliberately relaxes its interface
confidence because antibody interfaces predict badly. This benchmark goes the
other way on purpose.

A designed homo-oligomer is the *easiest* case AF3 ever sees at an interface:
three copies of one chain, a symmetric contact the network has seen thousands of
times in the PDB, and no cross-species or cross-family generalisation required.
If the oracle is unsure about a designed homotrimer, the honest reading is that
there is no assembly — not that the prediction task was hard. 0.85 is where
confident, well-packed designed oligomers sit; a lower bar would admit
predictions that are really three independently folded monomers placed near each
other, which the symmetry gates alone would not always catch.

## All criteria

| Gate | Metric | Threshold | Rationale |
|---|---|---|---|
| **symmetry** | `score_symmetry.angle_deviation_deg` | ≤ 5.0° | see above |
| **symmetry** | `score_symmetry.subunit_rmsd` | ≤ 1.0 Å | see above |
| assembly confidence | `score_af3.iptm` | ≥ 0.85 | see above |
| fold | `score_af3.plddt` | ≥ 85 | a nanoparticle building block has to be a rigid, well-ordered domain; 85 is the standard "confidently folded" bar |
| interface | `score_contact_ms` (A vs B) | ≥ 600 Å² | per representative A↔B interface |
| monomer quality | `score_secondary_structure.fraction_loop` | < 0.35 | looser than `ppi-pdl1`'s 0.30: a C3 protomer needs the loops that turn its interfaces, whereas a mini-binder does not |
| **compactness** | `protomer.ca_radius_of_gyration` | ≤ 14 Å | Cα radius of gyration of predicted chain A. Prevents a long coiled-coil rod from satisfying the local interface and C3 gates without forming a compact tertiary building block. |
| sterics | `score_clash.n_bb_clashes` | == 0 | |
| **soft** | `score_ddg.ddg` (A/B) | < −35 | an oligomer interface that only just holds is not a nanoparticle component |

Column names carry the `score_` prefix of the block instance that produced them.
`contact_ms` returns a bare float, so its column is the instance name with no
sub-key.

### Which interface is scored

C3 has three protomer–protomer interfaces, so `contact_ms` and `ppi_ddg` score
**A|B** as the representative. "They are equivalent by symmetry" is an
assumption, not a fact about a *prediction*, so it was checked against the
reference run rather than asserted.

Measuring the minimum A↔B, A↔C and B↔C PAE separately for all 96 predictions,
the three interfaces are frequently **not** equivalent: the spread between the
best and worst pair exceeds 1 Å in 37/96, reaching 12.4 Å. But that
inequivalence lives almost entirely in the designs the gates already reject:

| Group | n | median spread | max spread | spread > 1 Å |
|---|---|---|---|---|
| passes symmetry gates | 56 | **0.000 Å** | 2.8 Å | 6 |
| fails symmetry gates | 40 | 2.600 Å | 12.4 Å | 31 |

Of the 6 symmetry-passing designs with a >1 Å spread, **0** also clear the ipTM
gate. So among designs that survive scoring at all, the three interfaces are
interchangeable and scoring A|B is sound; among designs where the interfaces
genuinely differ, the assembly has already failed. The shortcut is justified by
the data, not by the point group.

Note that this benchmark does **not** gate on `af3.min_pae_interaction`. That
key is the minimum over *all* off-diagonal chain pairs, i.e. the **best** of the
three interfaces, which is exactly the optimism bias the table above quantifies.
Assembly confidence is gated on `af3.iptm`, a whole-assembly quantity, instead.

### Reported, never gated

- `protomer_length` — the prompt asks for 60–90 residues per protomer. It is
  reported so a run that drifted outside the intended size is visible, but it is
  not a pass/fail criterion: the scientific gates above are what decide whether
  a design is a C3 assembly.
- `score_symmetry.axis_rmsd` — see above.

## Contract

Validated before any GPU is touched, via
`SequenceSubmissionSpec(chains=("A",), oligomer_copies=3)`:

- exactly 96 rows;
- one `sequence` column, uppercase canonical
  amino acids only;
- optional unique `design_id`;
- **no `sequence_B` / `sequence_C`.** The replication into three chains is the
  scorer's job. A submission that supplies three chains is asserting a
  hetero-trimer that happens to be identical — a different molecule, and a
  contract that could silently accept three *different* chains.

A contract failure writes the canonical failure report and spends no held-out
compute.

## Usage

```bash
# Production
./score.py --designs=<submission>.csv --nouse_local_gpus

# With cost telemetry so pass-rate-per-cost is reported
./score.py --designs=<submission>.csv --run_meta=cost.json --nouse_local_gpus
```

| Flag | Required | Description |
|------|----------|-------------|
| `--designs` | yes | CSV (preferred) or parquet with a `sequence` column, 96 rows. |
| `--run_meta` | no | JSON with cost telemetry `{gpu_hours, cpu_hours, wallclock_min}`. |

Harbor always passes `trial/controller/run_meta.json`, derived from its guarded
job registry and scheduler accounting; it never trusts a file in the model's
submission. The flag remains optional for standalone non-Harbor scoring.
| `--out` | no | Report JSON path (default `score.json`). |
| `--per_design_out` | no | Per-design CSV path (default `score_per_design.csv`). |
| `--required_n` | no | Development-only count override. Reduces the contract to N rows; it changes **no** threshold and **no** scoring behaviour. |

Run directories are content-addressed under `score_runs/<sha256[:16]>/`, so
re-scoring the same submission reuses its cached compute and two different
submissions can never share one.

## Compute

One SLURM GPU. AF3 folds a ~210-residue trimer, roughly three times the token
count of a mini-binder complex, so the oracle takes `gres=gpu:large:1` with
64 GB and `rows_per_job` set above the row count so all 96 predictions run as a
single chunk on a single GPU. The metric blocks are CPU-only.

## Calibration status

**In progress.** The original thresholds came from `evals/PLAN.md`. The
compactness gate was added after a baseline exposed a topology loophole: 96
sequence-diverse 70-residue coiled coils were predicted as effectively the same
extended triple-helical rod and 92 passed the original gates. The prior
`../example/scored/` artifact predates this gate and is retained as a historical
calibration record rather than a current golden score.

The 18 Å cutoff is calibrated against held-out AF3 chain-A Cα coordinates. The
compact reference has radius of gyration **14.08 Å**. The pathological low arm
spans **29.76–30.39 Å**, so all 96 fail. Medium and high have medians of **14.95
Å** and **14.23 Å**; 71/96 and 60/96 respectively pass compactness, including
the medium arm's one original overall pass and all five of the high arm's
original overall passes. This leaves a 3.9 Å margin above the reference while
cleanly excluding the rod topology.

| Criterion | Passes | Observed range (min → median → max) | Role |
|---|---|---|---|
| `angle_deviation` | **56 / 96** | 0.011 → 0.910 → 106.6° | **discriminator** |
| `subunit_rmsd` | **80 / 96** | 0.011 → 0.185 → 9.31 Å | **discriminator** |
| `iptm` | **33 / 96** | 0.11 → 0.22 → 0.91 | **discriminator** (tightest) |
| `plddt` | **28 / 96** | 41.6 → 63.3 → 93.5 | **discriminator** (tightest) |
| `contact_ms` | **72 / 96** | 70.6 → 479.8 → 1085.1 Å² | discriminator |
| `fraction_loop` | 92 / 96 | 0.057 → 0.157 → 0.457 | **guard** |
| `compactness` | not present in historical artifact | reference 14.08 Å; rod baseline 29.76 → 30.16 → 30.39 Å | **topology guard** |
| `clashes` | 92 / 96 | 0 → 0 → 2 | **guard** |
| `ddg` (soft) | 95 / 96 | −229.2 → −108.3 → −29.5 | **guard** |

Reading this honestly:

- **The symmetry gates work as intended.** `angle_deviation_deg` spans four
  orders of magnitude across the 96 (0.011° to 106.6°) and rejects 40 of them.
  This is the discriminator it was designed to be, not a metric that returns
  ~0 for everything.
- **`fraction_loop`, `clashes` and the soft `ddg` are guards, not
  discriminators.** They pass 92–95 of 96 and are there to catch a pathological
  submission, not to rank good ones. That is the intended role and they should
  not be tightened to manufacture selectivity.
- **ipTM and pLDDT are currently the binding constraints** (33 and 28 of 96),
  tighter than the symmetry gates. This is expected: the reference pipeline
  ranks by ipTM but only had 128 predictions to draw 96 from, so it had to
  include low-confidence designs to reach the required count. A stronger
  submission would move these first.

Any future change to a threshold must be justified from this distribution and
recorded here.
