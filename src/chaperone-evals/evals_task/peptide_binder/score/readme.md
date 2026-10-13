# Scoring: peptide-binder eval

Held-out scorer for the `peptide-binder` eval. It takes a **CSV of 96 binder
sequences**, supplies the EIN2 site-1 peptide itself, co-folds every design with
it, re-measures every metric on that fold, applies the gates below, and reports
how many of the 96 pass.

Self-reported metrics are never trusted — every sequence is re-folded here.

## Usage

```bash
# Minimal
./score.py --designs=<submission>.csv --nouse_local_gpus

# With cost telemetry so pass-rate-per-cost is reported
./score.py --designs=<submission>.csv --run_meta=cost.json --nouse_local_gpus
```

| Flag | Required | Description |
|------|----------|-------------|
| `--designs` | yes | CSV (or parquet) with a `sequence` column, 96 rows. |
| `--run_meta` | no | JSON cost telemetry `{gpu_hours, cpu_hours, wallclock_min}`. |

Harbor always passes `trial/controller/run_meta.json`, derived from its guarded
job registry and scheduler accounting; it never trusts a file in the model's
submission. The flag remains optional for standalone non-Harbor scoring.
| `--out` | no | Report JSON (default `score.json`). |
| `--per_design_out` | no | Per-design CSV (default `score_per_design.csv`). |
| `--required_n` | no | Development-only count override. Never changes a threshold. |

Chain assignment is fixed by the scorer, not by the submission: the submitted
sequence is built as chain **A**, and `inputs/target.pdb` supplies the peptide
as chain **B**. Interface, packing, and ddG metrics are all computed A↔B.

## Why plain `af3`, not `af3_with_target_template`

`ppi-pdl1` templates its receptor, because PD-L1 is a folded domain whose
conformation is a fact about the target and not about the design. That
reasoning does not transfer here, and applying it would quietly destroy this
benchmark's headline gate.

The EIN2 site-1 peptide is a 15-residue Gly/Ser/Pro-rich segment. It has no
conformation of its own; it adopts one because a binder holds it. Supplying
`target.pdb` as an AF3 template would place the peptide backbone in the
reference conformation *before the design is consulted*, so the predicted
peptide would sit near the reference whether or not the binder can actually
hold it there — and "chain-B RMSD vs the reference" would then measure the
template, not the design.

The scorer therefore uses **plain `af3` with `load_context_chains`**: chain B
enters as *sequence only*, exactly like chain A, and AF3 must derive the
peptide's conformation from the interaction. The pose gate is then an
independent measurement.

The cost of this choice is honest and worth stating: an unconstrained co-fold of
a disordered peptide is harder than a templated one, so absolute pass rates on
this benchmark are lower than a templated variant would report. That is the
point — the templated variant would be reporting a number that does not mean
what it appears to.

## The peptide-pose gate, and its alignment convention

`peptide_core_rmsd` is the backbone (N, CA, C, O) RMSD between the AF3-predicted
chain B and chain B of `inputs/target.pdb` over the **buried groove epitope**
`B9–B15`, computed after superimposing those residues **on each other**,
residue-for-residue in sequence order.

Two choices need justifying: the alignment frame, and the residue subset.

### Why not "align on the binder"

The plan spec describes the gate as "chain-B backbone RMSD after aligning on the
binder backbone". That convention belongs to a design pipeline, where the
pre-prediction complex and the refold share the *same* binder, so aligning on it
turns the peptide RMSD into a measure of relative placement. A sequence-only
submission has no such shared frame: there is no reference binder, because every
submission brings its own, with a different sequence and a different length.
Aligning on the binder is therefore not merely inconvenient here, it is
undefined, and no amount of care recovers it.

What remains — and what the flexible-target concern is really about — is
conformation. The gate asks whether the design induces *the supplied bound
conformation*, or some other one it happened to prefer. A design that folds the
peptide into a helix, or into a differently-registered strand, fails; a design
that reproduces the supplied conformation passes, wherever in space it sits.
Interface *placement* is not left unmeasured — it is what `contact_ms` and the
two interface-confidence gates are for.

Consistency is enforced, not assumed: the predicted chain B must have the same
residue identities in the same order as the supplied peptide, and every backbone
atom of every residue must be present, or the scorer raises rather than scoring
a partial superposition.

### Why only B9–B15

Because the binder only holds B9–B15. `inputs/prepare_target.py --report_epitope`
prints the per-residue burial of the peptide in the reference complex:

| residue | B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | **B9** | **B10** | **B11** | **B12** | **B13** | **B14** | **B15** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buried fraction | 0.07 | 0.01 | 0.36 | 0.01 | 0.36 | 0.20 | 0.58 | 0.03 | **0.91** | **0.98** | **0.44** | **0.98** | **0.59** | **0.13** | **0.52** |
| closest approach to binder (Å) | 4.4 | 4.8 | 3.8 | 4.9 | 2.7 | 3.8 | 3.8 | 5.2 | **3.0** | **3.4** | **2.9** | **3.3** | **2.7** | **3.4** | **2.5** |

`B9–B15` is the contiguous segment that the groove encloses: it holds every
deeply buried residue (S9, F10, S12, L13) and every one of its residues contacts
the binder within 3.4 Å. `B1–B8` is a solvent-exposed flank — B1/B2 are 1–7 %
buried and never come within 4.4 Å of the binder — so nothing in the design
constrains where it goes.

The residue set is derived from the **reference structure**, not from how the
design population happens to scatter. That distinction matters: picking "the
residues that agree" would be fitting the gate to the data it is supposed to
judge.

Whole-peptide (`peptide_pose_rmsd`) and flank (`peptide_flank_rmsd`, measured
after superimposing the core) RMSDs are computed and written to
`score_per_design.csv` for every design. Nothing is hidden by the narrower gate.

## Gates

A design **passes** only if it clears every hard gate. `ddg` is soft: it gates
`soft_pass`, a strict subset of `pass`.

| Gate | Metric | Threshold | Role | Reference run | Rationale |
|---|---|---|---|---|---|
| `interface_pae` | `score_af3.min_pae_interaction` | ≤ 1.5 | **discriminator** | 24/96 | A genuine peptide groove — a peptide clamped on several faces — predicts far more confidently than a domain–domain interface. |
| `ipsae` | `score_ipsae.ipsae` | ≥ 0.30 | **discriminator** | 27/96 | Chain-pair-normalized, direction-symmetric interface confidence (Park & Cong). Deliberately *not* `min_pae_interface` — see deviation 2. |
| `peptide_core_pose` | `peptide_core_rmsd` (B9–B15) | ≤ 1.5 Å | **discriminator** | 55/96 | The flexible-target check, and the reason this benchmark exists. |
| `binder_fraction_loop` | `score_ss.fraction_loop` | < 0.35 | **discriminator** | 39/96 | Looser than `ppi-pdl1`'s 0.30: a peptide-binding groove is built from loops as well as regular secondary structure, and 0.30 would penalise the correct architecture. Tight but functioning — the reference median is 0.401, so it thins rather than blocks. |
| `contact_ms` | `score_contact_ms` | ≥ 300 Å² | discriminator (weak) | 81/96 | Contact molecular surface, chain A vs chain B. |
| `binder_plddt` | `score_ss.mean_plddt` | ≥ 85 | **guard** | 87/96 | Computed over chain A only, so the disordered peptide cannot drag the number down (or prop it up). Catches designs that did not fold; not expected to separate good binders. |
| `no_bb_clashes` | `score_clash.n_bb_clashes` | == 0 | **guard** | 91/96 | Any non-bonded backbone heavy-atom pair under 1.5 Å is a broken prediction, not a tight one. |
| **soft** `ddg` | `score_ddg.ddg` | ≤ −15 | soft tier | 49/96 | Rosetta PPI ddG (1 relax round, 1 replicate). See deviation 3. |

### Guards vs discriminators

Read the pass counts above with this distinction in mind, because a guard's high
pass rate is *not* evidence that the benchmark is easy.

- **Guards** (`binder_plddt`, `no_bb_clashes`) are expected to pass ~90–100 %.
  They exist to reject pathological submissions — an unfolded chain, a broken
  prediction — so that such designs cannot reach the verdict through a lucky
  interface score. They are supposed to be quiet.
- **Discriminators** (`interface_pae`, `ipsae`, `peptide_core_pose`,
  `binder_fraction_loop`) are what set `n_pass`. On the reference run these pass
  24, 27, 55 and 39 of 96 respectively, and their conjunction — together with
  the weakly-discriminating `contact_ms` — is what produced **1/96**.
- The two pose-related quantities that are *not* gated (`peptide_pose_rmsd`,
  `peptide_flank_rmsd`) are reported per design precisely so a reader can check
  the gate's decomposition rather than take it on trust.

## Deviations from the plan spec

`evals/PLAN.md` supplied the starting thresholds. Three were changed after the
reference campaign, each on measurement rather than taste.

### 1. Pose gate measures the buried epitope, not the whole peptide

**Spec:** chain-B backbone RMSD vs the supplied target ≤ 2.0 Å.
**Current:** backbone RMSD over the buried epitope `B9–B15` ≤ 1.5 Å; whole-peptide
and flank RMSDs reported per design.

The metric definition is unchanged from the original shipped scorer; the
current threshold is tighter. The original 96-design calibration was:

| Quantity (Å) | min | 25 % | median | 75 % | max | original ≤ 2.0 Å |
|---|---|---|---|---|---|---|
| whole peptide `B1–B15` | 1.74 | 4.33 | 6.13 | 7.31 | 9.60 | **1/96** |
| buried epitope `B9–B15` | 0.60 | 1.22 | 1.75 | 3.34 | 4.55 | **55/96** |
| exposed flank `B1–B8`, after superimposing the epitope | 4.84 | 10.47 | 12.56 | 16.29 | 20.04 | — |

Per-residue backbone deviation after whole-peptide superposition shows where the
whole-peptide number comes from (median Å):

| B1 | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | B11 | B12 | B13 | B14 | B15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9.71 | 7.81 | 6.20 | 4.44 | 4.68 | 3.30 | 2.69 | 4.50 | 5.39 | 5.61 | 4.91 | 4.94 | 4.27 | 6.10 | 5.80 |

The mobility is not uniform: it is concentrated in the N-terminal flank, which
the burial table above shows the binder does not touch. A whole-peptide RMSD is
therefore dominated by AF3's sampling of a free peptide tail — it measures the
oracle, not the design. Two further checks confirm the epitope RMSD is the
informative quantity: it correlates only 0.13 with the whole-peptide number
(they are not the same measurement), and only −0.18 with interface PAE, so it
carries information no other gate supplies. Whole-peptide RMSD at 2.0 Å passed
1/96 and, in conjunction, 0/96 — a gate three-fold below its own population
that could report nothing but zero.

This is the same resolution the sibling `rna-binder` benchmark reached: gate the
rigid sub-region the design is actually responsible for, report the rest.

### 2. ipSAE gate uses `ipsae`, not `min_pae_interface`

**Spec:** `ipsae.min_pae_interface ≤ 3.0`. **Shipped:** `ipsae.ipsae ≥ 0.30`.

`ipsae.min_pae_interface` and `af3.min_pae_interaction` are the same number.
Over the 96 reference designs they correlate at **r = 0.9989** with a maximum
absolute difference of **0.05** — both are the minimum PAE over the binder↔target
submatrix, differing only in rounding. As specified, the ipSAE gate at 3.0 could
never fail a design that the AF3 gate at 2.0 passed: it was a strictly weaker
duplicate occupying a slot meant for a second, independent view.

`ipsae.ipsae` is that independent view — the pTM-style, chain-pair-normalized,
direction-symmetric score the block was written to provide. It correlates −0.84
with AF3's min-PAE (related, not identical) and 0.05 with the pose metric.

The threshold is 0.30, not the 0.50 used for `dna-binder`, because ipSAE is
normalized by the number of valid target tokens and a 15-residue target
depresses it: the reference campaign spans 0.03–0.71 with a median of 0.21, and
0.50 would fire on a single design. 0.30 passes 27/96, comparable discriminating
power to the AF3 interface gate. `min_pae_interface` is still reported per
design.

### 3. Soft ddG tightened from −25 to −35

The spec's −25 assumed "a 15-residue interface will not reach the ddG of a domain
interface". The measurement says otherwise: the reference campaign spans −55.0
to −15.6 with a **median of −35.2**, and −25 passes 82/96. A soft tier that
admits 85 % of the field ranks nothing — the failure mode `ppi-pdl1` shipped,
where `ddg < −20` passed 96/96. −35 sits at the measured median and passes
49/96, so it can actually order the hard passes.

This one is explicitly **uncalibrated**: with `n_pass = 1` the reference run
cannot show whether the soft tier separates *among hard passes*. It should be
revisited on the first campaign that produces a double-digit `n_pass`.

## Every gate must be finite

`require_finite_metrics` runs over all eight gate columns before any verdict is
computed. A NaN is an upstream bug — a missing confidence file, a
`secondary_structure` block that saw crystal B-factors instead of pLDDT, a
failed pyrosetta relax — and the scorer raises rather than counting the design
as a failure. Nothing is dropped silently.

## Scoring tiers (higher tier always dominates)

1. **PRIMARY — pass rate (`n_pass` / 96)**: designs clearing every hard gate.
2. **SECONDARY — soft energetic quality (`n_soft_pass`)**: hard passes that also
   clear the ddG gate.
3. **TERTIARY — compute spent**: pass rate per GPU-hour and CPU-hour, from
   `--run_meta`. It never lifts a run above one with a higher `n_pass`.

## Where the reference pose came from, and why the pose gate is hard

`inputs/target.pdb` does not exist upstream. The source example
(`examples/peptide-binder`) supplies the peptide with **NaN coordinates** — a
sequence template, not a structure — so there was no pose to hold designs to.
`inputs/prepare_target.py` produced one: 48 de novo RFD3 designs against the
coordinate-free peptide, each sequenced with MPNN and refolded with AF3, then
the single design that was confident *and* self-consistent was kept and its
AF3 chain B committed.

That run is also the first calibration datum for the pose gate, and it is worth
stating plainly:

| Quantity, over the 48 derivation designs | min | median | max |
|---|---|---|---|
| chain-B RMSD, RFD3 design vs its own AF3 refold | 2.12 | 4.79 | 9.96 |
| chain-B RMSD vs the committed reference pose | 2.49 (excluding the reference itself, which is 0) | 6.21 | 9.76 |

Two things follow. First, a 15-residue Gly/Ser/Pro peptide really does have
several Ångströms of backbone freedom even when the fold around it is
reproduced to ~1 Å — the peptide is the flexible part, exactly as the benchmark
assumes. Second, independent designs place it in genuinely different
conformations (median 6.2 Å apart), so "reproduce *this* pose" is a real
constraint and not a formality. A 2.0 Å gate is correspondingly strict.

## Calibration

The committed reference run scores **1 / 96** (`../example/scored/score.json`,
`submission_sha256` matches `../example/submission.csv`). The single passing
design is a good one on every axis — interface PAE 1.74, ipSAE 0.385, epitope
RMSD 1.05 Å, binder pLDDT 94.5, loop fraction 0.21, contact_ms 348 Å²,
ddG −46.1 — which is the evidence that the gate stack is satisfiable rather
than merely severe.

1/96 is nonetheless a *low* number, and the honest reading is that the reference
pipeline is under-powered rather than that the thresholds are wrong: it applied
no design-side selection that survived (0 of its 144 candidates cleared even its
own filters, so `submission.csv` is the 96 best-ranked candidates, not 96
filtered ones). A pipeline that filtered on epitope-RMSD self-consistency before
submitting should do substantially better; nothing in this scorer prevents that.

Designing against a *fixed* peptide pose is also measurably harder than letting
the generator choose one. The 48-design derivation run, which co-designed the
peptide conformation, reached iptm 0.88 / min-PAE 0.80; the 144-design campaign
against the committed pose topped out at iptm 0.77. That gap is the benchmark's
subject matter, not an artefact.

The benchmark publishes as **In progress** until these thresholds have been
checked against a second campaign — in particular the soft ddG tier, which one
hard pass cannot calibrate.
