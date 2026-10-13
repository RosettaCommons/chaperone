# Ribozyme-design scorer

Status: **implementation complete; reference campaign and fresh calibration
controls pending**. The thresholds below are the starting values supplied by
the measured RZ-01 handoff. Do not retune them from benchmark trials.

## Usage

```bash
score.py --designs=submission.csv --out=score.json \
  --per_design_out=score_per_design.csv --rundir=/scratch/rz01-score \
  --pipelines_repo=/path/to/pipelines --run_meta=run_meta.json --required_n=96
```

| Flag | Meaning |
|---|---|
| `--designs` | Required sequence-only CSV or parquet submission. |
| `--out` | Aggregate `ScoreReport` JSON. |
| `--per_design_out` | Continuous metrics and gate verdicts per design. |
| `--rundir` | Content-addressed held-out work directory. |
| `--pipelines_repo` | Pipelines checkout supplied by Harbor. |
| `--run_meta` | Controller-owned compute telemetry. |
| `--required_n` | Required row count; 96 in production. |

The contract is checked before any held-out compute. It requires uppercase RNA
sequences of 50–90 nt, the six-residue 0-based JSON `motif_map`, the expected
base at every mapped position, native contiguity of `A20-A22` and `B6-B7`, no
duplicate sequence, and pairwise global nucleotide identity at most 70%.

The full 2OEU reference and the 503-sequence novelty database live under
`score/assets/`, so Harbor keeps them with the scorer. Only the six-nucleotide
motif in `inputs/` is copied into either agent workspace.

US-align must be installed into the host Pipelines environment with
`scripts/install_usalign.sh`. T6 additionally requires the licensed
`x3dna-dssr` executable configured through `X3DNA_BIN_DIR` (and `X3DNA_DIR` for
its data directory). It does not invoke `x3dna-snap`. The scorer still does not
need the optional Pipelines `rna` extra: that extra supplies ViennaRNA and
`tmtools`, while this task uses US-align for RNA TM-score and DSSR for T6.

## Chain assignment

The submitter supplies one RNA chain, chain A. The scorer materializes it with
RNA residue names and C3' anchors, then supplies one `MN` CCD ligand identically
to every design. The submitted motif map becomes `diffused_index_map` metadata;
all motif-aware measurements resolve the 2OEU reference IDs through that map.
Submitted coordinates, file paths, and self-reported metrics are rejected.

## Oracle and metric blocks

Plain AF3 refolds each RNA-plus-Mn input. `af3_with_target_template` is not used:
its target-template path is protein-specific and would not provide a valid RNA
reference. The refold is then measured with:

- `reference_motif_rmsd`, against the six-residue 2OEU crystal motif;
- `motif_geometry`, for catalytic distances and the in-line attack angle;
- `motif_confidence`, for motif/scaffold pLDDT and PAE;
- `tm_score_rna`, for RNA-calibrated global similarity to the 63-nt native;
- `sequence_identity_gate`, for scaffold-only global identity to 503 natural
  references and within-submission identity;
- `dssr_motif_base_pairing`, for report-only T6 pairing and native DSSR
  classification retention at each mapped catalytic residue.

The Mn atom is deliberately excluded from motif RMSD (`ligands=[]`). AF3 places
it about 36 Å from the crystallographic site even for the native sequence, which
changes the native control from 0.49 Å to 3.08 Å. The two metal distances are
reported but never gated. Protein TM-score and protein `n_chainbreaks` are not
used. T2 (`rna_thermo`) and T3 (`rna_ss_f1_mcc`) are inapplicable because this
six-residue motif contains no internal canonical base pair.

## Pass criteria

All five gates are hard and conjunctive.

| Criterion | Metric column | Threshold | Role and rationale |
|---|---|---:|---|
| `motif_rmsd` | `score_motif_rmsd.motif_rmsd` | ≤ 2.0 Å | Discriminator; 72/7,929 refolds from the campaign clear it. |
| `inline_attack_dist` | `score_geometry.inline_attack_dist` | ≤ 4.0 Å | Discriminator; a near attack distance is mechanistically necessary. |
| `inline_attack_angle` | `score_geometry.inline_attack_angle` | ≥ 140° | Discriminator; enforces an in-line 2'-O–P–O5' geometry. |
| `plddt` | `score_af3.plddt` | ≥ 70 | Guard against an RNA that did not fold coherently. |
| `novelty` | `score_novelty.max_identity` | < 0.563 | Hard memorization gate; measured shuffled-null maximum for the shipped reference set. |

The three catalytic gates are one calibrated unit: together they admit 23 of
7,929 campaign refolds (out of a set of 10,000 designs). The pLDDT gate is a guard and should pass nearly every
healthy refold; a population-wide drop is evidence of an oracle or input fault,
not increased task difficulty. Novelty excludes the mapped motif before global
Needleman–Wunsch alignment and takes the maximum over references. The guide's
0.40 placeholder is not used because it is below the measured 0.534 shuffled
null and the 0.544 design median.

Reported but not gated are the general-base and general-acid distances, both
metal distances, motif/scaffold confidence, RNA TM-score, and maximum identity
within the submission. T6 is also report-only. It emits `paired`, partner count,
native partner-type match, partner-in-motif, native Leontis–Westhof match, and
full native DSSR-classification match for each of `A20`, `A21`, `A22`, `A36`,
`B6`, and `B7`, plus fractions/counts across the six. The underlying pair
details retain DSSR's `name`, `Saenger`, `LW`, and `DSSR` fields and retain all
partners; native 2OEU C17 (`B6`) has both A37–C17 (`tHW`) and A38–C17 (`cWW`)
annotations and is not reduced to whichever appears first.
Classification strings in the per-residue vector are oriented motif-to-partner;
for example, DSSR's raw A37-to-C17 `tHW` annotation is reported from C17's
perspective as `tWH`.
Each residue's partner IDs, partner bases, and four DSSR classification fields
are serialized as deterministic JSON in
`t6.pair_classifications_<reference-id>` columns of `score_per_design.csv`.
The same vectors are keyed by design ID under
`t6.pair_classifications_by_design` in `score.json`; native DSSR details are
stored once under `t6.native_pair_details`.

The aggregate report also records a motif-reconstruction-only alternate pass
rate under `motif_rmsd_only_pass`: every design with motif RMSD below 2.0 Å,
ignoring the other four gates, contributes to its count and rate and is listed
by design ID. This fallback remains visible when no design clears all five
primary gates; it does not replace the primary conjunctive pass definition.

RNA TM-score is diagnostic: the campaign median is
0.225, close to the RNA random-pair baseline of 0.21 and below the same-family
boundary of 0.45. The 23 catalytic hits have median 0.203, so a TM-score gate
would reject the intended success mode.

## Calibration controls

The handoff records these existing measurements:

| Control | Recorded result | Required interpretation |
|---|---|---|
| Oracle ceiling, native 2OEU refold | motif RMSD 0.49 Å; in-line angle 164.3°; pLDDT 95.5; prior Biotite T6 5/6 | Structural values must reproduce before publication. DSSR T6 must be recorded afresh rather than assumed identical to the prior detector. |
| Memorization, native submitted as a design | Structural gates pass | Novelty must reject it. |
| Floor, dinucleotide-shuffled and random RNA | Metrics bottom out | RNA TM-score near 0.4 would indicate wrong normalization. |
| Naive motif plus A-form stems | Not supplied in the handoff | Must be run and recorded before calibration is complete. |

These controls have not been rerun from this checkout. The scorer and task
should therefore publish as **In progress** until the oracle ceiling,
memorization, floor, and naive baseline are reproduced with the final Pipelines
environment. In particular, the campaign's T6 frequencies were measured with
Biotite and are not calibration numbers for DSSR; DSSR T6 needs its own control
and campaign distributions before anyone interprets differences quantitatively.

## Reference run

The measured 10,000-design campaign produced 10,000 diffusions, 7,929 designs
below 1 Å insertion RMSD, 72 refolds below 2 Å motif RMSD, and 23 designs
passing the catalytic triple gate. Its best design measured 1.04 Å motif RMSD,
170.4° in-line angle, and 92.7 pLDDT.

`example/example_pipeline.py` is the runnable ranked reference generator. A
committed 96-design `submission.csv` and its scored artifacts are still
pending; consequently no per-criterion table for the *selected 96* is claimed
here. Once run, commit `example/submission.csv`, `example/top_design.cif`,
`example/scored/score.json`, and `example/scored/score_per_design.csv` and add
their exact per-gate counts to this section without silently changing a gate.

## Deviations from the original guide

- Only RZ-01 is implemented. Twister, pistol, and glmS are out of scope.
- The guide's T2/T3 gates are omitted because the target makes both metrics
  degenerate, not because of a software failure.
- TM-score_RNA is reported rather than gated because measured catalytic hits do
  not recover the native global fold.
- Metal contacts are reported rather than gated because the AF3 oracle fails
  the native metal-placement control.
- T6 uses the dedicated DSSR motif-pairing block instead of Biotite geometry so
  the scorer retains trusted `name`, `Saenger`, `LW`, and `DSSR`
  classifications and multiple partners per residue. This changes the T6
  detector, not any pass gate.
- The novelty cutoff is 0.563 rather than the guide's placeholder 0.40, using
  the measured null maximum for the exact 503-sequence reference set shipped
  with this task.
