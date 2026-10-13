# `dna-binder` scoring

Held-out scorer for de novo major-groove dsDNA recognition. Every threshold
below, and every deviation from the starting spec in `evals/PLAN.md §1`, is
documented here.

## Protocol

1. Validate the sequence-only submission (`sequence`, 96 unique rows) before
   any GPU is touched.
2. Load `inputs/target.pdb` chains **B** (sense, `CGCGAATTCGCG`) and **C**
   (antisense) via `load_context_chains`, and attach them to every design with
   `sequence_submission_to_handler(context=...)`. The submitted sequence
   becomes chain **A**.
3. Co-fold the three-chain assembly with **plain `af3`**
   (`diffusion_batch_size=1`).
4. Measure on the AF3 output: `ipsae`, `dssr_snap_metrics`, `contact_ms`,
   `rmsd` (nucleotide selections) and `clash`.
5. Require every gated metric to be finite for every row
   (`require_finite_metrics`), then apply the criteria below.

### Why plain `af3` and not `af3_with_target_template`

`af3_with_target_template` injects its template only into `entry["protein"]`
and silently skips a DNA or RNA `target_chain`. Used here it would fold the
duplex with no template while reporting that the target was templated. The
duplex is instead pinned explicitly by the `dna_preserved` RMSD gate, which is
the honest version of the same requirement.

`_sequences_to_cif` was verified to preserve `chain_type` through
`to_cif_file`: the round-tripped context CIF reports `ChainType.DNA` for B and
C and `ChainType.POLYPEPTIDE_L` for A, so `create_af3_json_from_design` emits
proper `{"dna": ...}` entities. No toolkit change was needed.

## Criteria

| Gate | Metric column | Threshold | Hard |
|---|---|---|---|
| `plddt` | `score_af3.plddt` | ≥ 80 | yes |
| `interface_pae` | `score_ipsae.min_pae_interface` | ≤ 5.0 Å | yes |
| `ipsae` | `score_ipsae.ipsae` | ≥ 0.30 | yes |
| `major_hbond` | `score_dssr.Major_hbond` | ≥ 3 | yes |
| `total_hbond` | `score_dssr.Total_hbond` | ≥ 8 | yes |
| `nt_aa` | `score_dssr.nt_aa` | ≥ 10 | yes |
| `contact_ms` | `score_contact_ms` | ≥ 400 Å² | yes |
| `dna_preserved` | `score_rmsd.nucleotide_bkbn_align.nucleotide_rmsd` | ≤ 2.0 Å | yes |
| `no_bb_clash` | `score_clash.n_bb_clashes` | == 0 | yes |
| `iptm` | `score_af3.iptm` | ≥ 0.60 | **soft** |

### Rationale

- **`plddt` ≥ 80** — the design must be a real fold before its interface means
  anything. Same bar as the other binder benchmarks.
- **`interface_pae` ≤ 5.0 Å** — protein–DNA interfaces predict less
  confidently than protein–protein ones, so the 4.0 Å used for `ppi-pdl1` is
  too tight here.
- **`ipsae` ≥ 0.30** — chain-pair length-normalized interface confidence,
  reported as the *worst* of the A↔B and A↔C pairs, so a binder that grips only
  one strand cannot pass on the other's score. `pae_cutoff=10.0` (Park & Cong
  default). Distinct from `interface_pae`, which is the single best token pair:
  ipSAE integrates confidence across the whole interface. Lowered from the
  spec's 0.50 for a target-size normalization artefact, see below.
- **`Major_hbond` ≥ 3** — hydrogen bonds to major-groove base edges (DA N6/N7,
  DG O6/N7, DC N4, DT O4) are *the* signature of sequence-specific readout.
  Backbone-phosphate contacts are sequence-agnostic and cannot substitute.
  Set from the native anchor below rather than from the spec's 4.
- **`Total_hbond` ≥ 8** — total protein↔DNA polar contacts, so a design cannot
  pass on four base hydrogen bonds and nothing else.
- **`nt_aa` ≥ 10** — SNAP's nucleotide↔amino-acid interaction count; a
  contact-count floor independent of the hydrogen-bond geometry. Lowered from
  the spec's 12, see below.
- **`contact_ms` ≥ 400 Å²** — protein–DNA interfaces are large; this is roughly
  double a peptide-groove interface. `binder_chains=["A"]`,
  `target_chains=["B","C"]`.
- **`dna_preserved` ≤ 2.0 Å** — all-nucleotide RMSD of the co-folded duplex
  against the supplied coordinates, superimposed on the nucleotide backbone.
  Without it a design can "win" by having AF3 bend or melt the duplex around
  it. Computed by `rmsd` with `which=[0, -1]`: index 0 is the scorer-built
  input CIF, which carries the supplied duplex coordinates verbatim.
- **`no_bb_clash` == 0** — `clash` filters to amino acids, so this is a
  protein-internal steric check only; protein↔DNA sterics are covered
  implicitly by `contact_ms` and the AF3 fold.
- **`iptm` ≥ 0.60 (soft)** — a global assembly-confidence read-out. Soft
  because ipTM on a three-chain system with a rigid duplex is noisier than the
  per-pair measures above.

## The native anchor

Guessing a contact-count threshold is unnecessary when the PDB contains
hundreds of genuine protein–dsDNA complexes. Nine were read through *this
scorer's own metric blocks* — each reduced to one protein chain plus its two
DNA strands, relabelled A/B/C so the topology matches a submission exactly —
to establish what a real complex actually scores. (`ipsae`, `plddt` and the
nucleotide RMSD gate need an AF3 prediction and cannot be anchored this way.)

| PDB | complex | `Major_hbond` | `Minor_hbond` | `Total_hbond` | `nt_aa` | `contact_ms` |
|---|---|---|---|---|---|---|
| 1AAY | Zif268 zinc fingers | 13 | 0 | 23 | 42 | 670 |
| 1ZAA | Zif268 zinc fingers | 16 | 0 | 26 | 43 | 658 |
| 1A1F | Zif268 variant | 16 | 0 | 22 | 42 | 664 |
| 1LMB | λ repressor | 12 | 0 | 22 | 23 | 479 |
| 1PER | Trp repressor | 5 | 0 | 14 | 21 | 430 |
| 1HDD | engrailed homeodomain | 3 | 2 | 16 | 16 | 441 |
| 1FJL | Prd paired domain | 2 | 3 | 20 | 15 | 500 |
| 1MDY | MyoD bHLH | 2 | 0 | 10 | 9 | 300 |
| *6CRO* | *Cro repressor* | *20* | *2* | *62* | *62* | *0* | 

6CRO is **excluded**: every atom in the deposited entry appears twice (36
atoms on a residue with 18 distinct atom names), which doubles all DSSR counts
and makes `contact_ms` fail outright on coincident atoms. Statistics below use
the eight clean entries.

|  | min | lower quartile | median | max |
|---|---|---|---|---|
| `Major_hbond` | 2 | 2.75 | 8.5 | 16 |
| `Total_hbond` | 10 | 15.5 | 21 | 26 |
| `nt_aa` | 9 | 15.25 | 22 | 43 |
| `contact_ms` | 300 | 437 | 490 | 670 |

Every hard contact threshold is now set at or below the native lower quartile,
so roughly three quarters of genuine native complexes clear it, while the
non-binders in the reference campaign — which score flat zero on all four —
fail by an enormous margin:

| Gate | Threshold | natives passing |
|---|---|---|
| `Major_hbond` ≥ 3 | native LQ 2.75 | 6 / 8 |
| `Total_hbond` ≥ 8 | native LQ 15.5 | 8 / 8 |
| `nt_aa` ≥ 10 | native LQ 15.25 | 7 / 8 |
| `contact_ms` ≥ 400 | native LQ 437 | 7 / 8 |

Note the natives split by structural class: zinc fingers (1AAY/1ZAA/1A1F) read
13–16 major-groove hydrogen bonds, while helix-turn-helix and bHLH proteins
read 2–5. The benchmark's own reference designs are helix-turn-helix folds, so
the HTH end of that range is the relevant comparison — which is exactly why the
`Major_hbond` gate cannot sit above it.

Reproduce with `example/rundir`-independent inputs from the PDB mirror; the
preparation and scoring script is not committed because it is calibration
scaffolding, not part of the benchmark.

## Deviations from the starting spec

Four, all forced by measurement rather than taste.

### 1. `interface_pae` uses `ipsae.min_pae_interface`

**`interface_pae` uses `ipsae.min_pae_interface`, not
`af3.min_pae_interaction`.** `evals/PLAN.md §1` specified
`af3.min_pae_interaction ≤ 5.0`. That metric is computed as the minimum over
**all** off-diagonal entries of AF3's `chain_pair_pae_min`
(`AlphaFold3Block._compute_min_pae_interaction`). This assembly has three
chains, and the B↔C duplex pair is always confidently predicted, so the metric
reports the DNA:DNA pair and never the interface under test.

Evidence, from the 12 designs of the prior protein–DNA campaign in
`examples/dna-binder/base/rundir`:

| design | `af3_2.iptm` | `af3_2.min_pae_interaction` | protein↔DNA `chain_pair_pae_min` |
|---|---|---|---|
| kmer4 (real binder) | 0.89 | 1.04 | 1.12 / 1.31 |
| 11 others (non-binders) | 0.07–0.09 | 1.48–1.90 | 23.8–30.3 |

The gate as specified would have passed all 12 designs, including 11 that make
zero contacts with the DNA (`Total_hbond = 0`, `nt_aa = 0`).
`ipsae.min_pae_interface` is the same physical quantity restricted to the
A↔{B,C} submatrix, so it reads 1.12 for the real binder and ≳24 for the
non-binders — the threshold value of 5.0 Å is kept unchanged.
`af3.min_pae_interaction` is still reported in `score_per_design.csv` and in
the report's `metric_summaries` for continuity; it is simply not gated.

The reference run of 96 designs settles this quantitatively. Across those 96:

| | range | ≤ 5.0 Å |
|---|---|---|
| `af3.min_pae_interaction` | 1.15 – 1.80 Å | **96 / 96** |
| `ipsae.min_pae_interface` | 1.90 – 30.1 Å | 60 / 96 |

Spearman/Pearson agreement between the two is only r = 0.72 with a maximum
absolute difference of **28.7 Å**. On a two-chain binder benchmark these two
metrics are interchangeable (the `peptide-binder` benchmark measures r = 0.999,
max difference 0.05). They diverge here purely because a third chain is
present: `_compute_min_pae_interaction` minimises over all chain pairs, and the
B↔C duplex pair is always ~1.2–1.8 Å. Any benchmark whose assembly has more
than two chains — a duplex, a substrate plus a metal, a homo-oligomer — must
not gate on `af3.min_pae_interaction`.

### 2. `nt_aa` lowered from ≥ 12 to ≥ 10

The `kmer4` design above is the only verified positive control available for
this design class: a genuine major-groove binder that AF3 refolds at ipTM 0.89
with 14 protein–DNA hydrogen bonds. Running the exact scoring stack of this
benchmark over it gives:

| Gate | Value | Verdict at spec threshold |
|---|---|---|
| `plddt` | 92.2 | pass (≥ 80) |
| `iptm` | 0.89 | pass (≥ 0.60) |
| `ipsae` | 0.777 | pass (≥ 0.50) |
| `min_pae_interface` | 1.10 Å | pass (≤ 5.0) |
| `Major_hbond` | 4 | pass, exactly at the bar (≥ 4) |
| `Total_hbond` | 14 | pass (≥ 8) |
| `nt_aa` | 11 | **fail (≥ 12)** |
| `contact_ms` (A vs B,C) | 418.7 Å² | pass, close to the bar (≥ 400) |
| nucleotide RMSD | 0.96 Å | pass (≤ 2.0) |
| `n_bb_clashes` | 0 | pass (== 0) |

`nt_aa ≥ 12` therefore rejects the one design known to be right, on a
one-count margin, while every other gate accepts it. That is a mis-set
threshold, not a discriminating one, so it is lowered to **10** — still a
substantial contact requirement, and still well above the 0 that all eleven
non-binders in the same campaign score. It is the only threshold moved for this
reason.

`contact_ms ≥ 400` is left at the spec value: the positive control clears it,
if only by 19 Å², and it sits just below the native lower quartile of 437 Å²
with 7 of 8 natives clearing it.

The native anchor independently corroborates this change: `nt_aa ≥ 12` would
reject 1HDD (16), 1FJL (15) and 1MDY (9) as well — three of eight genuine
complexes, including the engrailed homeodomain.

### 3. `Major_hbond` lowered from ≥ 4 to ≥ 3

The native anchor is unambiguous here. At the spec's 4, the gate rejects the
**engrailed homeodomain** (1HDD, 3 major-groove hydrogen bonds), the Prd paired
domain (1FJL, 2) and MyoD (1MDY, 2) — three of eight native complexes, and
precisely the helix-turn-helix structural class the benchmark's own reference
designs fall into. A gate that rejects the canonical major-groove reader is
measuring the wrong thing.

At 3 the gate admits 6 of 8 natives, sits at the native lower quartile (2.75),
still admits the AF3-modelled positive control (4), and still rejects outright
every design that makes no base-edge contact at all — which is what all eleven
non-binders of the reference campaign, and the large majority of the committed
reference submission, actually do.

Crystal structures carry no hydrogens, so DSSR's distance-only hydrogen-bond
detection is if anything conservative on them; the native numbers are a lower
bound on what a clean AF3 model of the same interaction would show. That argues
against lowering the gate any further than the native lower quartile.

### 4. `ipsae` lowered from ≥ 0.50 to ≥ 0.30

ipSAE's pTM-style scaling uses `d0 = max(1, 1.24·(L−15)^⅓ − 1.8)` over the
number of *target* tokens that pair below the PAE cutoff. Each DNA strand here
is 12 nucleotides, always under the L = 27 floor, so `d0` collapses to 1.0 and
the score is structurally compressed: the practical ceiling on this target is
not 1.0. The verified positive control — a genuine major-groove binder on a
12-bp duplex — reaches **0.777**, and the best design in the reference run
reaches 0.534. A 0.50 gate is therefore ~64 % of achievable on this target,
materially stricter than the same number would be on a normal-sized target.

The consequence was visible in the reference run as the exact single-gate
pathology that made `biotin-binder` report 0/96. At 0.50 only 2 of 96 designs
cleared it, **7 designs failed nothing else**, and dropping the gate alone took
the pass count from 1 to 8 — no other gate moved the total at all. Those seven
were not marginal: pLDDT 80–85, ipTM 0.59–0.76, interface PAE 2.0–3.3 Å, and
contact counts (`nt_aa` 10–24, `contact_ms` 528–774 Å²) at or above the native
anchor's median.

`ipsae` is also heavily redundant with gates already present — Spearman ρ = 0.96
against `af3.iptm` and ρ = −0.94 against `interface_pae` across the 96 — so at
0.50 it was acting as a third, far stricter copy of one signal rather than as
independent evidence.

At **0.30** the gate admits 12 of 96, sits at roughly 40 % of the measured
ceiling, still rejects every design with no coherent interface, and no longer
single-handedly determines the result: the reference run now passes 3, and
dropping any single gate changes that to at most 8.

### Gates that did not need moving

`dna_preserved` (≤ 2.0 Å) passes 96/96 on the reference run — observed range
0.74–1.96 Å, median 1.15 Å. It is not currently a discriminator, and that is
the expected behaviour: a 12-bp B-form duplex is rigid, and AF3 does not deform
it. The gate is a guard rail against a specific cheat (winning by melting the
duplex) that this submission never attempts, not a quality filter. Its maximum
sits just inside the threshold, so it would begin to bite at 1.5 Å; it is left
at 2.0 Å deliberately so that it fires only on real deformation.

`no_bb_clash` likewise passes 96/96. Both are retained because their cost is
zero and their absence would silently admit a pathological design.

## Calibration status

**In progress.** The four contact thresholds (`Major_hbond`, `Total_hbond`,
`nt_aa`, `contact_ms`) are calibrated against the eight-entry native anchor
above plus one verified AF3-modelled positive control (`kmer4`). The
confidence thresholds (`plddt`, `ipsae`, `interface_pae`, `iptm`) and the
`dna_preserved` RMSD gate are **not** anchored — they require predictions
rather than crystal structures, and the only prediction-side evidence available
is the single positive control. The committed reference submission in
`../example/scored/` is a draft from an untuned reference pipeline and is not
expected to pass. Until a real reference campaign exists, the pass count this
scorer reports is not a scoreable contract.

### What the reference submission scores

`../example/scored/` — **3 / 96 hard pass, 3 / 96 soft pass.**

| gate | passing |
|---|---|
| `dna_preserved` | 96 |
| `no_bb_clash` | 96 |
| `total_hbond` | 64 |
| `interface_pae` | 60 |
| `contact_ms` | 51 |
| `nt_aa` | 38 |
| `major_hbond` | 33 |
| `iptm` *(soft)* | 24 |
| `plddt` | 19 |
| `ipsae` | 12 |

Seven further designs fail exactly one gate (five on `ipsae`, two on
`major_hbond`), and removing any single gate raises the pass count to at most
8 — so no gate is individually deciding the outcome. The three passers are
credible: pLDDT 84–86, ipTM 0.68–0.77, interface PAE 1.9–2.7 Å, 3–9
major-groove hydrogen bonds, `nt_aa` 10–31, `contact_ms` 528–810 Å², duplex
RMSD 0.95–1.26 Å. The best of them (`dna_0018`) buries more surface than any
entry in the native anchor.

The binding constraint on this submission is fold quality (`plddt`, 19/96),
not DNA recognition — an untuned single-round RFD3 → LigandMPNN → AF3 pipeline
mostly produces designs AF3 is not confident in.

## Environment requirement

`dssr_snap_metrics` shells out to `x3dna-dssr` and `x3dna-snap` (3DNA v2.3).
Their locations come from the `X3DNA_BIN_DIR` / `X3DNA_DIR` settings. If SNAP
is unavailable, `nt_aa` is NaN; `require_finite_metrics` turns that into a
loud failure of the scoring run rather than a silent zero for every design.

Both settings were missing when this benchmark was built: the block's
docstring claimed they were configured in `.env.ipd`, but neither the field
nor the entry existed, so `dssr_snap_metrics` could only run with hand-passed
absolute binary paths. They are now declared in `src/pipelines/settings.py`
and point at the shared install `/projects/ml/pipelines/software/x3dna-v2.3`
in `.env.ipd`.
