# Scoring rubric: enzyme-protease eval

Held-out scorer for the de novo **zinc metalloprotease** eval. It takes a CSV
of **96 design sequences plus a `motif_index_map`**, co-folds every design with
the scorer-supplied substrate peptide and Zn site, and measures motif fidelity,
fold confidence, and substrate engagement on the refolded structure.

Self-reported metrics are never trusted: every submitted sequence is re-folded
here, from sequence alone.

> **These thresholds are uncalibrated.** They were written before any reference
> campaign existed and have not been moved since. `example/scored/score.json`
> is the first real measurement against them; see
> [Reference run](#reference-run-what-the-committed-example-actually-scores)
> below and treat the pass count as a *diagnostic*, not a settled contract.

## Usage

```bash
# 96-design submission, SLURM GPU
./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus

# one-design smoke test (contract size only; thresholds unchanged)
./score.py --designs=/absolute/path/to/one_row.csv --required_n=1 \
    --out=/tmp/score.json --per_design_out=/tmp/score_per_design.csv
```

| Flag | Required | Description |
|---|---|---|
| `--designs` | yes | CSV (preferred) or parquet submission table. |
| `--required_n` | no | Contract size, default 96. **Development only** — reduce to 1 for a smoke test. It changes nothing scientific. |
| `--out` | no | Aggregate report JSON (default `score.json`). |
| `--per_design_out` | no | Per-design metric/verdict CSV (default `score_per_design.csv`). |
| `--run_meta` | no | JSON cost telemetry `{gpu_hours, cpu_hours, wallclock_min}`; enables pass-per-cost. |

Harbor always passes `trial/controller/run_meta.json`, derived from its guarded
job registry and scheduler accounting; it never trusts a file in the model's
submission. The flag remains optional for standalone non-Harbor scoring.

Held-out compute runs under `score/score_runs/<submission-hash>/` — content
addressed, so re-scoring the same file reuses its cache and two different
submissions never share a directory. That tree is gitignored.

## The submission contract

Checked by `pipelines.evals.load_sequence_submission` **before any GPU is
touched**. A contract failure writes a report with `contract_ok: false` and
spends nothing.

| Column | Required | Rule |
|---|---|---|
| `sequence` | yes | The enzyme chain. Uppercase canonical amino acids only. Non-empty, and unique across the 96 rows. |
| `motif_index_map` | yes | `H293:A45;E294:A46;H297:A49;E316:A68;Y377:A129;E378:A130` — every one of the six reference positions, each mapped to a distinct 1-based residue index of `sequence`. |
| `design_id` | no | Unique, non-empty, no surrounding whitespace. Generated from row order if absent. |

Any other column (structures, paths, coordinates, the substrate, the metal)
fails the contract.

`motif_index_map` exists because a sequence-only submission carries no
coordinates: without it the scorer cannot tell which residues of the design are
meant to be catalytic, and motif RMSD is undefined. The map is validated three
ways — every reference position present, all six landing on distinct residues
inside the sequence, and **the amino acid at each mapped index matching the
reference residue type**:

| Reference position | Required residue | Role |
|---|---|---|
| `H293` | His | Zn coordination (HEXXH) |
| `E294` | Glu | general base (HEXXH) |
| `H297` | His | Zn coordination (HEXXH) |
| `E316` | Glu | second shell / substrate positioning |
| `Y377` | Tyr | oxyanion stabilisation |
| `E378` | Glu | third Zn ligand |

A mis-scaffolded motif therefore fails the **contract**, not the science.

## What the scorer supplies

From `inputs/theozyme.pdb`, co-folded with every design and never accepted from
the submitter:

- chain **B** — the 10-residue fibrin Aa substrate peptide (`GPRVVERHQS`);
- chain **C** — the `ZnO` zinc/hydroxide site.

The design occupies chain **A**. `load_context_chains` appends B and C verbatim
to a coordinate-free chain A built from the submitted sequence; AF3 then folds
the whole assembly. The `ZnO` residue is auto-resolved to SMILES `O.[Zn]`.

The same file is the **reference geometry** for motif RMSD.

## Scientific criteria

All three are hard gates; a design passes only if it clears every one.

| Gate | Metric | Threshold | Rationale |
|---|---|---|---|
| motif fidelity | `motif_rmsd` | ≤ **1.5 Å** | The headline metric. All-atom RMSD of the six mapped catalytic residues against the reference geometry, atoms paired **by name** within each residue and superimposed on those pairs alone. A metalloprotease is defined by the geometry of its Zn site: if the refolded design does not reproduce that arrangement, nothing else about it matters. 1.5 Å is loose for a local six-residue all-atom fit — it is a first-shell-geometry gate, not a "catalytically perfect" gate — and it is the value most in need of calibration. |
| fold confidence | `holo.plddt` | ≥ **80** | The standard AF3 confidence floor used across this eval suite. A design AF3 is unsure about is not a fold. |
| substrate engagement | `holo_ipsae.min_pae_interface` (A vs B) | ≤ **10 Å** | Deliberately far looser than a binder benchmark (PD-L1 uses <4 Å). The "target" here is a 10-residue flexible peptide threaded through an active site, not a folded domain face; a PPI-grade PAE gate on a decapeptide would report zero by construction. **The metric — not the threshold — was corrected**: see below. |

Every gate must be finite for every row. `require_finite_metrics` raises rather
than dropping a NaN, because a missing metric is an upstream bug, not a failing
design.

### Reported but not gated

- `holo_contact_ms` — contact molecular surface between design chain A and
  substrate chain B. Computed, required to be finite, and written to
  `score_per_design.csv`, but no threshold is applied. It is there to inform
  calibration: a packing gate is the most likely fourth criterion.
- `holo.min_pae_interaction` — retained purely for provenance and for the
  before/after comparison below. **Do not gate on it** (see the bug note).

### Fixed bug: motif maps were joined to structures by row position

`motif_rmsd` originally paired `frame["holo.path"]` with the submission's
`motif_index_map` column via a positional `zip()`. That silently assumes the
scored frame comes back in submission order. It does not — AF3's `compile()`
collects outputs with `glob.glob` and the handler merge iterates the new rows,
so the returned order is filesystem-dependent. Nearly every design was scored
against **another design's motif map**, and because every resulting value is
still finite and plausible, `require_finite_metrics` could never catch it.

The join is now keyed on `design_id`, with an explicit set-equality assertion
between the submission's IDs and the scored frame's IDs so any future mismatch
fails loudly instead of silently. No threshold changed.

### Fixed bug: the substrate gate was measuring the wrong interface

The `substrate_interface` gate originally read `holo.min_pae_interaction`.
That AF3 metric is the minimum over **all** off-diagonal chain pairs of
`chain_pair_pae_min` (`alphafold3.py:_compute_min_pae_interaction`) — it has no
notion of which interface is of interest. On a two-chain binder fold that is
harmless. This benchmark folds **three** chains (A design, B substrate,
C zinc), and the zinc sits in the substrate's own active site, so the B:C
substrate↔metal pair is trivially well predicted and routinely wins the
minimum. The gate was therefore reporting substrate-to-metal confidence under
the name "substrate engagement".

The gate now reads `holo_ipsae.min_pae_interface` from the `ipsae` block with
`binder_chain="A", target_chains=["B"]`, which is restricted to the
design×substrate PAE block in both directions. **The numeric threshold is
unchanged at 10 Å** — only the quantity being measured was corrected.

Evidence that the old metric was inert: across the 93-design reference pool,
`holo.min_pae_interaction` spanned only 0.88–5.69 Å and passed **93/93** at the
10 Å bar, i.e. it excluded nothing.

### One remaining semantic caveat, recorded deliberately

**`holo.plddt` is the whole-complex mean pLDDT**, not the design chain's. AF3
emits the mean over every atom, so the 10-residue peptide and the Zn site are
included. The design is ~95 % of the atoms, so the difference is small, but the
docstring's "pLDDT of the design chain" is an approximation. This one is left
as-is because correcting it would move the effective bar, which is a
calibration decision rather than a bug fix.

## Scoring tiers

1. **PRIMARY — pass rate (`n_pass` / 96).** All three hard gates.
   `n_soft_pass` equals `n_pass`; this benchmark declares no soft criteria.
2. **SECONDARY — compute spent.** Pass rate per GPU-hour and per CPU-hour from
   `--run_meta`. At comparable pass rate a leaner run ranks higher; it never
   lifts a run above one with a higher `n_pass`. Without `--run_meta`,
   `efficiency` is `null`.
3. **TERTIARY — distribution quality.** How far the whole 96-design
   distribution sits inside the pass region, from `metric_summaries` and
   `per_criterion`, rather than how many rows squeaked over a line.

## Output

`score.py` writes `score.json` (aggregate) and `score_per_design.csv`
(`design_id`, normalized internal `sequence_A`, the four metrics, and every
`pass_*` column plus
`pass` / `soft_pass`). `score.json` echoes the thresholds it used and records
`submission_sha256`, so a committed score can always be tied back to the exact
table it scored.

## Reference run: what the committed example actually scores

`example/submission.csv` is the reference submission produced by
`example/example_pipeline.py` (RFD3 → filter → LigandMPNN → AF3 holo
self-validation, 96 designs selected by rank-sum over the three scored
quantities). `example/scored/` is that table run through this scorer,
unmodified.

**Result: 31 / 96 pass (32.3 %).** `contract_ok: true`; every metric finite for
every row; `submission_sha256` `aff60c6e…` matches the committed CSV.

| Criterion | Threshold | Passing |
|---|---|---|
| `motif_fidelity` | `motif_rmsd` ≤ 1.5 Å | **35 / 96** |
| `plddt` | `holo.plddt` ≥ 80 | **77 / 96** |
| `substrate_interface` | `holo_ipsae.min_pae_interface` ≤ 10 Å | **96 / 96** |
| **all three (AND)** | | **31 / 96** |

Metric distributions over the 96 scored designs:

| Metric | min | median | mean | max |
|---|---|---|---|---|
| `motif_rmsd` (Å) | 0.587 | 1.760 | 2.688 | 9.040 |
| `holo.plddt` | 64.02 | 85.23 | 84.56 | 95.14 |
| `holo_ipsae.min_pae_interface` (Å) | 0.80 | 2.50 | 2.67 | 8.50 |
| `holo.min_pae_interaction` (Å, ungated) | 0.81 | 1.02 | 1.05 | 1.47 |
| `holo_contact_ms` (Å², ungated) | 201.2 | 435.0 | 427.0 | 661.6 |

Cumulative pass counts, for choosing a bar from evidence:

| `motif_rmsd` ≤ | 1.00 | 1.25 | **1.50** | 1.75 | 2.00 | 2.50 | 3.00 |
|---|---|---|---|---|---|---|---|
| designs | 13 | 24 | **35** | 48 | 53 | 65 | 68 |

| `holo.plddt` ≥ | 70 | 75 | **80** | 85 | 90 |
|---|---|---|---|---|---|
| designs | 92 | 88 | **77** | 49 | 21 |

| `min_pae_interface` ≤ | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | **10.0** |
|---|---|---|---|---|---|---|---|---|
| designs | 8 | 15 | 28 | 49 | 66 | 86 | 93 | **96** |

**Effect of the interface-metric fix**, same 96 designs, same 10 Å bar:

| | min | median | max | span | passing ≤10 |
|---|---|---|---|---|---|
| old `holo.min_pae_interaction` (all chain pairs) | 0.81 | 1.02 | 1.47 | **0.66 Å** | 96/96 |
| new `min_pae_interface` (A↔B only) | 0.80 | 2.50 | 8.50 | **7.70 Å** | 96/96 |

Pearson *r* between the two is only **0.242** and the maximum per-design
difference is **7.54 Å**. The old quantity was compressed into a 0.66 Å band
around ~1.0 Å regardless of the design — it was tracking the Zn site, not the
substrate interface, and carried essentially no information about the design.
The pass count at 10 Å is unchanged only because *both* are inert at that bar;
at any discriminating threshold they diverge completely (e.g. at ≤1.5 Å: old
96/96, new 15/96).

### Threshold recommendations

Evidence-based, from the distributions above. **Not applied** — the committed
thresholds are unchanged pending a decision.

| Gate | Current | Recommended | Why |
|---|---|---|---|
| `motif_rmsd` | ≤ 1.5 Å | **keep ≤ 1.5 Å** | The only gate already behaving well. It passes 35/96 (36 %) on this selected pool and 24/93 (26 %) on an *unselected* one — a real discriminator that is neither vacuous nor impossible. The distribution is smooth through the bar (24 at 1.25, 35 at 1.5, 48 at 1.75), so the exact placement is not knife-edge. |
| `holo.plddt` | ≥ 80 | **raise to ≥ 85** | 80 passes 77/96 (80 %) here and is doing little work; 85 passes 49/96 and sits near the median (85.2), splitting the pool roughly in half. Note this metric is the whole-complex mean, so it runs a little below a design-chain-only pLDDT. |
| interface PAE | ≤ 10 Å | **tighten to ≤ 3.0 Å** | Even on the corrected metric, 10 Å passes 96/96 — the observed maximum is 8.5 Å, so the gate can never fire. 3.0 Å passes 66/96 and is where the distribution actually has mass (median 2.5). Keep it looser than a PPI benchmark: the target is a flexible decapeptide. |

Combined effect of the recommended set (`≤1.5` / `≥85` / `≤3.0`): **22 / 96
(23 %)** — still a live, non-degenerate contract, with every individual gate
doing measurable work. For reference, the current set gives 31/96 but with one
gate inert.

A packing gate on `holo_contact_ms` is the natural fourth criterion if one is
wanted: ≥250 Å² passes 88/96, ≥300 Å² passes 82/96.
