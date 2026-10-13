# `antibody-cdr` scoring

Held-out scorer for CDR design on a supplied antibody framework. Run it as

```bash
./score.py --designs=/absolute/path/to/submission.csv --nouse_local_gpus
```

The submitted table is `design_id,sequence_H,sequence_L`. Everything else —
the antigen, the framework reference geometry, the oracle — is supplied here.

## What is contract and what is science

This benchmark is the first in the suite where part of the molecule is
*supplied*. That changes where the line sits.

A design that mutates a framework position is not a bad antibody; it is a
different antibody on a different scaffold, and scoring it would silently
compare two unlike things. So framework preservation is enforced by
`pipelines.evals.FixedRegionSpec` inside `load_submission_or_fail`, which
writes a `contract_ok: false` report and returns **before any GPU is
allocated**. The same is true of CDR-H3 length: 5–20 residues is a property of
being an antibody at all, not a property of being a good one.

`FixedRegionSpec` is deliberately general (it lives in `src/pipelines/evals.py`,
not here) because a nanobody or TCR benchmark needs exactly the same check.
It works by *decomposition* rather than position-wise comparison, so a design
that changed CDR-H3's length still validates:

```
sequence_H  ==  FW1 + CDR-H1 + FW2 + CDR-H2 + FW3 + CDR-H3 + FW4
```

with every `FW` piece byte-identical to the reference. If no such split exists
the framework was mutated (or a CDR length is out of bounds). If more than one
split exists the decomposition is refused rather than guessed, because every
region-level measurement downstream — CDR-H3 length, the developability
checks, the framework RMSD atom pairing — would otherwise be arbitrary.

## Inputs the scorer supplies

| File | Role |
|---|---|
| `../inputs/target.pdb` | hen egg-white lysozyme, chain **T**, co-folded with every design |
| `../inputs/framework.pdb` | reference D1.3 Fv, chains **H**/**L**; the geometry the framework-RMSD gate measures against |
| `../inputs/config.json` | reference sequences and AbM CDR spans, read directly by `score.py` |

Chains are `H`, `L`, `T`. The antigen is `T` and not `B` specifically so that
`_sequences_to_cif`'s collision check between context chains and submitted
chains can never fire, and so a reader of any per-design row knows immediately
which chain is which.

## Oracle

`af3_with_target_template` on the three-chain assembly (H + L + supplied T),
with `target_chain="T"` and `full_receptor_pdb=../inputs/target.pdb`.

**The template on the antigen is load-bearing, not cosmetic.** The pipelines
AF3 block runs with `--run_data_pipeline=False` and emits every chain with
`unpairedMsa: ""`, `pairedMsa: ""`, `templates: []` — i.e. single-sequence
mode. Antibody frameworks are stereotyped enough to survive that (H and L
predict at pLDDT 91-93), but hen egg-white lysozyme does not: with no MSA it
came out at **pLDDT ≈ 30 and chain pTM 0.24**, an unfolded blob. Nothing can
dock to an unfolded antigen, so every interface metric collapsed to a
structural zero. Injecting the supplied coordinates as an AF3 template restores
the antigen to pLDDT ≈ 90.

This is leak-free: the template is the antigen the scorer already hands out in
`inputs/target.pdb`. No paratope, interface or complex geometry is templated,
so the design still has to find the epitope on its own.

## Gates

| Gate | Metric | Threshold | Why |
|---|---|---|---|
| **contract** | mutations outside the six AbM CDR spans | == 0 | the framework is supplied |
| **contract** | CDR-H3 length | 5–20 | outside this is not a developable antibody |
| `interface_pae` | `ipsae.min_pae_interface` | ≤ 8.0 | paratope↔antigen minimum PAE |
| `ipsae` | `ipsae.ipsae` (`pae_cutoff=15`) | ≥ 0.05 | chain-pair-normalized interface confidence; recalibrated — the native complex scores 0.114, see below |
| `plddt` | `fold.plddt` | ≥ 80 | the complex must be a real, confident fold |
| `framework_preserved` | framework backbone RMSD vs `framework.pdb` | ≤ 1.5 Å | sequence identity is not conformational identity |
| `contact_ms` | `contact_ms` (H,L vs T) | ≥ 300 Å² | an antibody paratope buries a real surface |
| `no_clashes` | `clash.n_bb_clashes` | == 0 | |
| **soft** `developability` | CDR sequons == 0 **and** unpaired Cys ≤ 1 **and** net charge ∈ [−5, +5] **and** CDR-H3 hydrophobic run ≤ 4 | all must hold | the liabilities that kill an otherwise good binder downstream |

### Why the interface gate is loose

`≤ 8.0` Å is far looser than the peptide case (`≤ 2.0`) or PD-L1 (`≤ 4.0`).
That is deliberate and it is not a concession. Antibody–antigen complexes are
the class structure predictors are worst at: the paratope is six loops with no
evolutionary co-variation signal against the antigen, and even native
antibody–antigen complexes routinely predict with interface PAE well above
where a designed mini-binder helical bundle lands. A PPI-grade gate applied
here reports zero for every submission, and a benchmark that can only report
zero measures nothing — the same failure mode already recorded for
`biotin-binder` in `evals/PLAN.md`. The gate is set where it can discriminate
between designs that engage the antigen and designs that do not.

### Deviations from the starting spec in `evals/PLAN.md`

0. **`af3` → `af3_with_target_template`, and `ipsae` runs at
   `pae_cutoff=15.0` rather than the block default 10.0.** Both follow from the
   calibration run below; see "What the native control shows".

1. **`af3.min_pae_interaction` → `ipsae.min_pae_interface`.** The AF3 block's
   `min_pae_interaction` is the minimum over *all* off-diagonal chain pairs.
   On a three-chain H/L/T assembly the smallest pair is always VH–VL, the
   intra-Fv pairing, which predicts at ~0.5–1 Å for any correctly-framed
   antibody. It is therefore constant across submissions and says nothing
   about antigen engagement — a design whose CDRs never touch the antigen
   scores identically to one that buries 900 Å². `ipsae.min_pae_interface`
   with `binder_chain="T"` is the same statistic restricted to the
   paratope↔antigen submatrices, which is what the spec meant. The raw
   `fold.min_pae_interaction` is still reported per design as a diagnostic so
   the difference is visible.
2. **`af3.plddt` is kept as the fold gate, but `fv.mean_plddt` is reported
   alongside it.** Complex pLDDT includes the native lysozyme, which AF3
   predicts very confidently, so it flatters the design. The Fv-only mean
   pLDDT from `secondary_structure(chains=["H","L"])` is the honest number and
   is in every per-design row.

### How the H/L interface confidence is composed

The `ipsae` block takes `binder_chain` as a **single** chain, so it cannot be
handed "H and L together". Rather than instantiate two blocks, the scorer
inverts the roles: `binder_chain="T"`, `target_chains=["H","L"]`. One block
then evaluates both paratope pairs and emits

- `ipsae.ipsae_per_chain.T_H`, `ipsae.ipsae_per_chain.T_L` — per-chain detail,
- `ipsae.ipsae` = **min** over the two — the *worst* paratope chain,
- `ipsae.ipsae_max` = **max** over the two — the *dominant* paratope chain,
- `ipsae.min_pae_interface` = min PAE over both submatrices, both directions.

Each per-chain value is already direction-symmetric (`max(a→b, b→a)`), so
inverting binder and target does not change any number.

The gate is `ipsae.ipsae` — the worst of the two chains — which requires both
VH and VL to make a confident interface. Both statistics are reported so a
VH-dominant design (common, and not wrong) is visible in the per-design table
rather than hidden by the gate.

## What the native control shows

The single most useful calibration artifact for this benchmark is the *native*
D1.3 antibody: its CDRs are the crystallographic answer, so it is a true
positive by construction. Scoring the unmodified reference `sequence_H` /
`sequence_L` through this exact scorer, alongside three MPNN redesigns:

| | interface min PAE | ipSAE @10 | ipSAE @15 | ipSAE @20 | Fv pLDDT | contact_ms |
|---|---|---|---|---|---|---|
| **native D1.3** | **7.0** | 0.012 | **0.114** | 0.118 | 91.6 | 455 |
| redesign 0 | 16.7 | 0.000 | 0.000 | 0.002 | 88.7 | 390 |
| redesign 1 | 19.5 | 0.000 | 0.000 | 0.003 | 88.1 | 336 |
| redesign 2 | 22.2 | 0.000 | 0.000 | 0.000 | 87.7 | 412 |

Three things follow.

1. **`min_pae_interface ≤ 8.0` is well calibrated.** PLAN chose 8.0 before any
   run; the native lands at 7.0 and every redesign at 16.7-22.2. The gate sits
   in a wide empty band. It is kept unchanged.
2. **`ipsae ≥ 0.30` is structurally unpassable.** The native tops out at 0.118
   at *any* cutoff. ipSAE masks token pairs at or above `pae_cutoff`; at the
   default 10 A almost no antibody-antigen pair qualifies, and the score
   saturates at zero even for the real complex. Raising the cutoff to 15 A
   un-saturates it (native 0.114, redesigns exactly 0.000), so the gate is set
   to 0.05 — a factor of two below the only known true positive, inside a
   region the redesign distribution does not occupy at all. **This is a
   one-positive calibration and should be revisited when more real
   antibody-antigen pairs are run through it.**
3. **Single-sequence AF3 docks antibodies poorly even when it is right.** An
   interface PAE of 7 A for a 1.8 A crystal complex is not a confident dock; it
   is "adjacent, roughly right". That is the regime PLAN anticipated when it
   set this benchmark's interface gate far looser than the peptide (2.0) and
   PD-L1 (4.0) cases, and it is the honest ceiling of this oracle. A submission
   cannot be expected to beat the native.

## What the reference run actually scores: 0 / 96

The committed reference passes **0 / 96**, and that number survived a full
distribution check. It breaks down as:

| criterion | pass | note |
|---|---|---|
| `plddt` | **96 / 96** | was 0/96 before the antigen template fix |
| `framework_preserved` | **96 / 96** | the contract check working; RMSD 0.54-0.89 A |
| `no_clashes` | **96 / 96** | |
| `contact_ms` | 82 / 96 | median 365 A^2 |
| `developability` (soft) | 12 / 96 | |
| `interface_pae` | **0 / 96** | median 16.7 A vs an 8.0 gate; native 7.0 |
| `ipsae` | **0 / 96** | max 0.008 vs a 0.05 gate; native 0.114 |

The two remaining zeros are **not** plumbing, and each was checked against the
specific failure modes seen elsewhere in this benchmark suite:

- *Is the gate reading a chain-agnostic metric?* No. `interface_pae` reads
  `ipsae.min_pae_interface`, restricted to T<->{H,L}. The raw
  `af3.min_pae_interaction` is reported alongside it and is a different number
  entirely: across the 96 it ranges 0.99-2.51 A (it is reporting the native
  VH-VL pairing) against 9.1-26.0 A for the chain-restricted value.
  Correlation between the two is **r = 0.18** with a maximum absolute
  difference of **22.4 A**. Gating on the raw key would have passed 96/96 while
  measuring nothing about the antigen.
- *Is ipSAE deflated by how H and L are composed?* No. At `pae_cutoff=15` the
  native scores 0.1136 by per-chain minimum, 0.1165 by per-chain maximum and
  0.1173 treating H+L as one combined binder entity — under 4% spread. The
  cutoff, not the composition, was the whole effect.
- *Is the distribution shaped like a mis-set threshold?* No. 95 of 96 designs
  score **exactly** 0.0000 ipSAE. That is a structural zero in the designs,
  while the native scores 0.114 through the identical code path.

The conclusion is a real one about the reference method, not about the gates:
**MPNN inverse folding does not design antigen-binding CDRs.** It optimises
sequence given a backbone, and nothing in its objective knows the antigen is
there. Redesigning only the three contact-dominant loops (CDR-H3, CDR-H2,
CDR-L3) and leaving the canonical H1/L1/L2 native improves interface PAE from a
median of 16.7 to 12.5-15.7 A, but still passes 0 — so this is a property of
the design method, not of how much of the paratope is perturbed. The
`--design_cdrs` flag on the reference pipeline exposes that experiment.

The benchmark is therefore working as a benchmark: the native antibody clears
every gate, the inverse-folding baseline clears none, and roughly 10 A of
interface PAE separates them. What it needs is a stronger reference - an
antibody-aware generator such as the RFdiffusion3 antibody stack in
`examples/antibody-design`, which needs an external checkpoint this repository
does not ship.

## Developability, in detail

Every check is scoped to what the submitter controls. Framework sequons,
framework cysteines and framework charge are the benchmark's own and are not
counted against a design.

| Column | Definition |
|---|---|
| `n_cdr_sequons` | occurrences of `N-X-[S/T]`, `X != P`, where the Asn sits inside a CDR span. N-linked glycosylation in a CDR is a manufacturability and potency liability. |
| `n_cdr_cys` | cysteines inside CDR spans (raw count, reported not gated) |
| `n_unpaired_cys` | per chain, `n_cdr_cys % 2`, summed. The four framework cysteines are the canonical paired intradomain disulfides; a pair of CDR cysteines in the same chain can form a loop disulfide, an odd count leaves a free thiol. |
| `net_charge` | `(K+R) − (D+E)` over `sequence_H + sequence_L`. The native D1.3 Fv is **+3**, comfortably inside the ±5 window, so the gate measures the design and not the supplied framework. |
| `cdr_h3_max_hydrophobic_run` | longest run of `ACFILMVWY` in CDR-H3. Hydrophobic patches there drive aggregation and non-specific binding. |
| `cdr_h3_length` | reported; the 5–20 bound is enforced as contract, not as a gate |

Developability is a **soft** criterion: it gates `soft_pass`, never `pass`. A
design with a CDR sequon is still a real binder — it is just one that needs
reverting before it can be made.

## Failing loudly

- `require_finite_metrics` covers every gated column plus the diagnostics. A
  NaN is an upstream bug, not a failing design.
- `framework_backbone_rmsd` raises per row (naming the row) if fewer than 200
  framework backbone atoms can be paired, which is what a missing chain or a
  renumbered prediction would look like.
- The reference `example/example_pipeline.py` calls `FixedRegionSpec.check`
  on every MPNN output before writing `submission.csv`, so a broken framework
  surfaces at design time rather than at scoring time.

## Calibration status

**Partially calibrated — In progress.** The thresholds above are the starting
spec from `evals/PLAN.md` plus the deviations recorded here. The interface
gates are anchored to a single true positive (the native D1.3 antibody); the
fold, framework, packing and sterics gates are not yet anchored to anything. The committed
`../example/scored/` is a single reference campaign, not a calibration set. See
that report and `../example/example_pipeline.py`'s docstring for what the
reference actually is: an inverse-folding baseline on the native D1.3 backbone
with no oracle-based selection, chosen precisely so the numbers are not
pre-filtered by the metrics they are meant to calibrate.
