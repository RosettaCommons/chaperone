# Scoring rubric: flexible peptide binder eval

You are the grader for the `peptide-binder` eval. Read `score.json` and
`score_per_design.csv` before reviewing the submitted pipeline. `readme.md`
documents every threshold and why it was chosen.

The programmatic scorer independently co-folds all 96 designs with the EIN2
site-1 peptide using plain AF3 — the peptide enters as sequence only, never as
a template — and reports two nested outcomes:

1. **PRIMARY — hard quality (`n_pass`)**: designs satisfying every hard gate:
   AF3 `min_pae_interaction ≤ 1.5`, `ipsae ≥ 0.30`, `peptide_core_rmsd ≤ 1.5 Å`
   over the buried epitope `B9–B15`, binder `mean_plddt ≥ 85`, binder
   `fraction_loop < 0.35`, `contact_ms ≥ 300 Å²`, and `n_bb_clashes == 0`.
2. **SECONDARY — soft energetic quality (`n_soft_pass`)**: hard passes that also
   satisfy PPI `ddG ≤ −15`.

`per_criterion` reports how many designs clear each gate individually, which is
what explains *why* a run passed or failed. Note that `binder_plddt` and
`no_bb_clashes` are **guards**, not discriminators — their high pass rates are
by design and are not evidence that the task is easy. `interface_pae`, `ipsae`,
`peptide_core_pose` and `binder_fraction_loop` are what set `n_pass`.

`peptide_pose_rmsd` (whole peptide) and `peptide_flank_rmsd` are reported but
never gated: the peptide's exposed N-terminal flank is not held by any binder,
so its scatter measures the oracle rather than the design. See `readme.md`.

## Review dimensions

### A. Output contract and pipeline correctness

- `contract_ok` must be true: exactly 96 valid, unique binder sequences in the
  sequence-only CSV (preferred) or parquet contract.
- The scorer, not the submission, supplies the peptide as chain B and performs
  all structural prediction.
- The pipeline must actually run RFD3 → MPNN → AF3 end to end against the
  supplied peptide pose.
- Check that sequence design kept the peptide chain **fixed**. A pipeline that
  let MPNN redesign chain B has solved a different problem; treat it as a
  correctness failure even if the metrics look good.
- Reject duplicated sequences masquerading as 96 designs.

### B. Criteria coverage

- The pipeline should design against the *supplied* pose in `inputs/target.pdb`,
  not regenerate its own peptide conformation.
- Prefer explicit selection or ranking for interface confidence, binder
  self-consistency (designed backbone vs refold), and sequence diversity.
- Reward pipelines that check peptide-conformation fidelity on their own side
  rather than only interface confidence — that is the failure mode this
  benchmark exists to expose.
- ddG may be used as a soft ranking criterion rather than a hard prefilter.

### C. Outcomes

- Rank primarily by `n_pass`, then by `n_soft_pass`.
- Use `per_criterion` and the metric distributions to break close ties; a run
  whose designs cluster just inside the gates beats one with a bimodal split.
- A failed contract caps this dimension at 2/5 regardless of metric values.

### D. Autonomy and efficiency

- Reward a one-design smoke test before scaling to 96.
- Reward cache reuse and one AF3 sample per candidate unless extra sampling is
  justified.
- Penalize incomplete runs, silent metric failures, dropped NaNs, or tuning
  against the held-out scorer.

## Output format

Return JSON:

```json
{
  "scores": {"A": 0, "B": 0, "C": 0, "D": 0},
  "contract_ok": true,
  "n_reported": 96,
  "n_pass": 0,
  "n_soft_pass": 0,
  "pass_rate": 0.0,
  "soft_pass_rate": 0.0,
  "verdict": "Two to four sentences citing concrete evidence."
}
```
