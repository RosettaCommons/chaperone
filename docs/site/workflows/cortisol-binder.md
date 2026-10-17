# Cortisol binder

RFD3 generation around cortisol (HCY),
LigandMPNN sequence design with aromatic and hydrogen-bond contact fixing,
and two gated AF3 refinement cycles followed by a fresh final AF3 prediction.

```text
RFD3 → [LigandMPNN → AF3 → ligand remapping → H-bonds → cycle gate] × 2
     → final AF3 → ligand remapping → H-bonds → final gate
```

Each cycle requires minimum interaction PAE ≤ 2 Å and at least 2 ligand–protein
hydrogen bonds. Only passing rows enter the next round. The final AF3 prediction
must have minimum interaction PAE ≤ 1.25 Å, at least 4 hydrogen bonds, and mean
AF3 pLDDT ≥ 90. All final metrics refer to that same predicted structure.
`score.py` supplies both the engine filters and score reports; missing or
nonfinite metrics fail. HBPLUS supplies the hydrogen-bond counts.

The default is two 120–155-residue backbones, one packed sequence per backbone
per round, and two refinement cycles. `--test` reduces generation to one
backbone while retaining both cycles and every scientific gate. `--cycles=N`
changes the bounded number of rounds. A zero-pass round stops; it does not
verify execution of skipped downstream stages or establish a successful binder.

```bash
.venv/bin/python examples/cortisol-binder/pipeline.py --test --cycles=2 \
  --rundir=/absolute/path/cortisol-smoke
```

Use `--local --use_local_gpus` within an allocated GPU job. Configure Foundry,
LigandMPNN, AF3 weights/runtime, and `HBPLUS_COPY` for your site. Runtime-specific
`command_template` overrides can be supplied with `--config`; the portable
recipe does not hard-code cluster paths or partitions.

Inputs are ideal ligand-only coordinates from Biotite 1.6.0's bundled CCD and
the canonical isomeric SMILES from the [RCSB HCY record](https://www.rcsb.org/ligand/HCY).
`files/HCY.cif` retains the component chemistry; `files/cortisol.cif` places the
ligand in chain B, residue 1. Formula: C21H30O5; InChIKey:
`JYGXADMDTFJGBT-VWUMJDOOSA-N`. RFD3 fixes all 26 heavy atoms, conditions carbon
burial, uses O2/O3/O5 as donors and O1–O5 as acceptors. AF3 ligand names are
remapped to HCY before H-bond scoring and the next sequence-design round.

Outputs beneath `--rundir` include `pipeline.json`, cached block outputs,
`scores/cycle_*/{report.json,per_design.csv}`, final scores, and passing
`results.csv`. `driver.done` records successful execution, including an empty
scientific result; `driver.failed` records an execution exception.

CPU tests check cortisol chemistry, inclusive threshold boundaries, missing
values, current-round metric selection, surviving-structure handoff, and early
termination. GPU test results are recorded with the submitted run.

## GPU smoke result — 2026-10-17

The two-backbone test completed on one H200 (final Slurm attempt `682357`).
RFD3, aromatic-fixed LigandMPNN packing, AF3, ligand remapping, HBPLUS, and the
first-cycle filter all executed. Both AF3 structures retained all 26 cortisol
heavy atoms and finite coordinates. MPNN fixed all detected aromatic contacts
(three residues in the first design and one in the second).

| RFD3 model | Minimum interaction PAE | Ligand–protein H-bonds | Cycle gate |
|---|---:|---:|---|
| 0 | 5.79 | 3 | Fail PAE |
| 1 | 2.44 | 3 | Fail PAE |

Zero candidates passed the PAE ≤2 gate, so cycle two and final AF3 were correctly
skipped. Their control flow and exact final thresholds pass CPU tests; this
run does not claim GPU coverage of those skipped stages or a passing binder.

The same campaign resumed after repairing two engine defects: local execution
inherited submitit worker variables and replaced its driver; cycle engines
failed to inherit container bind settings. Regression tests cover both. The
three scheduler attempts used approximately 3 min 54 s of one-GPU allocation
in total and reused completed RFD3 outputs.

Run artifacts: `rundir/smoke-20261017/`, including per-candidate metrics in
`scores/cycle_0/per_design.csv`, the engine manifest, frozen submitted source,
and per-attempt Slurm logs. Empty `results.csv` retains readable column headers.
