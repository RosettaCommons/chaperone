# FGFR2 Competitive Binder Example

De novo design of 50-AA mini-protein inhibitors that compete with FGF for binding to the FGFR2 extracellular domain (PDB 1DJS). Hotspots on FGFR2 are auto-detected from the natural FGF interface and used to steer RFD3; designs are validated with AF3 and ranked by **ipSAE** — a per-interface confidence score derived from the AF3 PAE matrix.

## Pipeline

```
prep_target.py  (one-shot, fetches 1DJS, picks hotspots, writes target PDB + RFD3 config)
    │
    ▼
rfd3 → mpnn → af3 → rmsd → ipsae → round_1_filter   (loose: ipsae≥0.4, iptm≥0.7, min_pae≤2.0)
    → mpnn_2 → af3_2 → rmsd_2 → ipsae_2 → round_2_filter   (tight: ipsae≥0.6, iptm≥0.85, min_pae≤1.1)
```

## Key details

- **Hotspot auto-detection.** `files/prep_target.py` downloads 1DJS, picks the FGFR2 chain (largest protein chain) and the FGF chain in closest contact, then computes FGFR2 sidechain residues within `--radius` Å (default 4 Å) of any FGF atom. The result is baked into `files/rfd3_config.json` as `select_hotspots`. Re-run with `--radius 5.0` to widen, or with `--pdb <other>` to retarget a different complex.
- **Hotspot overrides — two routes.** Either:
  - **Re-run prep** with `--anchor B284 --cluster-radius 14 --max-hotspots 12` to re-cluster around a chosen residue (regenerates `files/rfd3_config.json`), OR
  - **Edit `pipeline.py`** — set the `HOTSPOTS` constant near the top to an explicit list (e.g. `["B281","B283",...]`). On the next run, the pipeline writes `files/rfd3_config_override.json` and points RFD3 at it; the auto-generated `rfd3_config.json` stays as a baseline. Set `HOTSPOTS = None` to revert.
- **Chain convention.** Receptor (FGFR2) is re-lettered to **B** so the diffused binder lands on **A** by default. The `IpSAE` block is configured `binder_chain="A", target_chains=["B"]` accordingly.
- **Two cleaned files.** `1djs_complex.pdb` (FGFR2 chain B + FGF chain C) is kept for visual inspection only; `1djs_target.pdb` (FGFR2 alone) is the file RFD3 reads.
- **Two-round filtering.** Round 1 casts a wide net on AF3 confidence + ipSAE; round 2 re-sequences the survivors with stricter thresholds. Same pattern as `examples/peptide-binder` — avoids spending compute on round 2 for clearly bad designs.
- **ipSAE as the headline selection metric.** ipSAE (Park & Cong, https://github.com/flyark42/ipsae) is implemented as a reusable block at `src/blocks/metrics/ipsae.py`. It reads the full per-token PAE matrix from the AF3 `*_confidences.json`, computes the canonical pTM-style interface score with `pae_cutoff=10` Å, and reports `ipsae` (worst-pair, the value to maximize), `ipsae_max` (best-pair), and per-chain detail.

## Workflow levels

### Code-level features
- `select_hotspots` extraction via `_get_closest_residues_by_sidechain_dist` (the same helper backing `AnnotateBlock`) — no manual residue picking.
- New reusable `IpSAE` metric block — works on any AF3 / RF3 prediction output.
- Two-round structure with the exact filter shape of `peptide-binder`, swapped to operate on a receptor target instead of a peptide.

### Pilot-level experiments
Parameters worth sweeping on ~100–1000 backbones:
- `--radius` in `prep_target.py` (3.5 / 4.0 / 5.0 Å) — narrower hotspot patches drive tighter contacts, wider patches give RFD3 more freedom.
- `--af3_ipsae.pae_cutoff` (8 / 10 / 12) — tighter cutoffs reward only the highest-confidence interfaces.
- `--rfd3.jitter_ori_sigma` (0.5 / 1.0 / 2.0) — controls the orientation noise around hotspot anchors.
- `--round_2_filter.af3_2_ipsae.ipsae` (0.5 / 0.6 / 0.7) — round-2 ipSAE threshold; tightening trades recall for quality.

### Campaign-level criteria (orderable)
Minimum criteria for an FGFR2 binder design to be considered orderable:
- Round 2 filter passed: `iptm ≥ 0.85`, `min_pae_interaction ≤ 1.1`, `bkbn_rmsd ≤ 2 Å`.
- `ipsae ≥ 0.7` on the FGFR2 interface (set the round-2 threshold higher when you have enough designs).
- Sequence diversity: do not order >30 sequences from the same RFD3 parent backbone.
- Hotspot occupancy: AF3 prediction places binder atoms within 5 Å of the auto-detected FGFR2 hotspots (verify per-design before ordering — not enforced by a filter).

Scale guidance: at the script's defaults (`n_batches=100, batch_size=50` → 5,000 RFD3 backbones), expect a small but informative survivor set. For a real campaign aiming at ~100 orderable designs, scale to `--rfd3.n_batches=500 --rfd3.batch_size=20` (10,000 backbones) and tighten round-2 ipSAE in proportion to the survivor count.

## Adapting to a different receptor

The script is target-agnostic; everything specific to FGFR2/FGF is in `files/`:
1. Re-run `prep_target.py --pdb <ID> --binder-length <N>` to swap the target.
2. The same hotspot auto-detection works for any 2-chain protein–protein complex (one chain = receptor, one chain = natural ligand whose pocket the binder will compete for).
3. Hotspot chain is always reported as **B** in the output config — no pipeline edits needed.

## Verification

```bash
# 1. Generate inputs (~1 minute, runs once)
./examples/fgfr2-binder/files/prep_target.py

# 2. Smoke test, pilot, or prod via the wrapper. The orchestrator runs in a
#    `srun` CPU allocation (login-node ulimits are too low for polars), and
#    submits GPU jobs to the cluster. Output is tee'd to <rundir>/run.log.
./examples/fgfr2-binder/run.sh           # test  → examples/fgfr2-binder/runs/test/
./examples/fgfr2-binder/run.sh pilot     # ~200 backbones → runs/pilot/
./examples/fgfr2-binder/run.sh prod      # full scale on gpu-bf → runs/prod/

# Override base directory (e.g. for /net/scratch on a different cluster):
RUNS_BASE=/net/scratch/$USER/fgfr2 ./examples/fgfr2-binder/run.sh

# Override GPU type or partition per-cluster:
GRES=gpu:a100:1 PARTITION=gpu ./examples/fgfr2-binder/run.sh
```
