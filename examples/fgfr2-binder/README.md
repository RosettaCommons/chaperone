# FGFR2 Binder Example

Design 40–50 residue binders against a cropped FGFR2 D3 surface from PDB 1DJS,
then predict the complex with the full receptor using templated AF3. The bundled
hotspots target the D3 outer face; this example does not establish competition
with FGF or experimental binding.

## Pipeline

```text
rfd3 → ligand_mpnn → mpnn_score → select_top_seq
     → af3_with_target_template → rmsd → ipsae → contact_ms → peppr → round_1_filter
```

The defaults generate four backbones and eight sequences per backbone. MPNN
returns sequence rows; the confidence filter selects the top sequence per
backbone before AF3. There is one refolding/filtering round. The final gate
requires ipTM ≥ 0.7, minimum interface PAE ≤ 2 Å, backbone RMSD ≤ 3 Å, and
ipSAE ≥ 0.4. These are computational screening criteria.

## Inputs and conditioning

- `files/1djs_target.pdb` contains the full receptor on chain **B**. RFD3 keeps
  B287–296, B307–328, and B339–350, with a new binder on chain **A**. AF3 replaces
  the cropped receptor with the full receptor and uses it as a template.
- The default hotspot list is B289, B291, B311, B344, and B352. The first four
  occur in the retained RFD3 contig. B352 is outside that crop and does not
  supply an RFD3 hotspot atom. Keep retained hotspot residues when changing
  the crop; an empty hotspot selection cannot define the diffusion origin.
- `HOTSPOTS`, `HBOND_DONOR`, and `HBOND_ACCEPTOR` in `pipeline.py` customize
  conditioning. Generated overrides are written to
  `<rundir>/inputs/rfd3_config_override.json`; bundled inputs stay unchanged.
  Set `HOTSPOTS = None` to use the baseline JSON's hotspot selection.
- `files/1djs_complex.pdb` retains FGF for inspection. The optional
  `files/prep_target.py` fetches/prepares a complex and can select interface
  hotspots. If regenerating inputs, check that its hotspot selection and
  receptor contig describe the same surface, then update the pipeline constants.

## Run

The bundled inputs are ready to use. Configure the required containers, model
weights, and cluster settings, then start with one sample:

```bash
python examples/fgfr2-binder/pipeline.py --test --rundir=/path/to/fgfr2-smoke
```

Use `--local` inside an appropriate GPU allocation for local execution. Slurm
partition and GPU requests normally come from `CHAPERONE_GPU_PARTITION` and
`CHAPERONE_GRES_MAP`. The example requests the `large` GPU class.

The optional `run.sh` wrapper starts a Slurm CPU allocation for the driver:

```bash
RUNS_BASE=/path/to/fgfr2-runs ./examples/fgfr2-binder/run.sh test
RUNS_BASE=/path/to/fgfr2-runs ./examples/fgfr2-binder/run.sh pilot
```

`pilot` requests 200 backbones; `small` and `prod` currently use the four-backbone
script defaults. Increase sampling only after a successful bounded run. Final
metrics are written to `<rundir>/scores.csv`, sorted by ipSAE when available.
