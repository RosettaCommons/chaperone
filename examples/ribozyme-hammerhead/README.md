# Hammerhead ribozyme design

RFD3NA rescaffolds the six catalytic nucleotides of the 2OEU hammerhead motif
into a 60-nt RNA; AF3 refolding is evaluated against the crystal reference.

See the [worked guide](../../docs/site/workflows/ribozyme-hammerhead.md) for
inputs, model setup, geometry gates, output fields, and validation limits.

```bash
python examples/ribozyme-hammerhead/pipeline.py --test --rundir=/absolute/path/hammerhead-smoke
```

Requires Foundry with the RFD3NA checkpoint and AF3 with separately obtained
parameters. Append `--local` inside an allocated GPU session. A one-sample run
may stop at a scientific filter before AF3; that does not verify the skipped
stage. Successful computation does not establish catalytic activity.

## GPU verification

A one-design H200 check completed RFD3NA (public 1190 checkpoint, 200 steps,
seed 42), AF3 (one sample, seed 42), crystal motif RMSD, and catalytic geometry.
The output contains 60 RNA residues and one Mn with finite coordinates and
metrics. The candidate fails the original scientific gates (motif RMSD 5.166 Å,
attack angle 73.12°); diagnostic refolding deliberately continued after its
initial clash failure. This proves execution, not an accepted ribozyme.
Native AtomWorks 3 Foundry and an existing AF3 image were used; a fresh public
container build remains a separate check. See the guide for complete details.
