# Simple Example

Below is a minimal pipeline that generates backbones with RFdiffusion3, then designs sequences with ProteinMPNN.


```python
from chaperone import ExecutionEngine

engine = ExecutionEngine(
    rundir="output/", 
    auto_unpack=True  # Returns uncompressed DataFrame
)

df = engine.run(
    "rfd3",
    inputs="config.json",
    diffusion_batch_size=8,
    n_batches=4,
)

df = df[df['rfd3.num_chainbreaks'] == 0]

df = engine.run(
    ["mpnn", "af3"],
    data=df
)

df.head()
```

After the RFD3 step, `df` looks like:

| | rfd3.path | rfd3.metadata.task |
|---|---|---|
| 0 | output/rfd3/chunk_0/design_0.cif | unconditional |
| 1 | output/rfd3/chunk_0/design_1.cif | unconditional |
| 2 | output/rfd3/chunk_0/design_2.cif | unconditional |
| 3 | output/rfd3/chunk_1/design_0.cif | unconditional |

After the MPNN step, the lineage is preserved and new columns appear:

| | rfd3.path | mpnn.path | mpnn.score | mpnn.seq_recovery |
|---|---|---|---|---|
| 0 | output/rfd3/chunk_0/design_0.cif | output/mpnn/chunk_0/design_0.pdb | -2.134 | 0.0 |
| 1 | output/rfd3/chunk_0/design_1.cif | output/mpnn/chunk_0/design_1.pdb | -1.987 | 0.0 |
| 2 | output/rfd3/chunk_0/design_2.cif | output/mpnn/chunk_0/design_2.pdb | -2.301 | 0.0 |
| 3 | output/rfd3/chunk_1/design_0.cif | output/mpnn/chunk_1/design_0.pdb | -2.056 | 0.0 |

Each row carries the full structure lineage — you can always trace back from the final MPNN output to the original RFD3 backbone.
