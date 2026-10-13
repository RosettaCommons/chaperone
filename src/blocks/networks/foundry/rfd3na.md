Failed to create stream fd: Operation not permitted
Failed to create stream fd: Operation not permitted
Failed to create stream fd: Operation not permitted
# RFdiffusion3NA (`rfd3na`) input reference

The `rfd3na` generation block accepts a JSON/YAML file or an inline Python
dictionary of named design specifications. It generates proteins, RNA, DNA,
and mixed-biopolymer structures as compressed CIF files, with one metadata
JSON file per design.

Upstream source: [RosettaCommons/foundry `models/rfd3na`](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3na).

Install the shared runtime with `pipelines install foundry`. The default IPD
checkpoint is `/projects/ml/pipelines/weights/foundry/checkpoints/rfd3na_1190.ckpt`;
outside that environment, pass `ckpt_path="rfd3na"` after installing the
checkpoint with Foundry's checkpoint installer.

## Block usage

```python
from blocks import Blocks

RFD3NA = Blocks["rfd3na"]
block = RFD3NA(
    inputs={"rna": {"contig": "40-50R", "length": "40-50"}},
    n_batches=2,
    batch_size=4,
)
```

Each named specification produces `n_batches * batch_size` designs. Pipelines
uses the chunk ID to vary the seed and output prefix, so chunked generation
does not repeat samples. `read_sequence_from_sequence_head=False` and input
prevalidation are enabled by default, following the upstream RFD3NA inference
recommendation.

## Configuration structure

The top level maps a filesystem-safe design name to a specification. Names may
contain letters, digits, `_`, and `-`, but not periods.

```json
{
  "job-name": {
    "input": null,
    "contig": "40-50R,/0,20-20D,/0,80-100",
    "length": "140-170",
    "ligand": null,
    "unindex": null,
    "select_fixed_atoms": null,
    "select_unfixed_sequence": null,
    "select_buried": null,
    "select_partially_buried": null,
    "select_exposed": null,
    "select_hbond_donor": null,
    "select_hbond_acceptor": null,
    "select_hotspots": null,
    "redesign_motif_sidechains": false,
    "ori_token": null,
    "ori_jitter": null,
    "infer_ori_strategy": null,
    "is_non_loopy": null,
    "partial_t": null,
    "symmetry": null,
    "extra": {}
  }
}
```

Input paths in file-backed configs are resolved relative to the config file.
The block validates them and stages referenced structures into the execution
sandbox before invoking the container.

## Core fields

| Field | Type | Default | Description |
|---|---|---:|---|
| `input` | string or null | `null` | PDB/CIF containing motifs, nucleic acids, or ligands. |
| `contig` | selection string | none | Ordered input and designed segments. |
| `length` | integer or `"min-max"` | inferred | Total token/residue count constraint. |
| `ligand` | string | none | Comma-separated component names or selections to retain. |
| `unindex` | selection | none | Motif components whose output sequence positions are not fixed. |
| `select_fixed_atoms` | bool/string/object | input atoms fixed | Coordinates to preserve from the input. |
| `select_unfixed_sequence` | bool/string/object | input sequence fixed | Input components whose residue identity may change. |
| `select_buried` | selection | none | Atoms conditioned as buried. |
| `select_partially_buried` | selection | none | Atoms conditioned as partly buried. |
| `select_exposed` | selection | none | Atoms conditioned as exposed. |
| `select_hbond_donor` | selection object | none | Atoms conditioned as hydrogen-bond donors. |
| `select_hbond_acceptor` | selection object | none | Atoms conditioned as hydrogen-bond acceptors. |
| `select_hotspots` | selection | none | Target atoms/residues for interface generation. |
| `redesign_motif_sidechains` | bool/string | `false` | Redesign sidechains while retaining motif backbone coordinates. |
| `ori_token` | `[x, y, z]` | inferred | Desired center for the generated region. |
| `ori_jitter` | float | none | Mean distance in angstroms for per-batch origin jitter. |
| `infer_ori_strategy` | `"com"`/`"hotspots"` | none | Infer the origin from the input or hotspot geometry. |
| `is_non_loopy` | bool | none | Bias toward or away from non-loop protein structure. |
| `partial_t` | float | none | Partial-diffusion noise in angstroms; start near 2 and increase cautiously. |
| `symmetry` | object | none | Upstream RFD3NA symmetry configuration. |
| `extra` | object | `{}` | User metadata retained in the output specification. |

## Polymer and selection syntax

Designed contig segments use suffixes to choose polymer type:

- `50-70` generates a 50–70 residue protein segment.
- `30-30R` generates a 30-nucleotide RNA segment.
- `20-20D` generates a 20-nucleotide DNA segment.
- `/0` starts a new chain.
- `A10-25` copies residues 10–25 from input chain A.

For example, `40-50R,/0,20-20D,/0,80-100` generates separate RNA,
DNA, and protein chains. Polymer suffixes apply to newly generated segments;
input selections such as `A10-25` retain their input polymer identity.

Selection fields accept a boolean, a contig-like residue selection, or an
object mapping selections to atom names. Common values are `ALL`, explicit
comma-separated atom names, and an empty string to select no atoms. The
protein shorthands `BKBN` and `TIP` do not apply to nucleic-acid residues;
name nucleic-acid atoms explicitly.

```json
{
  "select_fixed_atoms": {
    "A10-14": "ALL",
    "B3": "P,O5',C5',C4',C3',O3'",
    "B4-8": ""
  }
}
```

## Nucleic-acid secondary-structure conditioning

The current RFD3NA parser also accepts these NA-specific fields:

| Field | Type | Description |
|---|---|---|
| `ss_dbn` | string | Dot-bracket notation applied to the first matching NA tokens. |
| `ss_dbn_dict` | object | Dot-bracket strings keyed by output chain/range. |
| `paired_region_list` | list[string] | Paired region pairs such as `"A20-25,B10-15"`. |
| `paired_position_list` | list[string] | Explicit paired positions such as `"A3,B18"`. |
| `loop_region_list` | list[string] | Output regions conditioned as loops. |

Dot-bracket length must match the selected NA region. Pseudoknot bracket
families are supported by the upstream parser, but malformed or conflicting
pair assignments fail prevalidation.

## Complete examples

### Unconditional RNA

```json
{
  "rna_hairpin": {
    "contig": "48-48R",
    "length": "48-48",
    "ss_dbn": "((((((((........))))))))........................"
  }
}
```

### Mixed RNA, DNA, and protein

```json
{
  "mixed_complex": {
    "contig": "40-50R,/0,20-20D,/0,80-110",
    "length": "140-180"
  }
}
```

### Two-chain RNA with explicit pairing

```json
{
  "rna_duplex": {
    "contig": "30-30R,/0,30-30R",
    "length": "60-60",
    "paired_region_list": ["A6-20,B6-20"],
    "loop_region_list": ["A21-30", "B21-30"]
  }
}
```

### Ligand-conditioned RNA aptamer

```json
{
  "amp_aptamer": {
    "input": "inputs/AMP.pdb",
    "ligand": "AMP",
    "contig": "40-50R",
    "length": "40-50",
    "ori_jitter": 1.0,
    "select_buried": {"AMP": "ALL"},
    "select_hbond_acceptor": {"AMP": "N7,O4',O1P,O2P,O3P,N3,N1"},
    "select_hbond_donor": {"AMP": "N6,O3',O2'"}
  }
}
```

### Protein binder around fixed DNA

```json
{
  "dna_binder": {
    "input": "inputs/dna_target.cif",
    "contig": "C5-18,/0,D24-37,/0,70-100",
    "length": "98-128",
    "select_fixed_atoms": {
      "C5-18": "ALL",
      "D24-37": "ALL"
    },
    "select_hotspots": "C9-14,D28-33",
    "infer_ori_strategy": "hotspots",
    "is_non_loopy": true
  }
}
```

### Unindexed RNA/protein active-site scaffolding

```json
{
  "rnase_active_site": {
    "input": "inputs/rnase_active_site.cif",
    "contig": "50-80R,/0,100-120,/0,C1-4,C79-86",
    "length": "162-212",
    "ligand": "MG,PO4",
    "unindex": "B49-52,B321,/0,A56-58",
    "select_fixed_atoms": {
      "B49-52": "ALL",
      "B321": "ALL",
      "A56-58": "ALL",
      "C1-4": "ALL",
      "C79-86": "ALL"
    }
  }
}
```

### Multiple jobs in one file

```json
{
  "rna_40": {"contig": "40-40R", "length": "40-40"},
  "dna_24": {"contig": "24-24D", "length": "24-24"},
  "protein_rna": {
    "contig": "80-100,/0,30-40R",
    "length": "110-140"
  }
}
```

## Outputs

Each design is compiled into the `rfd3na` structure namespace. The paired JSON
provides `rfd3na.*` metrics plus metadata including the original specification,
checkpoint, seed, motif index mapping, and any NA secondary-structure metrics
emitted by upstream RFD3NA. Diffusion trajectory CIFs are excluded from the
compiled designs.

## Known limitations and gotchas

1. RFD3NA is GPU-oriented. Reduce `batch_size` or enable `low_memory_mode` for
   long multipolymer systems.
2. Hydrogen-bond conditioning or metrics that depend on HBPLUS require a
   separate HBPLUS installation; HBPLUS is not bundled in the Foundry
   container.
3. Use explicit nucleic-acid atom names; protein-only `BKBN` and `TIP`
   shorthands are not valid NA atom definitions.
4. A provided input structure must actually be referenced by `contig`,
   `unindex`, `ligand`, or partial diffusion, or prevalidation rejects it.
5. Keep `read_sequence_from_sequence_head=False` unless intentionally testing
   upstream experimental behavior.
6. Trajectories are large. Enable `dump_trajectories` only for debugging.

## GPU sizing

The block defaults to the `gpu` partition with `gpu:small:1` because typical
RNA design jobs are small. On the current cluster, the small class includes
A4000/A5000/B4000/4000 Ada-class devices. Large or known memory-intensive
designs should explicitly request `gpu:large:1`; the current large class
includes A6000/L40/L40S/A100/H200-class devices.

RFD3NA does not publish a supported sequence-length cutoff between these size
classes. Peak memory depends on total polymer tokens, `diffusion_batch_size`,
input atomization, and whether `low_memory_mode` is enabled; base pairs alone
are not a sufficient predictor for mixed-polymer jobs.

Callers can opt into a large GPU through Pipelines job arguments:

```python
engine.run(
    RFD3NA(inputs="inputs.json", batch_size=4),
    job_args={"partition": "gpu", "gres": "gpu:large:1"},
)
```

Reducing `batch_size` and enabling `low_memory_mode` remain useful ways to
reduce peak memory. There is no validated automatic sequence-length boundary;
future profiling may provide sizing guidance, but the block does not infer a
GPU class from sequence length.
