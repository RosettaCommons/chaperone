# DISCO Input Format

DISCO uses a JSON input file following the **AlphaFold Server format**. The file contains a list of jobs, each specifying the entities to co-design.

## JSON Structure

```json
[
  {
    "name": "job_name",
    "sequences": [
      { "<entity_type>": { ... } },
      { "<entity_type>": { ... } }
    ],
    "covalent_bonds": []
  }
]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | yes | Unique job name, used in output filenames |
| `sequences` | list[object] | yes | List of entities (protein, ligand, DNA, RNA, ion) |
| `covalent_bonds` | list[object] | no | Inter-entity covalent bonds |

Each entry in `sequences` is an object with exactly **one** entity type key.

---

## Entity Types

### proteinChain

```json
{"proteinChain": {"sequence": "MKTL------VPEG", "count": 1}}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sequence` | string | required | Amino acid sequence. `-` = designable (masked) position |
| `count` | int | 1 | Number of copies of this chain |
| `modifications` | list | optional | Post-translational modifications |

**Masking**: `-` (hyphen) marks designable positions. Standard amino acid letters (A-Y) are fixed. `X` = unknown.

```
"--------"              # fully masked 8-residue protein
"MKTL----VPEG"          # fix termini, design middle
"--------------------"  # fully masked 20-residue protein
```

**Modifications** (optional):
```json
{"modifications": [{"ptmPosition": 5, "ptmType": "CCD_SEP"}]}
```
- `ptmPosition`: 1-indexed residue position
- `ptmType`: must start with `CCD_` followed by the CCD code (e.g. `CCD_SEP` for phosphoserine)

**Limitation**: Only a **single** `proteinChain` entity is supported per job. Multiple protein chains will raise an error (the language model was trained on single chains only).

### ligand

Three ways to specify a ligand:

```json
// SMILES string (auto-generates 3D conformer via RDKit)
{"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 1}}

// File path (SDF, MOL, MOL2, PDB) -- prefixed with "FILE_"
{"ligand": {"ligand": "FILE_path/to/molecule.sdf", "count": 1}}

// CCD code -- prefixed with "CCD_"
{"ligand": {"ligand": "CCD_ATP", "count": 1}}
```

| Method | Prefix | Example | Notes |
|--------|--------|---------|-------|
| SMILES | none | `"CC(=O)Oc1ccccc1C(=O)O"` | Must be valid RDKit SMILES |
| File | `FILE_` | `"FILE_heme_b.sdf"` | Resolved relative to input JSON dir. Must contain 3D conformer |
| CCD code | `CCD_` | `"CCD_ATP"` | Looked up from Chemical Component Dictionary |
| Multi-CCD | `CCD_` | `"CCD_NAG_BMA_BGC"` | Underscore-separated codes |

- Supported file formats: `.sdf`, `.mol`, `.mol2`, `.pdb`
- Files **must** contain 3D coordinates (2D-only files are rejected)
- Internally assigned residue names `l01`, `l02`, ..., `l99` (max 99 ligands per job)
- `count` specifies copies of the ligand

### dnaSequence

```json
{"dnaSequence": {"sequence": "GATTACAGATC", "count": 1}}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sequence` | string | required | DNA bases: `A`, `T`, `G`, `C`, `N`/`X` (unknown) |
| `count` | int | 1 | Number of copies |
| `modifications` | list | optional | Base modifications |

- Masking (`-`) is **not supported** for DNA -- all positions must be specified
- For **double-stranded DNA**, add each strand as a separate `dnaSequence` entry

### rnaSequence

```json
{"rnaSequence": {"sequence": "GGCUAGCCAUUUGAC", "count": 1}}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sequence` | string | required | RNA bases: `A`, `U`, `G`, `C`, `N`/`X` (unknown) |
| `count` | int | 1 | Number of copies |
| `modifications` | list | optional | Base modifications |

- Masking (`-`) is **not supported** for RNA

### ion

```json
{"ion": {"ion": "ZN", "count": 1}}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ion` | string | required | CCD code for the ion (e.g. `ZN`, `MG`, `NA`, `FE`, `CA`) |
| `count` | int | 1 | Number of copies |

---

## Covalent Bonds

Define inter-entity covalent bonds (e.g. covalent ligand attachment):

```json
{
  "covalent_bonds": [
    {
      "left_entity": 1,
      "left_position": 42,
      "left_atom": "SG",
      "right_entity": 2,
      "right_position": 1,
      "right_atom": "C1"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `left_entity` / `right_entity` | int | 1-indexed entity position in `sequences` list |
| `left_position` / `right_position` | int | Residue index within entity |
| `left_atom` / `right_atom` | string or int | Atom name, or SMILES atom map number |
| `left_copy` / `right_copy` | int (optional) | Which copy if `count > 1` |

Leaving atoms are automatically removed based on CCD data.

---

## Complete Examples

### Unconditional (fully masked protein)

```json
[{"name": "uncon_100", "sequences": [
  {"proteinChain": {"sequence": "----------------------------------------------------------------------------------------------------", "count": 1}}
]}]
```

### Ligand-conditioned (SMILES)

```json
[{"name": "aspirin_binder", "sequences": [
  {"proteinChain": {"sequence": "----------------------------------------", "count": 1}},
  {"ligand": {"ligand": "CC(=O)Oc1ccccc1C(=O)O", "count": 1}}
]}]
```

### Ligand-conditioned (SDF file)

```json
[{"name": "heme_b_200", "sequences": [
  {"proteinChain": {"sequence": "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------", "count": 1}},
  {"ligand": {"ligand": "FILE_heme_b.sdf", "count": 1}}
]}]
```

### Ligand-conditioned (CCD code)

```json
[{"name": "atp_binder", "sequences": [
  {"proteinChain": {"sequence": "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------", "count": 1}},
  {"ligand": {"ligand": "CCD_ATP", "count": 1}}
]}]
```

### DNA-conditioned (double-stranded)

```json
[{"name": "nucleic_example", "sequences": [
  {"proteinChain": {"sequence": "----------------------------------------------------------------------", "count": 1}},
  {"dnaSequence": {"sequence": "TTTGCACCAA", "count": 1}},
  {"dnaSequence": {"sequence": "GTAATTGCGG", "count": 1}}
]}]
```

### RNA-conditioned

```json
[{"name": "rna_example", "sequences": [
  {"proteinChain": {"sequence": "------------------------------------------------------------", "count": 1}},
  {"rnaSequence": {"sequence": "UCUCUGGUUAGGAAACUAACUAGGGA", "count": 1}}
]}]
```

### Ion-conditioned (zinc finger)

```json
[{"name": "zinc_binder", "sequences": [
  {"proteinChain": {"sequence": "----------------------------------------------------------------------------------------------------", "count": 1}},
  {"ion": {"ion": "ZN", "count": 1}}
]}]
```

### Partial design (fixed termini + ligand)

```json
[{"name": "partial_heme", "sequences": [
  {"proteinChain": {"sequence": "MKTL------------------------------------------------------------VPEG", "count": 1}},
  {"ligand": {"ligand": "CCD_HEM", "count": 1}}
]}]
```

### Multiple jobs in one file

```json
[
  {"name": "short_100", "sequences": [
    {"proteinChain": {"sequence": "----------------------------------------------------------------------------------------------------", "count": 1}},
    {"ligand": {"ligand": "CCD_HEM", "count": 1}}
  ]},
  {"name": "long_200", "sequences": [
    {"proteinChain": {"sequence": "--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------", "count": 1}},
    {"ligand": {"ligand": "CCD_HEM", "count": 1}}
  ]}
]
```

---

## CLI Arguments

DISCO uses Hydra-style `key=value` arguments:

```bash
python runner/inference.py experiment=designable effort=max input_json_path=input.json seeds=[101,102] dump_dir=./output
```

### Key arguments

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `experiment` | `designable`, `diverse` | required | Preset: `designable` uses noisy guidance, `diverse` does not |
| `effort` | `fast`, `max` | `fast` | `fast`: 100 steps, 2 cycles. `max`: 200 steps, 4 cycles |
| `input_json_path` | path | — | Path to input JSON |
| `seeds` | list[int] | `[101,102]` | Each seed produces one sample per job |
| `num_inference_seeds` | int | null | Alternative: generates seeds `[0..N-1]` |
| `dump_dir` | path | `./output` | Output directory |
| `use_deepspeed_evo_attention` | bool | `True` | Use DeepSpeed EvoformerAttention (faster, needs CUTLASS) |
| `need_atom_confidence` | bool | `False` | Save per-atom confidence |
| `deterministic` | bool | `True` | Deterministic inference |

### Diffusion parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_diffusion.N_step` | 200 (max) / 100 (fast) | Diffusion steps |
| `model.N_cycle` | 4 (max) / 2 (fast) | Recycling cycles |
| `sample_diffusion.noise_scale_lambda` | 0.1 | Noise scale |
| `sample_diffusion.gamma0` | 0.8 (designable) / 1.6 (diverse) | Gamma |

### Noisy guidance (designable only)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_diffusion.noisy_guidance.enabled` | true | Enable/disable guidance |
| `sample_diffusion.noisy_guidance.omega_struct` | 1.5 | Structure guidance strength |
| `sample_diffusion.noisy_guidance.omega_seq` | 2.0 | Sequence guidance strength |
| `sample_diffusion.noisy_guidance.guidance_start_frac` | 0.3 | When guidance starts |
| `sample_diffusion.noisy_guidance.guidance_end_frac` | 0.8 | When guidance ends |

---

## Output Structure

```
dump_dir/
  pdbs/
    <name>_sample_<seed>.pdb          # generated structure
    <name>_sample_<seed>_ligands.txt  # ligand SMILES (if ligands present)
  sequences/
    <name>_sample_<seed>.txt          # designed sequences per chain
  ERR/
    <name>.txt                        # errors for failed samples
```

**Note on ligand naming**: DISCO assigns ligands internal residue names `l01`, `l02`, etc. in the output PDB. Downstream tools (e.g. AF3) may need explicit CCD mapping via `specified_ccd={"l01": "HEM"}` to correctly interpret these residues.

---

## Known Limitations

1. **Single protein chain only** -- multiple `proteinChain` entries will error
2. **No nucleic acid masking** -- DNA/RNA sequences must be fully specified
3. **No motif scaffolding** -- cannot fix a structural motif and scaffold around it (future)
4. **Max 99 SMILES/file ligands** per job (naming: `l01`-`l99`)
5. **Use `effort=max` for conditional generation** -- `effort=fast` recommended only for unconditional
