---
name: order-collection
description: "Collect and package all prediction structures, metrics, and pipeline summary into a handoff bundle (tar.gz + reference CSV + summary). Use when the user says 'collect orders', 'package results', 'order collection', 'I am done order these', 'bundle designs', or 'prepare handoff'."
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion", "Agent"]
---

# Order Collection Skill

Package a completed design campaign into a handoff bundle containing:
1. **Compressed archive** (`prediction_structures.tar.gz`) — all holo prediction structures + confidence data
2. **Passing-structures folder** (`passing_structures/`) — flat directory of ordered holo CIFs, one per unique sequence (quick-access, no untar needed)
3. **Reference CSV** (`reference.csv`) — every row in the tar.gz, with metrics and ordered flags (keyed by protein sequence)
4. **Passing CSV** (`passing.csv`) — slim subset: one row per unique ordered sequence, matching `passing_structures/`
5. **Campaign summary** (`campaign_summary.md`) — funnel, filter thresholds, compute accounting

## Trigger

When the user says "collect orders", "package results", "order collection", "I'm done order these", "bundle designs", "prepare handoff", or indicates they've finished a design campaign and want to collect the results.

## Important: Design Identity

Design identity is determined by **protein sequence**, NOT by filename. This is critical because partial diffusion and cycling create naming ambiguity — the same backbone can produce different MPNN sequences, and the same sequence can appear under different filenames across branches or cycles. The protein sequence is the true molecular identity (what actually gets synthesized).

Sequence fingerprints are built by concatenating protein chain sequences sorted by chain ID, e.g. `A:MKTII...|B:DPVAL...`. Non-protein chains (ligands, ions) return `None` from `get_sequences()` and are excluded.

**Critical**: Same lineage does NOT mean same ordering decision. Multiple MPNN sequences from the same backbone can have different fates.

**Two columns distinguish ordered vs. contributing designs:**
- **`ordered`**: `True` only for rows from the **holo prediction stage** (e.g. `af3_final`, `af3_direct`) whose protein sequence passed the order-gate filter. These are the actual holo (ligand-bound) structures that get ordered/synthesized.
- **`true_structure`**: `True` for ANY row across ALL stages whose protein sequence matches an ordered sequence. These are the scaffolds and intermediates (RF3 cycle 0, RF3 cycle 1, AF3 holo, AF3 apo) that belong to a final ordered design — useful for tracing a design's lineage back through the pipeline.

Both are determined by comparing each design's protein sequence against the ordered set — never by path or lineage matching.

**Holo vs. apo (IMPORTANT)**: The structure collection stage is always a **holo** prediction (ligand-bound, e.g. `af3_final`). Apo predictions (no ligand, e.g. `af3_apo`) are verification-only — they check fold stability without the ligand but are NOT the artifact to ship. Apo predictions are **excluded from `prediction_structures.tar.gz` AND from `reference.csv`** so the CSV mirrors the archive 1:1. The apo filter step (e.g. `apo_filter`) is still the right **order gate** because it certifies both holo confidence AND apo stability — but once a sequence is gated in, only its holo structure is bundled.

**Unique-sequence invariant for passing artifacts**: `passing.csv` and `passing_structures/` contain exactly one row / CIF per **unique ordered protein sequence**. Even if the same sequence appears in multiple holo-final rows (two MPNN siblings from different backbones converging on the same AA sequence), only the best representative (lowest `min_pae`, iptm tiebreak) is shipped. This guarantees `len(passing.csv) == len(passing_structures/*.cif) == len(ordered sequences from filter step)`.

## Steps

### Step 0: Locate the rundir

Identify the pipeline run directory. Check:
1. If a rundir is already known from the conversation context
2. If there's a `pipeline.json` in the current directory or an obvious subdirectory

If ambiguous, ask the user:
```
Which rundir should I collect results from?
```

Validate that `{rundir}/pipeline.json` exists. Create the output directory:
```bash
mkdir -p {rundir}/order_collection
```

### Step 1: Parse manifest and classify steps

Read `{rundir}/pipeline.json` with the Read tool. Parse the `steps` array and classify each step by its `block_name` field (NOT the `name` field, which may include cycling namespaces):

**Prediction blocks** (structure prediction oracles):
- `af3`, `boltz2`, `rf3`, `rf3_dev`, `chai1`, `protenix`, `openfold3`

**Filter blocks**:
- `filter`

**Cycling namespaces**: Steps inside `engine.cycle()` have names like `rf3_cycle/cycle_0/rf3_cycle0`. Always use `block_name` (e.g. `"af3"`) to identify the block type, not `name`.

**Always ask the user** which filter step to use for the order gate, but default the structure-collection stage to the **holo** prediction that immediately precedes the gate:
```
Which filter step should I use to determine ordered sequences?
(Confirm) I'll collect holo structures from `<block_before_gate>`. Apo predictions are verification-only and will NOT be bundled.
```
Common pattern: `apo_filter` gates sequence selection (apo fold stability) but `af3_final` / `af3_direct` / `<oracle>_cycleN` supplies the holo structure that actually gets ordered. Never collect apo (`*_apo`, `remove_ligands=True`) as the shipped structure.

**Edge cases**:
- No prediction steps found → error: "No prediction steps in this pipeline"
- No filter steps found → warn user, treat ALL designs as ordered
- Empty final filter (0 rows) → warn user: "0 designs passed the final filter"

### Step 2: Extract ordered sequences

Extract protein sequences from the final filter's output to build the "ordered" set. Run via Bash using the Chaperone venv:

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import json, os, sys
from chaperone.data.handler import PipelineDataHandler

rundir = '{rundir}'
parquet_path = os.path.join(rundir, '{final_filter_step_parquet}')
handler = PipelineDataHandler(parquet_path)

ordered_seqs = set()
for loaded in handler.get_data(which='last', load=True):
    # Protein chains only — ligand chains return None
    protein_seq = '|'.join(
        f'{k}:{v}' for k, v in sorted(loaded.sequence.items()) if v is not None
    )
    if protein_seq:
        ordered_seqs.add(protein_seq)

outdir = os.path.join(rundir, 'order_collection')
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, '_ordered_seqs.json'), 'w') as f:
    json.dump(sorted(ordered_seqs), f)
print(f'Extracted {len(ordered_seqs)} ordered sequences')
"
```

**Note**: Filter blocks do NOT add structures to the lineage list. So `which="last"` on the final filter's parquet yields the last prediction output structure — its `.sequence` gives the designed protein chains.

**Performance**: The final filter output is typically small (tens to hundreds of designs). This step is fast.

### Step 3: Collect prediction files for tar.gz (Output 1)

**Only include the best model per design** — not all seed/sample predictions. The pipeline's `compile()` step already selects the best model (via `read_most_confident_model=True`), and that's what's referenced in the output parquet. Users don't compare across seeds, so including all 5 samples would just bloat the archive.

**Exclude apo / verification-only prediction steps** from the tar.gz. Apo predictions (step names ending in `_apo`, or any step run with `remove_ligands=True`) are not part of the shipped deliverable — they only inform the order gate. The archive bundles the holo lineage (RFD3/MPNN cycles + holo final oracle) so the recipient can see what was actually designed. Programmatically: skip steps whose `name` ends with `_apo`, plus any step whose `remove_ligands` field is `true` in `.pipeline/conf.<block_name>.yaml`. (`pipeline.json`'s `StepInfo` does **not** persist the block's pydantic config — only `.pipeline/conf.<block>.yaml` does.)

For each remaining prediction step, read its output parquet to get the **selected best-model paths**, then collect each structure file + its companion confidence files:

| Block | Confidence file patterns (same directory as structure) |
|-------|-------------------------------------------------------|
| AF3 | `*_summary_confidences.json`, `*_confidences.json` |
| Boltz2 | `confidence_*.json` |
| RF3 | `*_summary_confidences.json` |
| Chai1 | `scores.*.npz` |
| Protenix | `*_summary_confidence_*.json` |
| OpenFold3 | `*_confidences_aggregated.json` |

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import os, re, json, yaml
import polars as pl

rundir = '{rundir}'
manifest = json.load(open(os.path.join(rundir, 'pipeline.json')))
prediction_blocks = {'af3', 'boltz2', 'rf3', 'rf3_dev', 'chai1', 'protenix', 'openfold3'}
outdir = os.path.join(rundir, 'order_collection')

def is_apo(step, _cache={}):
    # StepInfo in pipeline.json does NOT persist the block's pydantic config,
    # so step.get('config') is always absent. Primary signal is the name suffix;
    # fall back to reading .pipeline/conf.<block_name>.yaml where the resolved
    # config (with remove_ligands) is actually stored.
    if step['name'].endswith('_apo'):
        return True
    block = step['block_name']
    if block not in _cache:
        try:
            with open(os.path.join(rundir, '.pipeline', f'conf.{block}.yaml')) as fh:
                _cache[block] = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _cache[block] = {}
    return bool(_cache[block].get('steps', {}).get(step['name'], {}).get('remove_ligands'))

def best_model_companions(cif_path, _parent_cache={}):
    '''Return files in cif_path's parent dir that belong to this best-model CIF.
    Replaces the prior parent-wide \"*confidences*.json/*scores*.npz\" glob, which
    on Boltz2/Chai1/Protenix-cycle layouts pulled all 5 samples per design.
    Caches one os.listdir() per parent directory to replace the per-file
    NFS-cache-busting fh.read(1) that hangs at 20K+ files against /net/scratch.
    '''
    parent = os.path.dirname(cif_path)
    cif_stem = os.path.splitext(os.path.basename(cif_path))[0]
    # Best-model suffix discriminators:
    #   <design>_model       AF3 / OpenFold3 / Chai1 (CIF is often a symlink)
    #   <design>_model_<N>   Boltz2
    #   <design>_sample_<N>  Protenix cycle
    m = re.match(r'^(.*?)_(model(?:_\d+)?|sample_\d+)$', cif_stem)
    design_prefix = m.group(1) if m else cif_stem
    model_suffix  = m.group(2) if m else None
    if parent not in _parent_cache:
        try:
            _parent_cache[parent] = set(os.listdir(parent))
        except OSError:
            _parent_cache[parent] = set()
    entries = _parent_cache[parent]
    found = {cif_path}
    # Chai1 only: best-model CIF is a symlink to pred.model_idx_<N>.cif; pull
    # its scores.model_idx_<N>.npz companion (and the real CIF target).
    if os.path.islink(cif_path):
        tgt = os.readlink(cif_path)
        idx_match = re.search(r'model_idx_(\d+)', tgt)
        if idx_match:
            tag = f'model_idx_{idx_match.group(1)}'
            for e in entries:
                if tag in e:
                    found.add(os.path.join(parent, e))
    for e in entries:
        if not e.endswith(('.json', '.npz')):
            continue
        ent_stem = os.path.splitext(e)[0]
        # Boltz2-style: 'confidence_<design>_model_0', 'pae_<design>_model_0'.
        if model_suffix and ent_stem.endswith(f'_{model_suffix}') and design_prefix in ent_stem:
            found.add(os.path.join(parent, e))
            continue
        # AF3 / OpenFold3 / Protenix af3_final: confidence file starts with
        # <design>_ and the samples live in subdirectories (listdir misses them).
        # Guard against a neighbouring sample file whose index differs from ours.
        if e.startswith(design_prefix + '_'):
            other = re.findall(r'_(?:model|sample)_(\d+)', e)
            ours  = re.search(r'_(\d+)$', model_suffix or '')
            if ours and other and ours.group(1) not in other:
                continue
            found.add(os.path.join(parent, e))
    return found

all_files = []
for step in manifest['steps']:
    if step['block_name'] not in prediction_blocks:
        continue
    if is_apo(step):  # skip apo / verification-only predictions
        continue
    pq_path = os.path.join(rundir, step['parquet'])
    if not os.path.exists(pq_path):
        print(f'WARNING: parquet not found: {pq_path}')
        continue
    df = pl.read_parquet(pq_path)
    step_name = step['name']
    # Blocks like Chai1 register the structure under the design's name, not the
    # step's, so strict name equality silently drops every row from that block.
    # Fall back to 'structure path lives under this step's output directory'.
    step_out_dir = os.path.join(rundir, os.path.dirname(step['parquet'])) + os.sep
    for row in df.iter_rows(named=True):
        for s in row['structures']:
            path = s.get('path') or s.get('path_handle')
            if not path:
                continue
            if s['name'] != step_name and not str(path).startswith(step_out_dir):
                continue
            if not os.path.exists(path):
                continue
            all_files.extend(best_model_companions(path))

# Deduplicate and make relative to rundir
rel_files = sorted(set(os.path.relpath(f, rundir) for f in all_files if os.path.exists(f)))

filelist_path = os.path.join(outdir, '_filelist.txt')
with open(filelist_path, 'w') as f:
    f.write('\n'.join(rel_files))
print(f'Found {len(rel_files)} files ({len([f for f in rel_files if any(f.endswith(e) for e in (\".cif\", \".cif.gz\", \".pdb\"))])} structures + confidence data)')
"
```

Then create the tar.gz. Prefer `pigz` (parallel gzip) when available — ~4× faster than single-threaded `gzip` on the ~1GB archives typical at 10K-row scale:
```bash
cd {rundir} && if command -v pigz >/dev/null 2>&1; then
  tar --use-compress-program="pigz -p $(nproc)" -cf order_collection/prediction_structures.tar.gz -T order_collection/_filelist.txt
else
  tar -czf order_collection/prediction_structures.tar.gz -T order_collection/_filelist.txt
fi
```

Report the archive size to the user.

### Step 4: Build reference CSV (Output 2)

**Optimization**: Metrics come from the parquet JSON column (already computed by the folding oracle). Sequence extraction still requires loading structure files (unavoidable for sequence-based ordering), parallelized via `multiprocessing.Pool`. **Use processes, NOT threads.** Benchmarked on this codebase: threads=16 runs ~25× *slower* than sequential on 10K rows (GIL contention), processes=16 give ~7× speedup (43ms → 6ms per row). The docstring at `src/chaperone/data/handler.py:771-773` claims `parse_to_atom_array` releases the GIL — this is not borne out in practice; atomworks' Python-level work dominates and holds the GIL.

Main process reads parquets (metrics + paths in memory), then `Pool.map` dispatches sequence extraction to workers.

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import json, os, csv, yaml
from multiprocessing import Pool
import polars as pl
from chaperone.io import parse_to_atom_array, get_sequences

def _extract_sequence(path):
    '''Worker: parse structure, return sequence fingerprint or error sentinel.'''
    try:
        arr = parse_to_atom_array(path, assign_types=False)
        seq = get_sequences(arr)
        return '|'.join(f'{k}:{v}' for k, v in sorted(seq.items()) if v is not None)
    except Exception as e:
        return f'__ERROR__:{type(e).__name__}:{e}'

rundir = '{rundir}'
manifest = json.load(open(os.path.join(rundir, 'pipeline.json')))

with open(os.path.join(rundir, 'order_collection', '_ordered_seqs.json')) as f:
    ordered_seqs = set(json.load(f))

prediction_blocks = {'af3', 'boltz2', 'rf3', 'rf3_dev', 'chai1', 'protenix', 'openfold3'}
# Leave 4 cores free; cap at 24 (diminishing returns past that).
NUM_WORKERS = max(1, min(os.cpu_count() or 4, 24) - 4)

def is_apo(step, _cache={}):
    # See Step 3 rationale: step['config'] is always absent in pipeline.json;
    # .pipeline/conf.<block_name>.yaml is the authoritative config source.
    if step['name'].endswith('_apo'):
        return True
    block = step['block_name']
    if block not in _cache:
        try:
            with open(os.path.join(rundir, '.pipeline', f'conf.{block}.yaml')) as fh:
                _cache[block] = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            _cache[block] = {}
    return bool(_cache[block].get('steps', {}).get(step['name'], {}).get('remove_ligands'))

# Collect (path, step_name, bare, step_metrics) tuples across all non-apo prediction steps.
# Metrics come straight from the parquet JSON column — no structure loading needed.
work_items = []
for step in manifest['steps']:
    if step['block_name'] not in prediction_blocks:
        continue
    if is_apo(step):
        continue
    step_name = step['name']
    bare = step_name.rsplit('/', 1)[-1]
    pq_path = os.path.join(rundir, step['parquet'])
    if not os.path.exists(pq_path):
        print(f'WARNING: parquet not found for {step_name}: {pq_path}')
        continue
    df = pl.read_parquet(pq_path)
    for row in df.iter_rows(named=True):
        path = None
        for s in row['structures']:
            # Match on full namespaced name OR bare block name
            # (manual compilation may use bare names like 'rf3_cycle1'
            #  instead of 'rf3_cycle/cycle_1/rf3_cycle1').
            if s['name'] == step_name or s['name'] == bare:
                path = s.get('path') or s.get('path_handle')
                break
        if not path:
            continue
        metrics_raw = row['metrics']
        metrics = json.loads(metrics_raw) if isinstance(metrics_raw, str) else metrics_raw
        step_metrics = metrics.get(step_name, metrics.get(bare, {}))
        work_items.append((path, step_name, bare, step_metrics))

print(f'Extracting sequences for {len(work_items)} designs with {NUM_WORKERS} processes...', flush=True)

# Parallel sequence extraction. chunksize=32 reduces IPC overhead for tiny work units.
with Pool(NUM_WORKERS) as pool:
    sequences = pool.map(_extract_sequence, [w[0] for w in work_items], chunksize=32)

# Stitch rows together.
rows = []
n_failed = 0
step_counts = {}
for (path, step_name, bare, step_metrics), protein_seq in zip(work_items, sequences):
    if protein_seq and protein_seq.startswith('__ERROR__:'):
        n_failed += 1
        if n_failed <= 3:
            print(f'WARNING: {step_name}: {protein_seq[10:]}')
        continue
    if protein_seq is None:
        continue
    # Sequence matching (NOT path/lineage-based).
    # structure_step = user-specified prediction step for structure collection (e.g. 'af3_final').
    seq_in_ordered = protein_seq in ordered_seqs
    rows.append({
        'sequence': protein_seq,
        'design_name': os.path.splitext(os.path.basename(path))[0],
        'structure_path': os.path.relpath(path, rundir),
        'design_stage': step_name,
        'ordered': seq_in_ordered and step_name == structure_step,
        'true_structure': seq_in_ordered,
        'min_pae': step_metrics.get('min_pae_interaction', step_metrics.get('min_pae')),
        'plddt': step_metrics.get('plddt'),
        'iptm': step_metrics.get('iptm'),
        'ptm': step_metrics.get('ptm'),
        'ranking_score': step_metrics.get('ranking_score'),
        'ligand_plddt': step_metrics.get('ligand_plddt'),
    })
    step_counts[step_name] = step_counts.get(step_name, 0) + 1

for sn, c in step_counts.items():
    print(f'  {sn}: {c}')

csv_path = os.path.join(rundir, 'order_collection', 'reference.csv')
fieldnames = ['sequence', 'design_name', 'structure_path', 'design_stage',
              'ordered', 'true_structure', 'min_pae', 'plddt', 'iptm', 'ptm',
              'ranking_score', 'ligand_plddt']
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

n_ordered = sum(1 for r in rows if r['ordered'])
print(f'\\nWrote {len(rows)} rows to {csv_path}')
print(f'  Ordered: {n_ordered} / {len(rows)}, failed: {n_failed}')
"
```

**Design decisions**:
- **Process-based parallelism, NOT threads**: Benchmarked on 500 rows in this repo: sequential 43ms/row, `ThreadPoolExecutor(16)` 205ms/row (5× *slower* — GIL contention), `multiprocessing.Pool(16)` 6ms/row (7× speedup). 500/500 sequences matched between paths. Pickle cost is tiny because workers receive only string paths and return short strings.
- **Metrics from parquet**: the main process reads all parquets and extracts metrics dicts before parallelizing. Workers only touch structure files.
- **Lightweight parsing**: worker uses `parse_to_atom_array(path, assign_types=False)` — skips slow type assignment, gets sequence via `get_sequences()`.
- **Sequence-based ordered flag**: Each design's protein sequence is compared against `ordered_seqs`. Same lineage does NOT imply same ordering — multiple MPNN siblings from the same backbone can have different fates.
- **Relative paths**: `os.path.relpath(path, rundir)` for portability with the tar.gz.
- **Apo predictions**: `structure_step` must be a **holo** step (e.g. `af3_final`). Apo rows appear in the CSV with `ordered=False` (verification-only) but still get `true_structure=True` when their sequence matches an ordered one — so lineage tracing still works. Do NOT set `structure_step='af3_apo'`; the shipped structure must be ligand-bound.

### Step 4b: Passing artifacts (slim CSV + folder, one per unique sequence)

After the main `reference.csv` is written, emit the ordered subset: one row / CIF per **unique protein sequence** that passed the order gate. Dedup is required because the same AA sequence can appear in multiple holo-final rows (e.g. two MPNN siblings from different RFD3 backbones converging on the same sequence).

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import os, csv, shutil, polars as pl

rundir = '{rundir}'
outdir = os.path.join(rundir, 'order_collection')
ref = pl.read_csv(os.path.join(outdir, 'reference.csv'))
ordered = ref.filter(pl.col('ordered') == True)  # holo-stage rows only

# Dedup by sequence — keep the lowest min_pae (iptm tiebreaker)
# Sort so first-kept-per-sequence is the best
ordered_sorted = ordered.sort(['min_pae', 'iptm'], descending=[False, True], nulls_last=True)
passing = ordered_sorted.unique(subset=['sequence'], keep='first', maintain_order=True)

# Slim CSV — keep the useful columns, drop lineage ones
passing_cols = ['sequence','design_name','structure_path','min_pae','plddt',
                'iptm','ptm','ranking_score','ligand_plddt']
passing.select([c for c in passing_cols if c in passing.columns]).write_csv(
    os.path.join(outdir, 'passing.csv')
)

# Folder of CIFs (flat, one per unique sequence, named by design_name)
pass_dir = os.path.join(outdir, 'passing_structures')
os.makedirs(pass_dir, exist_ok=True)
n_copied = n_missing = 0
for row in passing.iter_rows(named=True):
    src = os.path.join(rundir, row['structure_path'])
    if os.path.exists(src):
        # Preserve original extension (.cif, .cif.gz, .pdb)
        ext = ''.join(s for s in ['.gz','.cif','.pdb','.cif.gz'] if src.endswith(s)) or os.path.splitext(src)[1]
        shutil.copy2(src, os.path.join(pass_dir, row['design_name'] + ext))
        n_copied += 1
    else:
        n_missing += 1

print(f'passing.csv: {passing.height} unique sequences')
print(f'passing_structures/: {n_copied} CIFs copied, {n_missing} missing')
"
```

**Invariants** (sanity-check after running):
- `passing.csv` row count == number of unique ordered sequences (what the filter step produced).
- `ls passing_structures/*.cif | wc -l` == `passing.csv` row count.
- Every `structure_path` in `passing.csv` appears in `reference.csv` with `ordered=True`.

### Step 4 (alternative): Block-based CSV for very large campaigns (>200K designs)

For campaigns with >200K prediction designs, threaded local iteration is bounded by the driver host's core count. Escape hatch: `OrderCollectionBlock` at `src/blocks/util_blocks/order_collection.py` (`_chunked = True`) parallelizes across SLURM nodes.

```python
from blocks import Blocks
from chaperone import ExecutionEngine

rundir = '{rundir}'
engine = ExecutionEngine(rundir=rundir, reinit=False)

# Ordered sequences from Step 2
ordered_seqs = [...]  # loaded from _ordered_seqs.json

# Run for each prediction step — SLURM-chunked
for step_name in prediction_step_names:
    engine.run(
        Blocks.get("order_collection")(
            name=f"oc_{step_name}",
            prediction_step=step_name,
            ordered_sequences=ordered_seqs,
        ),
        data=engine.compile(steps=[step_name]),
        job_args={"partition": "cpu", "cpus_per_task": 8, "rows_per_job": 2000, "time_limit": "1:00:00"},
    )

# Compile and export CSV
handler = engine.compile()
df = handler.unpack(to_pandas=True)
oc_cols = [c for c in df.columns if c.startswith("oc_")]
df[oc_cols].to_csv(f"{rundir}/order_collection/reference.csv", index=False)
```

**When to use which approach (cutoffs are for the dominant prediction step, not total rows):**
- **< 20K designs**: Direct iteration with `num_workers=16` (Step 4 above) — ~1 min CSV build on a 20-core host. SLURM would queue/overhead-dominate.
- **20K – 200K designs**: Still direct iteration; bump `NUM_WORKERS` to `min(nproc, 24)` and tar with `pigz -p $(nproc)`. Single-host, no SLURM overhead.
- **> 200K designs**: Block-based (this section). Use `rows_per_job=2000` to keep per-chunk compute above ~2 min — smaller chunks fall into the NFS attr-cache race zone on sub-minute SLURM jobs. Each chunk itself runs threaded (`num_workers=8` inside the block) given `cpus_per_task=8`.

### Step 5: Generate campaign summary (Output 3)

Build a human-readable markdown summary. Collect information from multiple sources:

#### 5A. Funnel statistics

Read each step's output parquet row count:

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import json, os
import polars as pl

rundir = '{rundir}'
manifest = json.load(open(os.path.join(rundir, 'pipeline.json')))

print('| Step | Block | Rows |')
print('|------|-------|------|')
for step in manifest['steps']:
    pq = os.path.join(rundir, step['parquet'])
    if os.path.exists(pq):
        n = pl.read_parquet(pq).height
        print(f'| {step[\"name\"]} | {step[\"block_name\"]} | {n} |')
    else:
        print(f'| {step[\"name\"]} | {step[\"block_name\"]} | ? |')
"
```

#### 5B. Pipeline configuration

Check if `{rundir}/.pipeline/conf.yaml` exists. If not, generate it:

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
from chaperone import ExecutionEngine
e = ExecutionEngine(rundir='{rundir}', reinit=False)
e.to_yaml()
print('Generated conf.yaml')
"
```

Then read the conf.yaml with the Read tool.

#### 5C. Filter thresholds (CRITICAL for reproducibility)

Extract filter definitions from **all filter steps** so the campaign can be exactly reproduced. Two complementary approaches:

**Approach 1 — Parse `conf.filter.yaml`**: Read `.pipeline/conf.filter.yaml`. **Known limitation**: multiple filter steps share the same filename — only the last-written survives. This captures only the final filter.

**Approach 2 — Parse the pipeline script** (preferred, captures all filters): Locate the pipeline script:
1. Ask the user or check conversation context for the script path
2. In the rundir itself (`*.py`)
3. In the current working directory

Read the script with the Read tool and extract every `MetricFilter(...)` instance along with the filter key it's assigned to. Format as a structured table per filter step:

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import ast, re, json, os, yaml

# Read conf.filter.yaml for the last filter step
conf_path = os.path.join('{rundir}', '.pipeline', 'conf.filter.yaml')
if os.path.exists(conf_path):
    with open(conf_path) as f:
        fc = yaml.safe_load(f)
    print('=== From conf.filter.yaml (last filter only) ===')
    for step_name, step_cfg in fc.get('steps', {}).items():
        print(f'\n{step_name}:')
        for key, filt in (step_cfg.get('filters') or {}).items():
            parts = []
            if filt.get('min') is not None: parts.append(f'min={filt[\"min\"]}')
            if filt.get('max') is not None: parts.append(f'max={filt[\"max\"]}')
            if filt.get('quantile') is not None: parts.append(f'quantile={filt[\"quantile\"]} ({filt.get(\"quantile_direction\", \"lower\")})')
            if filt.get('parent_based'): parts.append('parent_based')
            print(f'  {key}: {', '.join(parts)}')
        for grp_name, grp in (step_cfg.get('filter_groups') or {}).items():
            print(f'  [group {grp_name}]:')
            for key, filt in grp.items():
                parts = []
                if filt.get('min') is not None: parts.append(f'min={filt[\"min\"]}')
                if filt.get('max') is not None: parts.append(f'max={filt[\"max\"]}')
                print(f'    {key}: {', '.join(parts)}')
"
```

**Approach 3 — Script-based extraction**: If the pipeline script path is known, read it with Read tool and manually extract all filter blocks. Look for patterns like:
- `filters={...}` dicts containing `MetricFilter(...)`
- `filter_groups={...}` dicts
- Filter variable names (e.g. `RFD3_FILTERS`, `AF3_FILTERS`)

Include ALL filters in the summary under a dedicated "## Filter Thresholds" section with one subsection per filter step. This is the single most important section for reproducibility — without it, the exact pass criteria are lost.

#### 5D. Compute accounting

Extract from manifest execution summaries:

```bash
source {chaperone_repo}/.venv/bin/activate && python3 -c "
import json, os

rundir = '{rundir}'
manifest = json.load(open(os.path.join(rundir, 'pipeline.json')))

print('| Step | Wall (h) | GPU (h) | Records | OK/Fail |')
print('|------|----------|---------|---------|---------|')
total_wall = total_gpu = total_records = total_ok = total_fail = 0
for step in manifest['steps']:
    ex = step.get('execution_summary')
    if ex and ex.get('records', 0) > 0:
        wall_h = ex['wall_seconds'] / 3600
        gpu_h = ex.get('gpu_seconds', 0) / 3600
        ok = ex.get('successful_records', 0)
        fail = ex.get('failed_records', 0)
        print(f'| {step[\"name\"]} | {wall_h:.2f} | {gpu_h:.2f} | {ex[\"records\"]} | {ok}/{fail} |')
        total_wall += wall_h
        total_gpu += gpu_h
        total_records += ex['records']
        total_ok += ok
        total_fail += fail
print(f'| **TOTAL** | **{total_wall:.2f}** | **{total_gpu:.2f}** | **{total_records}** | **{total_ok}/{total_fail}** |')
"
```

#### 5E. Assemble the summary

Use the Write tool to create `{rundir}/order_collection/campaign_summary.md` with:

```markdown
# Campaign Summary

## Overview
- **Rundir**: {rundir}
- **Date**: {today's date}
- **Total designs generated**: {from first prediction step}
- **Total designs ordered**: {from ordered set count}
- **Pipeline script**: {path to script if found}

## Funnel Statistics
{table from 5A}

## Filter Thresholds
{from conf.yaml + pipeline script}

## Pipeline Configuration
```yaml
{contents of conf.yaml}
```

## Compute Accounting
{table from 5D}

## Key Parameters
{extracted from conf.yaml: ligand info, batch sizes, diffusion settings, etc.}
```

### Step 6: Report to user

Print a summary:
```
Order collection complete! Files saved to {rundir}/order_collection/:
  - prediction_structures.tar.gz ({size})  — all holo structures + confidence
  - passing_structures/ ({N_passing} CIFs) — ordered subset, one per unique sequence
  - reference.csv ({N_rows} rows)          — mirrors the tar.gz
  - passing.csv ({N_passing} rows)         — slim ordered subset
  - campaign_summary.md                     — reproducibility report

{any warnings about missing files or failed extractions}
```

Clean up temp files:
```bash
rm -f {rundir}/order_collection/_ordered_seqs.json {rundir}/order_collection/_filelist.txt
```

## Caveats

1. **Metrics come from parquet, sequences require loading structures**: Metrics (min_pae, plddt, etc.) are read directly from the parquet JSON column — no file I/O needed. Sequence extraction still requires loading each structure file, but `multiprocessing.Pool(16)` parallelizes the parse. Measured: 10,221 rows in 64s (5.5ms/row, 9.4× faster than sequential baseline). Do NOT use threads — benchmarked 5× slower than sequential due to GIL contention in `atomworks.parse`; the docstring in `src/chaperone/data/handler.py:771-773` is inaccurate for this workload.

2. **Filter config collision**: Multiple filter steps share `.pipeline/conf.filter.yaml` — only the last-written survives. The summary supplements this by reading the pipeline script when available.

3. **`compile_as_metric=True`**: If a prediction step used this mode, its structure path is in `path_handle` instead of `path`. The `get_data(which="last")` call handles this transparently via `LoadedStructure._resolve_path()`.

4. **Non-standard amino acids**: Mapped to "X" in the sequence fingerprint. Two designs differing only in non-standard AA placement may collide.

5. **Passing folder is one-CIF-per-unique-sequence**: `passing_structures/{design_name}.cif`. If multiple holo-final rows share the same protein sequence (e.g. two MPNN siblings from different RFD3 backbones that converged on the same AA sequence), only the lowest-`min_pae` (iptm tiebreak) representative is copied. `passing.csv` is deduplicated the same way, so `len(passing.csv) == len(passing_structures/*.cif) == len(ordered sequences from the filter step)`.
