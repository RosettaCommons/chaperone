---
name: order-designs
description: Generate DNA eblock orders from design PDBs using the cowboy-ssm gene-order wrapper (JohnBercow + preflight/QC/retry) on IPD digs.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "Edit", "AskUserQuestion"]
---

# Order Designs Skill

Help users generate DNA eblock orders from protein design PDB files using **`cowboy-ssm gene-order`** at `/net/software/lab/cowboy_ssm_beta/`. The wrapper drives JohnBercow under the hood but adds preflight PDB apo-ification, post-run QC gates (hairpins, homopolymers, off-target Type IIS sites, cryptic E. coli motifs, adapter/vector consistency, protein round-trip), and optional best-of retry — all the bad designs that raw JohnBercow would have silently shipped to IDT are caught here.

## Trigger

When the user asks to "order designs", "generate eblocks", "order eblocks", "golden gate", "run johnbercow", "prep for ordering", or wants to go from PDB structures to DNA orders.

**Route to `/order-ssm` instead if the user mentions:** SSM, site-saturation, alanine scan, library, combinatorial mutants, round 2, beneficial fragments, IPD Blocks chip, GGA assembly, fragment reuse. That skill drives `cowboy-ssm generate` for multi-variant library design.

## Steps

### 1. Check Environment

Verify the cowboy-ssm launcher is accessible and its resources resolve:
```bash
[ -x /net/software/lab/cowboy_ssm_beta/cowboy-ssm ] && echo "cowboy-ssm: OK" || echo "cowboy-ssm: NOT FOUND"
/net/software/lab/cowboy_ssm_beta/cowboy-ssm env-check --vector LM0627
```

`env-check` confirms the apptainer image, BsaI fidelity CSV, JohnBercow path, and Python deps. If any line reports missing, the user is likely not on IPD digs.

**Non-IPD fallback:** `apptainer exec /net/software/containers/universal.sif /net/software/lab/cowboy_ssm_beta/main.py gene-order ...` — use only if the launcher is unavailable.

**Raw JohnBercow escape hatch:** if the user specifically needs `--no_adapters` (Twist / custom PCR adapters), fall back to `/net/software/lab/johnbercow/JohnBercow_v2.py` — `cowboy-ssm gene-order` does not expose this flag yet. Surface this as a known gap.

### 2. Gather Order Parameters

Use `AskUserQuestion` to collect required parameters. Show defaults where applicable:

- **Input** — PDB directory (`--order-pdbs`) *or* protein FASTA (`--order-fasta`); can be combined
- **Order name** — e.g. `2026_0418_GG16` → `--order-name`
- **Design prefix** — e.g. `LM` → `--design-prefix`
- **Starting design ID** — e.g. `1503` (accepts `0001` zero-padded) → `--design-id`
- **Vector** — see step 3
- **Species** — `e_coli` (default) / `h_sapiens` / `s_cerevisiae` → `--species`
- **Plate layout** — standard tubes or ECHO 384-well (`--echo`)
- **QC strictness** (new vs. old skill):
  - **Standard** (default): `--fail-on-hairpin medium --fail-on-poly-run 6`
  - **Strict / production** (recommended for final orders): `--fail-on-hairpin medium --fail-on-poly-run 6 --fail-on-rare-codon-cluster --fail-on-cryptic-start --johnbercow-runs 3 --qc-reruns 1`
  - **Permissive** (testing only): all gates off; also add `--allow-extra-enzyme-sites` if needed
- **Dry-run first?** — yes/no. If yes, run the command with `--dry-run` first to surface preflight issues, then re-run for real on user confirmation.

### 3. Show Available Vectors

Call `list-vectors` live — the registry has 40+ entries and evolves over time, so dynamic listing beats a stale hardcoded table:

```bash
/net/software/lab/cowboy_ssm_beta/cowboy-ssm list-vectors
# Or filter by enzyme:
/net/software/lab/cowboy_ssm_beta/cowboy-ssm list-vectors --enzyme BsaI
```

Pipe the output directly to the user for vector selection.

**Vector selection guidance:**
- When in doubt between a vector and its promoter variant, pick the **promoter** version — it does not change function.
- All ccdb vectors must be propagated in **NEB Stable** cells (resistant to the lethal ccdb gene).
- Non-standard stickies (e.g. LM1423 `TATG`/`TAAT`) are **not compatible** with the standard `AGGA`/`TTCC` vectors — pick one family and stick with it for a given order.
- Physical vector FASTAs: `/net/software/lab/johnbercow/entry_vectors/`
- Plasmid maps (SnapGene): `/home/lmilles/0proteins/__orders/plasmids`

### 4. Build and Run the Command

Construct the command. All flags are hyphenated (was underscored in raw JohnBercow):

```bash
/net/software/lab/cowboy_ssm_beta/cowboy-ssm gene-order \
    --order-pdbs <PDB_DIR> \
    --order-name <ORDER_NAME> \
    --design-prefix <PREFIX> \
    --design-id <START_ID> \
    --gg-vector <VECTOR> \
    --species <SPECIES> \
    --pdb-preprocess apo \
    [--echo] \
    [--fail-on-hairpin medium] [--fail-on-poly-run 6] \
    [--fail-on-rare-codon-cluster] [--fail-on-cryptic-start] \
    [--johnbercow-runs 3] [--qc-reruns 1] \
    -o <OUTPUT_DIR>
```

**Notes:**
- `--pdb-preprocess apo` (default) strips HETATM records before handing to JohnBercow — lingering ligands/waters will otherwise silently corrupt the ORF. Use `--pdb-preprocess as-is` only for PDBs you've already cleaned.
- `--no-layout` is **default on** (matching current lab usage); pass nothing to keep it.
- `--gg-vector` accepts multiple vectors space-separated if you want to order the same designs into several vectors at once.
- Old `--no_adapters` (Twist / custom PCR adapters) is not exposed; fall back to `JohnBercow_v2.py` with a warning if needed.
- First do a `--dry-run` if the user opted in; confirm preflight passes before running for real.

### 5. Parse the Post-Run Manifests

After the command completes, inspect the two manifest JSONs in `<OUTPUT_DIR>`:

```bash
# High-level run status, new files, warnings
cat <OUTPUT_DIR>/*__cowboy_gene_order_manifest.json | jq '.returncode, .warnings, .new_files | keys'

# Per-eblock QC details
cat <OUTPUT_DIR>/*__cowboy_gene_order_qc_manifest.json | jq '.post_qc.issue_counts, [.post_qc.per_eblock[] | select(.qc_issues | length > 0) | {name: .eblock_name, issues: .qc_issues}]'
```

**Summarize to the user:**
- Total eblocks generated vs. failed (look for a `*_FAILED.fa` file — this means at least one design could not be converted).
- QC issue counts by type (hairpin / homopolymer / off-target enzyme site / rare codon / cryptic motif / adapter mismatch / failed expression check).
- For each failed eblock, the issue type + position + severity.
- If `--johnbercow-runs > 1`, which attempt was selected per design and the QC score distribution across attempts. Merged best-of outputs are at `*__cowboy_best_of_df.csv` / `*__cowboy_best_of.xlsx` / `*__cowboy_best_of.fa`.

Exit code 0 = all gates passed. Non-zero = at least one gate failed; partial output may still be written. Point the user at the QC manifest for the rationale.

### 6. Post-Order Checklist

After a successful order, remind the user:

1. **Verify GG assembly** — Run the reaction in SnapGene, Benchling, or the NEB Golden Gate tool: https://goldengate.neb.com/
2. **Order eblocks on IDT** — Request them in **IDTE buffer** (10 mM Tris, 0.1 mM EDTA) for long-term DNA stability, **not** in H2O.
3. **Propagating GG plasmids (NEB Stable):**
   - Always grow from **fresh plated transformations** — do NOT grow directly from glycerol stocks (always plate first).
   - Use **LB media** (not TB2 or other rich media).
   - Transform into NEB Stable cells, plate on LB-Kan, grow overnight at 37°C (or longer at 30°C).
   - Pick clones, grow 50 mL overnight cultures for DNA midipreps in LB.
4. **Selection mechanism** — The ccdb lethal gene kills standard E. coli (DH5α, BL21). Successful GG assembly excises ccdb, so only correctly assembled plasmids survive.

## References

- Gene-order details and flag reference: `/net/software/lab/cowboy_ssm_beta/docs/GENE_ORDER_JOHNBERCOW.md`
- Full CLI parameter reference: `/net/software/lab/cowboy_ssm_beta/docs/CLI_PARAMETERS.md`
- File formats and output schemas: `/net/software/lab/cowboy_ssm_beta/docs/INPUTS_AND_OUTPUTS.md`
- QC failure semantics (what each issue type means, blocking vs. advisory): `/net/software/lab/cowboy_ssm_beta/docs/VALIDATION_AND_WARNINGS.md`
- Sister skill for SSM libraries: `/order-ssm`
