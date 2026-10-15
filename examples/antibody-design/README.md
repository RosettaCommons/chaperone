# Antibody CDR-H3 redesign

Uses regular public RFD3 to remodel an eight-residue H3 loop in the supplied
8S6Z Fv–CD28 complex, then ProteinMPNN and AF3. Framework, light chain and antigen
are fixed generation context; ProteinMPNN keeps their sequences fixed.

```bash
python examples/antibody-design/pipeline.py --test --rundir=runs/antibody-smoke
```

Requires Foundry, LigandMPNN and AF3 runtimes/weights. Add `--local` on an allocated
GPU. No private antibody checkpoint or abMPNN-PoE is required. General RFD3 is
not validated here as a replacement for the antibody-tuned model's quality.

See the [workflow guide](../../docs/site/workflows/antibody-design.md) for precise
input cropping, numbering, limits and validation. The input derives from PDB
8S6Z in Foundry's public binder-design tutorial. `files/residue_mapping.json`
maps its source author labels (including insertions) to the bundled PDB.

GPU check: one complete RFD3 → packed ProteinMPNN → AF3 → RMSD run passed on an
H200. Packed residue identities matched the sampled sequence and retained all fixed
sequences. With Chaperone's packing correction, all 2,628 observed fixed heavy
atoms exactly retained their staged input coordinates. AF3 pLDDT was 83.5,
complex backbone RMSD 6.23 Å; no quality success
is claimed. See the workflow guide for runtime revisions and limitations.
