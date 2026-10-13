# Eval task: de novo binder for a flexible peptide (EIN2 site 1)

REQUIRED_N = {{REQUIRED_N}}
**TL;DR — submit exactly {{REQUIRED_N}} de novo mini-protein designs that bind the supplied
15-residue peptide *in the supplied conformation*.**

The supplied target, `target.pdb`, is the EIN2 site-1 peptide
`TVGSDGPPSFRSLSG` (15 residues, renumbered `B1`–`B15`) as chain **B**. Your
designed binder must be chain **A**.

This target is not a folded domain. It is a glycine/serine/proline-rich segment
that is disordered in isolation, so its conformation is a property of the
complex, not of the peptide. `target.pdb` therefore fixes one specific bound
conformation, and **that conformation is part of the task**: a binder that
grips the peptide in some other arrangement is solving a different problem, no
matter how confidently it does so.

`config.json` showcases the design criterias which we want to run, where the peptide is a
fixed motif spanning `B1`–`B15`, every peptide residue is a hotspot, and the
binder should diffused at 60–150 residues.


## Required quality criteria

The scorer supplies the peptide as chain B and predicts every complex from
sequence only; the peptide is not templated. It applies these public physical
constraints:

1. **Bound peptide conformation** — backbone RMSD over the buried groove
   epitope `B9–B15`, fitted and measured on that same segment, must be ≤ 1.5 Å.
   The solvent-exposed `B1–B8` flank and whole-peptide RMSDs are reported but
   are not gated.
2. **Binder structure** — the fraction of chain A assigned as loop must be
   `< 0.35`. Peptide grooves need loops, so this allowance is looser than for a
   domain–domain interface.
3. **Packing** — binder↔peptide contact molecular surface must be
   `contact_ms ≥ 300`.
4. **Sterics** — the predicted complex must contain zero backbone clashes.
5. **Interface energetics** - soft interface ddG must be `< −15` for passing design gates.
6. **High interface confidence** — ipSAE must be `≥ 0.30`, and the held-out
   structure-prediction oracle checks the designed interaction. Make sure to
   create a high-confidence interaction region to the best of your abilities.

## Deliverables

Exactly **{{REQUIRED_N}}** rows in a flat CSV table.

- Required column: `sequence`, containing only uppercase canonical
  amino-acid letters.
- Optional column: `design_id`, a unique non-empty identifier. The
  scorer generates deterministic row IDs when this column is omitted.
- Every pair of designs must have global
  Needleman–Wunsch/BLOSUM62 identity ≤ 70%, using affine gap penalties
  −10/−0.5.
- Do not submit structures, coordinates, paths, the peptide, or any other
  column.

A submission fails the output contract — before any held-out compute — if it
has a count other than {{REQUIRED_N}}, a duplicate sequence, a non-canonical
residue, or a pairwise identity above 70%.
