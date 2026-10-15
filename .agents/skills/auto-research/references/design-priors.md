# Protein-design search priors

Treat these as bounded hypotheses, not universal thresholds. The user's
`score.py` criteria and domain-specific final oracle remain authoritative.
Change one conceptual factor per matched comparison; use a compact factorial
only when the interaction is itself the hypothesis.

## General funnel order

Test cheap physical feasibility before expensive confidence or function:

1. immutable sequence/motif constraints, clashes, topology, core packing, and
   buried unsatisfied polar atoms;
2. required interface, pocket, catalytic, metal, epitope, or assembly geometry;
3. sequence plausibility, soluble-surface liabilities, and a quick apo fold;
4. final complex, multi-state, energy, or functional-equivalent oracle.

For soluble proteins, measure exposed hydrophobic liability as separate fields:
largest connected hydrophobic patch, exposed nonpolar SASA/fraction, longest
hydrophobic sequence window, and net charge at the intended pH. Do not penalize
buried core hydrophobics, required functional interfaces, or membrane-facing
surfaces. Avoid one universal cutoff; calibrate against matched expressed
controls. If no reusable metric block is available, implement one small
`transform` rather than misusing ligand `rasa`. Surface-patch scores complement,
not replace, apo fold agreement, monomericity, and expression measurements.

Run an early ligand/partner-free fold on sequences surviving cheaper checks and
compare it with the designed backbone. Default to `esmfold2` for fast soluble
single-chain triage; use `colabfold`/AlphaFold2 when its MSA, templates, or a
matched protocol are required. Predefine confidence and backbone-agreement
checks. An apo pass supports fold stability, not binding, catalysis, specificity,
assembly, or expression.

## Reusable search knobs

- Sweep a few plausible protein-length bands. Compare fold agreement, core
  packing, surface patches, functional geometry, diversity, and cost—not raw
  generation yield alone.
- Compare unbiased MPNN, global `bias_AA`, and mechanistic
  `bias_AA_per_residue` arms when sequence composition is limiting. Use classic
  `mpnn` for those exact flags; `src/blocks/utils/ligand_bias.py` can build
  geometry-derived per-residue biases. Never bias a functional surface globally
  without a mechanistic reason.
- For coordinate-conditioned generation, compare plausible atom/residue hotspot
  sets with generator settings matched. For RFD3 placement, sweep `ori_token`
  using the directional/radial lattice in `src/blocks/utils/ori_sweep.py` and
  record coordinates and tier. ORI proximity changes generated-protein COM and
  thus burial or interface enclosure.
- Start predict–redesign cycling at one cycle. Test two and at most three only
  when physical and apo gates pass while the final confidence gate—often minimum
  interface PAE—remains limiting. Score every cycle and stop without useful
  marginal gain. Do not confuse design cycles with model recycling.

## Motif grafting and scaffolding

Explore motif grafting across enzymes, metal sites, epitopes, and interfaces.
Preserve the smallest causally sufficient atomic motif, including essential
backbone contacts, proton relays, second-shell interactions, or rotamers.

Cap the first arm for one contiguous helix or loop at 10–15 motif residues.
Treat this as a search prior, not a hard scientific gate: do not truncate a
longer or discontinuous functional motif merely to satisfy the cap. Require an
explicit rationale for a larger envelope and compare nested encodings when
affordable: minimal causal atoms, atoms plus second shell/flexible flanks, and
the full structural segment.

Use indexed RFD3 motifs when sequence position and conformation are known;
consider unindexed atom-level motifs when placement or rotamer identity is part
of the search. Preserve annotations with `select_motif`, keep motif identities
fixed during MPNN when required, and use `restore_sidechains` or a bounded
`rotamer_sweep` only with subsequent clash/repack validation. Track motif atom
and insertion RMSD, rotamer state, pocket/accessibility, apo refolding, and the
domain's final functional gate. Structural mimicry alone is not function.

## Domain-specific negative design

- Soluble proteins and biologics: penalize exposed hydrophobic clusters,
  aggregation-prone sequence windows, unintended self-association, and extreme
  charge patches while protecting binding/catalytic surfaces.
- Protein interfaces: score intended binding separately from monomer folding;
  include off-targets, self-association, alternate oligomers, and site choice.
- Enzymes: check pocket access, catalytic rotamers, causal mutants, and every
  mechanistically required reaction/intermediate state; static motif RMSD is
  insufficient.
- Membrane proteins: do not apply soluble-surface hydrophobic rules to the
  membrane-exposed region. Test insertion, orientation, topology, intended
  oligomerization, soluble amyloid/off-pathway states, and cellular trafficking.
- Epitope scaffolds: require accessibility and native-like presentation, then
  domain-relevant competition, immunogenicity, or neutralization; cognate
  antibody binding alone is insufficient.

## Evidence anchors

Connected exposed hydrophobic patches can remain abnormal even when total
surface composition looks native-like ([Jacak et al., 2012](https://doi.org/10.1002/prot.23241)); surface redesign has rescued soluble expression in de novo
oligomers ([Gerben et al., 2023](https://doi.org/10.1021/acs.biochem.2c00497)).
Separate monomer and interface prediction enriched PPI binder success
([Bennett et al., 2023](https://doi.org/10.1038/s41467-023-38328-5)), while
multi-state filtering exposed why accurate static enzyme designs failed
([Lauko et al., 2025](https://doi.org/10.1126/science.adu2454)). Short 7–15
residue interface-loop grafts have succeeded in one regime
([Strauch et al., 2017](https://doi.org/10.1038/nbt.3907)), but successful
19–24-residue and discontinuous motifs show why 15 cannot be universal
([Wang et al., 2022](https://doi.org/10.1126/science.abn2100)).
