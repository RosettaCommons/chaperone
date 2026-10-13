# Third-party notices

Chaperone-authored code and documentation use the MIT license in `LICENSE`.
That project license does not replace the notices or terms attached to bundled
third-party code, example outputs, external tools, model weights, or databases.
The following bundled material has its own notices.

## AlphaFold 3 runner

`src/blocks/scripts/run_alphafold_AF.py` contains an adapted AlphaFold 3 runner.
Its original header identifies:

- Copyright 2024 DeepMind Technologies Limited.
- [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The file retains that copyright and license notice. It is included in Chaperone
wheel and source distributions; its upstream license is not replaced by MIT.
Chaperone's adaptations include manifest and sequence input handling, configured
model seeds, template handling, and execution/resume integration.

The runner separately points to the upstream
[AlphaFold 3 model access process](https://github.com/google-deepmind/alphafold3)
and [model parameter terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).
Model parameters are not included in Chaperone distributions.

## AlphaFold 3 example outputs

The following files under `src/chaperone-evals/evals_task/` retain embedded
`_pdbx_data_usage` notices referring to the
[AlphaFold 3 output terms](https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md),
including a non-commercial-use notice:

- `antibody_cdr/example/top_design.cif`
- `biotin_binder_design/example/top_design.cif`
- `dna_binder/example/top_design.cif`
- `enzyme_glycosidase/example/top_design.cif`
- `enzyme_protease/example/top_design.cif`
- `nanoparticle_c3/example/top_design.cif`
- `peptide_binder/example/top_design.cif`
- `ppi_pdl1/example/top_design.cif`
- `rna_binder/example/top_design.cif`

These notices remain intact. These files are available in the repository's eval
materials and are excluded from Chaperone's wheel and source distributions.
The MIT license on Chaperone's eval code does not replace the output notices.

## Installed tools and downloaded assets

Container recipes install separate upstream projects; their software and assets
retain their own terms. In particular, the public LigandMPNN backend uses the
upstream MIT-licensed source and public checkpoints. PyRosetta/Rosetta, model
parameters, and databases may have separate access or use requirements.

See the [container guide](https://github.com/RosettaCommons/chaperone/blob/main/docs/site/containers.md)
and each block's upstream links for the selected runtime. Build and GPU test
results describe installation and execution, not a change to upstream terms.
