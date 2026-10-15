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

## Installed tools and downloaded assets

Container recipes install separate upstream projects; their software and assets
retain their own terms. In particular, the `ligandmpnn` image installs the
MIT-licensed upstream LigandMPNN source and its published checkpoints. PyRosetta/Rosetta, model
parameters, and databases may have separate access or use requirements.

The `bindcraft2` recipe installs PacesaLab/BindCraft2 at revision
`c0ee638d11bc010817c2c97ee58ed5c323e664e2` under the
[BindCraft2 Source-Available License (Hosting-Restricted)](https://github.com/PacesaLab/BindCraft2/blob/c0ee638d11bc010817c2c97ee58ed5c323e664e2/LICENSE).
Its source tree and license are retained inside the image. This is separate from
Chaperone's MIT-licensed wrapper; AlphaFold2 and ProteinMPNN assets retain their
own upstream terms.

See the [container guide](https://github.com/RosettaCommons/chaperone/blob/main/docs/site/containers.md)
and each block's upstream links for the selected runtime. Build and GPU test
results describe installation and execution, not a change to upstream terms.

## Public structural example inputs

The antibody example's `8s6z_fv_cd28.pdb` is a cropped, consecutively numbered
protein-only derivative of PDB **8S6Z**, obtained from the public Foundry
binder-design tutorial. Its adjacent residue mapping preserves source author
labels and insertion codes. The RCSB PDB entry and associated publication should
be cited when using this structure scientifically.

Other ported example fixtures retain the inputs of the original
`baker-laboratory/pipelines` examples at snapshot `95ef0b6`. Workflow guides
identify their provenance and any changes. These fixtures are example inputs,
not newly generated validation results or evidence of experimental activity.

## Documentation images

`docs/site/.assets/icon.png` is unchanged from `baker-laboratory/pipelines`
commit `95ef0b6fd87eb27e1e73d2cfaa1482a39a0a132f`.

The three illustrations in `src/blocks/networks/foundry/.assets/` are unchanged
from [`RosettaCommons/foundry`, commit
`c97ad246877e2c7cb3b50e60aab7d76d02e8d393`](https://github.com/RosettaCommons/foundry/tree/c97ad246877e2c7cb3b50e60aab7d76d02e8d393/models/rfd3/docs/.assets):
`input_selection.png`, `input_selection_large.png`, and `partial_diff.png`.
They accompany the adapted RFD3 input reference. Copyright (c) 2025, Institute
for Protein Design, University of Washington; the upstream BSD 3-Clause license
is retained in `src/blocks/networks/foundry/.assets/LICENSE.md` and included
alongside the images in package distributions.
