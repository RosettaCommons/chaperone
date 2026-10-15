# Block Reference

Auto-generated parameter reference for all pipeline blocks.
Highlighted parameters are *primary* — they directly affect output count, quality, or runtime.
Computed defaults resolve from settings or model logic at runtime; no documentation builder's site configuration is embedded here.

Resources list what a block requests: `partition` is the class `cpu` or `gpu`, and `gpu_class` is the GPU size (`small` or `large`). Each site maps these to a real partition and `--gres` through its `CHAPERONE_*` settings; see [Slurm Site Configuration](syntax.md#slurm-site-configuration).

---

**Network Blocks**

Choose a predictor whose supported molecules, inputs, and validation evidence fit the task. Listing a block here does not establish scientific equivalence or a successful fresh-cluster installation. See [Core Examples](core-examples.md) for benchmark scope and [Validation](validation.md) for prediction comparisons.

- [esmfold2](#esmfold2) — ESMFold2 structure prediction with optional MSA conditioning
- [openfold3](#openfold3) — OpenFold 3 structure prediction
- [boltz2](#boltz2) — Boltz-2 structure prediction [[paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1)]
- [boltz2_steered](#boltz2_steered) — Boltz-2 with helix steering (Protein Hunter)
- [rf3](#rf3) — RosettaFold3 structure prediction [[paper](https://www.biorxiv.org/content/10.1101/2025.08.14.670328v1)]
- [protenix](#protenix) — Protenix structure prediction
- [chai1](#chai1) — Chai-1 structure prediction
- [af3](#af3) — AlphaFold3 structure prediction [[paper](https://doi.org/10.1038/s41586-024-07487-w)]
- [af3_with_target_template](#af3_with_target_template) — AlphaFold3 with the target chain structure as a fixed template
- [colabfold](#colabfold) — ColabFold (AlphaFold2) structure prediction
- [bindcraft](#bindcraft) — AlphaFold2-guided de novo protein binder design [[paper](https://www.biorxiv.org/content/10.1101/2024.09.30.615802)]
- [boltzgen](#boltzgen) — BoltzGen protein design block [[paper](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1)]
- [disco](#disco) — Multimodal protein sequence-structure co-design
- [e1](#e1) — E1 masked language model for sequence prediction [[paper](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1)]
- [esm2](#esm2) — ESM-2 protein language model for sequence embeddings [[paper](https://doi.org/10.1126/science.ade2574)]
- [foundry_mpnn](#foundry_mpnn) — Foundry MPNN for inverse folding [[paper](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1)]
- [manifest_val_structures](#manifest_val_structures) — Manifest reference PDB assemblies for prediction validation
- [mpnn](#mpnn) — Inverse folding with LigandMPNN (protein, ligand, soluble and membrane models) [[paper](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1)]
- [protein_hunter](#protein_hunter) — De novo binder design via Boltz structure hallucination [[paper](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v1)]
- [proteina_complexa](#proteina_complexa) — All-atom protein binder sequence/structure co-design [[paper](https://arxiv.org/abs/2603.27950)]
- [rfd1](#rfd1) — RFdiffusion v1 backbone-level protein design
- [rfd2](#rfd2) — RFdiffusion2 (rf_diffusion) block for atomic-level protein design
- [rfd3](#rfd3) — RFdiffusion3 block [[paper](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)]
- [rfd3_partial](#rfd3_partial) — RFdiffusion3 partial diffusion
- [rfd3na](#rfd3na) — RFdiffusion3NA block (all-atom protein/DNA/RNA design) [[paper](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2)]
- [rfoptimization](#rfoptimization) — RF3 gradient / Boltz-2 cycling with external MPNN sequence redesign [[paper](https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2)]
- [rpxdock](#rpxdock) — Dock a z-oriented C3 asymmetric unit into an I3 cage and emit C3 trimers
- [score_val_structures](#score_val_structures) — Score predictions against manifested validation references
- [simplefold](#simplefold) — Apple SimpleFold sequence-only protein monomer prediction
- [tmol_interface](#tmol_interface) — tmol protein-ligand interface ddG + per-term breakdown
- [tmol_minimize](#tmol_minimize) — tmol Cartesian all-atom energy minimisation (beta_nov2016_cart)

**Metric Blocks**

- [aromatic_neighbors](#aromatic_neighbors) — Count aromatic residues near ligand atoms
- [chirality_check](#chirality_check) — Per-atom CIP-code chirality comparison vs a reference ligand
- [clash](#clash) — Count non-bonded backbone heavy-atom clashes
- [contact_ms](#contact_ms) — Compute contact molecular surface between binder and target chains
- [dssr_motif_base_pairing](#dssr_motif_base_pairing) — Mapped motif base pairing and native DSSR classification retention.
- [dssr_snap_metrics](#dssr_snap_metrics) — Per-design biophysical metrics for protein-DNA complexes via x3dna-dssr and x3dna-snap (3DNA v2.3).
- [foldseek](#foldseek) — Cluster structures by structural similarity using Foldseek [[paper](https://doi.org/10.1038/s41587-023-01773-0)]
- [hbond_metrics](#hbond_metrics) — Compute hydrogen bond metrics for ligand interactions
- [ipsae](#ipsae) — Interface predicted Score from Aligned Errors
- [ligand_clash](#ligand_clash) — Measure the closest protein-ligand heavy-atom contact
- [ligand_ddg](#ligand_ddg) — Compute Rosetta DDG binding energy
- [ligand_ddg_soft](#ligand_ddg_soft) — Soft-repulsive ddG (pack hard, score soft, trimmed mean)
- [ligand_pocket](#ligand_pocket) — Distributed ligand contact, enclosure, and shell metrics
- [metal3d](#metal3d) — Predict transition-metal binding sites with the Metal3D 3D-CNN
- [motif_confidence](#motif_confidence) — Motif/scaffold pLDDT and PAE split, via diffused_index_map
- [motif_geometry](#motif_geometry) — Distances/angles between motif atoms, via diffused_index_map
- [mpnn_score](#mpnn_score) — Parse MPNN FASTA confidences as ranking metrics
- [peppr](#peppr) — Compute structural evaluation metrics using peppr
- [plip](#plip) — Profile protein-ligand interactions using PLIP
- [pocket_coulomb](#pocket_coulomb) — Electrostatic complementarity between charged ligand atoms and charged pocket residues
- [ppi_ddg](#ppi_ddg) — Protein-protein binding energy with FastRelax + ddG
- [rasa](#rasa) — Compute solvent accessible surface area for ligand atoms
- [reference_motif_rmsd](#reference_motif_rmsd) — Motif RMSD against the input reference, via diffused_index_map
- [rmsd](#rmsd) — Compute alignment RMSD metrics between two designs
- [rna_connectivity](#rna_connectivity) — O3'(i)->P(i+1) backbone breaks in nucleic-acid chains
- [rna_ss_f1_mcc](#rna_ss_f1_mcc) — Motif-restricted secondary-structure F1 and MCC of an RNet2 prediction against the reference's canonical pairs
- [rna_thermo](#rna_thermo) — ViennaRNA Boltzmann probability, pair probabilities and ensemble defect for a required motif pair set
- [rotamer_metrics](#rotamer_metrics) — Compute rotamer quality metrics for protein sidechains
- [secondary_structure](#secondary_structure) — Per-fold SS composition + loop/pLDDT disorder, globally and pocket-around-ligand
- [sequence_identity_gate](#sequence_identity_gate) — Global NW identity of the scaffold region against natural references; hard novelty gate
- [symmetry](#symmetry) — Cyclic point-group deviation of a homo-oligomeric assembly
- [tm_score](#tm_score) — TM-score of each structure against an external reference fold
- [tm_score_rna](#tm_score_rna) — TM-score_RNA (Gong 2019) against a reference fold via US-align, normalized by the reference length

**Utility Blocks**

- [annotate](#annotate) — Annotate molecular structures
- [filter](#filter) — Filter pipeline rows by metrics predicate
- [ingest](#ingest) — Ingest data from various sources
- [materialize](#materialize) — Write stored structures to disk
- [mutate_residues](#mutate_residues) — Create named protein sequence variants with explicit lineage
- [nanoparticle_prepare](#nanoparticle_prepare) — Align antigen clash reference and preserve a C3 scaffold's symmetry axis
- [order_collection](#order_collection) — Extract sequences, check ordered status, collect metrics
- [package_submission](#package_submission) — Validate and write a sequence-only design submission
- [remap_ligand](#remap_ligand) — Rename predicted ligand atom/residue names to match a reference CIF
- [restore_sidechains](#restore_sidechains) — Restore selected motif rotamers from an upstream structure
- [rotamer_sweep](#rotamer_sweep) — Enumerate explicit functional side-chain chi combinations
- [select_chains](#select_chains) — Keep specified chains, preserving residue IDs and parent lineage
- [select_diverse](#select_diverse) — Select a diverse design subset under sequence/structure cutoffs
- [select_motif](#select_motif) — Apply a function to select motif residues for downstream fixing
- [selective_sidechain_repack](#selective_sidechain_repack) — Repack only explicitly selected protein sidechains
- [to_cif](#to_cif) — Parse structures via get_data and re-save as CIF
- [to_pdb](#to_pdb) — Parse structures via get_data and re-save as PDB
- [transform](#transform) — Apply an atomworks Transform to each pipeline row

---

## Network Blocks

*Blocks that run neural network inference — structure prediction, design, sequence models.*

### esmfold2

*ESMFold2 structure prediction with optional MSA conditioning.*

> [Upstream code / documentation](https://github.com/Biohub/esm) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/esmfold2.py)

Class: `ESMFold2Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`48GB`, time=`5:00:00` · Container: `esmfold2`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_loops`</span> | `int` | `3` | Trunk refinement loops (API default 3; reference driver used 20). |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `seed` | `Optional[int]` | `0` | Random seed for input prep + diffusion sampling. |
| `model` | `str` | `'biohub/ESMFold2'` | HuggingFace model id (or local snapshot path) for ESMFold2. |
| `use_local_msas` | `bool` | `False` | Condition each protein chain on its precomputed A3M from the on-disk MSA stores. False runs ESMFold2 in single-sequen... |
| `msa_dirs` | `Optional[list]` | `None` | Sharded MSA roots searched in order, as paths or {dir, extension, directory_depth} dicts. None uses PROTEIN_MSA_DIRS. |
| `max_msa_sequences` | `Optional[int]` | `16384` | Maximum A3M depth supplied to ESMFold2's MSA encoder. |
| `raise_if_missing_msa_for_protein_of_length_n` | `Optional[int]` | `None` | Fail when a protein chain at least this many residues long has no on-disk MSA instead of silently using single-sequen... |

---

### openfold3

*OpenFold 3 biomolecular structure prediction.*

> [Upstream code / documentation](https://github.com/aqlaboratory/openfold-3) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/openfold3.py)

Class: `OpenFold3Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`64GB`, time=`5:00:00` · Container: `openfold3`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_model_seeds`</span> | `int` | `1` | Number of model seeds to use for each query. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`use_msa_server`</span> | `bool` | `False` | Use ColabFold MSA server for alignments. False for single-sequence prediction. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `use_templates` | `bool` | `False` | Use ColabFold MSA server for template alignments. |
| `template_json` | `Optional[str]` | `None` | Path to template JSON file for OpenFold3 input. If provided, sequences from upstream designs are filled into this tem... |
| `cache_dir` | `str` | `<computed at runtime>` | Directory containing OpenFold3 model weights. |
| `runner_yaml` | `Optional[str]` | `None` | Path to a runner YAML for advanced model/dataset configuration. |

---

### boltz2

*Boltz-2 biomolecular structure prediction.*

> Passaro, Corso, Wohlwend et al. (2025) — Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction
> [Paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1) · [Upstream code / documentation](https://github.com/jwohlwend/boltz) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/boltz2.py)

Class: `Boltz2Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `boltz2`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`override`</span> | `bool` | `True` | Override existing predictions if found. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`step_scale`</span> | `float` | `1.638` | Diffusion step scale. Lower = more diversity (recommended 1-2). |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `max_parallel_samples` | `int` | `1` | Max samples to predict in parallel. |
| `output_format` | `str` | `'mmcif'` | Output format: 'mmcif' or 'pdb'. |
| `use_msa_server` | `bool` | `False` | Auto-generate MSAs via mmseqs2 server. |
| `msa_server_url` | `Optional[str]` | `None` | MSA server URL (default: ColabFold). |
| `use_potentials` | `bool` | `False` | Use inference-time potentials for better physical plausibility. |
| `predict_affinity` | `bool` | `False` | Enable affinity prediction for protein-ligand complexes. |
| `cache_dir` | `str` | `<computed at runtime>` | Directory containing Boltz model weights and CCD data. |
| `template_yaml` | `Optional[str]` | `None` | Path to a template YAML file. If provided, sequences from upstream designs are filled into this template. |
| `no_kernels` | `bool` | `False` | Disable cuEquivariance kernels (for older GPUs). |

---

### boltz2_steered

*Boltz-2 prediction with helix-steering from Protein Hunter.*

> [Upstream code / documentation](https://github.com/jwohlwend/boltz) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/hallucination/boltz2_steered.py)

Class: `Boltz2SteeredBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `protein_hunter`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`override`</span> | `bool` | `True` | Override existing predictions if found. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`step_scale`</span> | `float` | `1.638` | Diffusion step scale. Lower = more diversity (recommended 1-2). |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`randomly_kill_helix_feature`</span> | `bool` | `False` | Scale pair embeddings at (i, i+4) positions by a random negative factor on the binder chain, disrupting helix formation. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`negative_helix_constant`</span> | `float` | `0.2` | Magnitude of the negative scaling factor for helix disruption. Higher = stronger suppression. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`binder_chain`</span> | `str` | `'A'` | Chain ID of the binder to apply helix steering to. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `max_parallel_samples` | `int` | `1` | Max samples to predict in parallel. |
| `output_format` | `str` | `'mmcif'` | Output format: 'mmcif' or 'pdb'. |
| `use_msa_server` | `bool` | `False` | Auto-generate MSAs via mmseqs2 server. |
| `msa_server_url` | `Optional[str]` | `None` | MSA server URL (default: ColabFold). |
| `use_potentials` | `bool` | `False` | Use inference-time potentials for better physical plausibility. |
| `predict_affinity` | `bool` | `False` | Enable affinity prediction for protein-ligand complexes. |
| `cache_dir` | `str` | `<computed at runtime>` | Directory containing Boltz model weights and CCD data. |
| `template_yaml` | `Optional[str]` | `None` | Path to a template YAML file. If provided, sequences from upstream designs are filled into this template. |
| `no_kernels` | `bool` | `False` | Disable cuEquivariance kernels (for older GPUs). |

---

### rf3

*Vanilla RosettaFold3 block for structure prediction (container-based).*

> Krishna et al. (2025) — Accelerating Biomolecular Modeling with AtomWorks and RF3
> [Paper](https://www.biorxiv.org/content/10.1101/2025.08.14.670328v1) · [Upstream code / documentation](https://github.com/RosettaCommons/foundry) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/rosettafold3.py)

Class: `RosettaFold3Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `foundry`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `10` | — |
| `num_diffusion_steps` | `int` | `50` | — |
| `ckpt_path` | `Optional[str]` | `<computed at runtime>` | Checkpoint path or Foundry registry name (default: rf3). |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `early_stopping_plddt_threshold` | `float` | `0.5` | Stop diffusion early if pLDDT falls below this threshold after the initial recycle. Set to 0.5 for 10-20x throughput ... |
| `dump_trajectories` | `bool` | `False` | Write the noisy and denoised diffusion trajectories for each sample. Trajectory files are large, place the final pred... |
| `seed_offset` | `int` | `42` | Base random seed; each chunk increments by 1. Ignored when model_seeds is set. |
| `model_seeds` | `Optional[list[int]]` | `None` | List of random seeds for multi-seed inference per input. Each seed runs a completely independent trunk pass, giving m... |
| `msa_dirs` | `Optional[list[str]]` | `None` | Absolute on-disk protein MSA roots. Chaperone exports these to both LOCAL_MSA_DIRS and PROTEIN_MSA_DIRS (supported Fo... |
| `raise_if_missing_msa_for_protein_of_length_n` | `Optional[int]` | `None` | Fail when a protein at least this many residues long has no MSA. Set to 0 to require an MSA for every protein chain. |
| `create_bond_graph` | `bool` | `False` | Assign ligand bonds from specified_smiles and write inputs as CIF with bond topology. Required when input structures ... |
| `input_mode` | `Literal` | `'file'` | 'file' passes CIF/PDB directly (current default). 'sequence' builds RF3 component JSON from sequences + CCD/SMILES. |
| `template_json` | `Optional[str]` | `None` | Path to RF3 JSON template. Protein sequences are filled from upstream designs; everything else (ligands, bonds, templ... |
| `template_selection` | `Optional[list[str]]` | `None` | Token-level templates for polymers and small molecules. Uses AtomSelection format, e.g. ['A/*/30', 'A/*/53', 'X']. |
| `ground_truth_conformer_selection` | `Optional[list[str]]` | `None` | Lock selected residues/chains to ground truth conformers. Useful for keeping known ligand conformations while allowin... |
| `cyclic_chains` | `Optional[list[str]]` | `None` | Chain IDs that should be cyclized. If input is CIF, cyclization is auto-detected. |

---

### protenix

*Protenix biomolecular structure prediction (v2).*

> [Upstream code / documentation](https://github.com/bytedance/Protenix) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/protenix.py)

Class: `ProtenixBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `protenix`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model_name`</span> | `str` | `'protenix_base_20250630_v1.0.0'` | Model checkpoint name. Options: protenix-v2 (464M, best accuracy, TFG support), protenix_base_20250630_v1.0.0 (368M, ... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`seeds`</span> | `str` | `'101'` | Comma-separated random seeds for independent runs. Each seed runs a completely independent prediction. Example: '101,... |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `dtype` | `str` | `'bf16'` | Precision: 'bf16' (faster, default) or 'fp32' (more precise). |
| `use_msa` | `bool` | `False` | Enable multiple sequence alignment features. Requires HMMER databases accessible via /databases bind mount. Default F... |
| `use_template` | `bool` | `False` | Use template structures for guided prediction. |
| `use_tfg_guidance` | `bool` | `False` | Enable Training-Free Guidance (TFG) during diffusion sampling. TFG enforces geometric constraints (steric, torsion, b... |
| `need_atom_confidence` | `bool` | `False` | Output per-atom confidence scores (generates full_data JSON). |
| `enable_cache` | `bool` | `True` | Cache shareable diffusion variables for faster inference. |
| `enable_fusion` | `bool` | `True` | Enable kernel fusion optimizations. |
| `enable_tf32` | `bool` | `True` | Enable TF32 for FP32 matrix multiplications (faster on Ampere+). |
| `trimul_kernel` | `str` | `'cuequivariance'` | Triangle multiplication kernel: 'cuequivariance' or 'torch'. |
| `triatt_kernel` | `str` | `'cuequivariance'` | Triangle attention kernel: 'cuequivariance', 'deepspeed', 'triattention', or 'torch'. |
| `cache_dir` | `str` | `<computed at runtime>` | Directory for Protenix model weights (auto-downloaded on first run). |
| `template_json` | `Optional[str]` | `None` | Path to a template Protenix JSON file. If provided, protein sequences from upstream designs are filled into this temp... |

---

### chai1

*Chai-1 biomolecular structure prediction.*

> [Upstream code / documentation](https://github.com/chaidiscovery/chai-lab) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/chai1.py)

Class: `Chai1Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `chai1`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `seed` | `Optional[int]` | `None` | Random seed for reproducibility. None = random. |
| `use_esm_embeddings` | `bool` | `True` | Use ESM embeddings for proteins. |
| `use_msa_server` | `bool` | `False` | Auto-generate MSAs via ColabFold mmseqs2 server. |
| `msa_server_url` | `Optional[str]` | `None` | MSA server URL (default: ColabFold). |
| `low_memory` | `bool` | `True` | Use memory-efficient inference mode. |
| `cache_dir` | `Optional[str]` | `<computed at runtime>` | Directory for Chai-1 model weights. Defaults to CHAI_DOWNLOADS_DIR setting. |

---

### af3

*AlphaFold3 structure prediction.*

> Abramson et al. (2024) — Accurate structure prediction of biomolecular interactions with AlphaFold 3
> [Paper](https://doi.org/10.1038/s41586-024-07487-w) · [Upstream code / documentation](https://github.com/google-deepmind/alphafold3) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/alphafold3.py)

Class: `AlphaFold3Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `alphafold3`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `jax_cache_dir` | `Optional[str]` | `<computed at runtime>` | Writable compilation cache mounted directly into the container |
| `input_mode` | `Literal` | `'mmcif'` | AF3 input preparation mode. The default 'mmcif' serializes the complete AtomArray and uses AF3's native Input.from_mm... |
| `use_local_msas` | `bool` | `False` | Inject each protein chain's precomputed a3m from the on-disk MSA stores into the AF3 input. Works with both native mm... |
| `msa_dirs` | `Optional[list]` | `None` | Sharded MSA roots searched in order, as paths or as {"dir", "extension", "directory_depth"} dicts. None uses PROTEIN_... |
| `max_msa_sequences` | `Optional[int]` | `10000` | Truncate injected MSAs to this depth. |
| `raise_if_missing_msa_for_protein_of_length_n` | `Optional[int]` | `None` | Fail when a protein at least this many residues long has no on-disk MSA, instead of silently falling back to single-s... |
| `template_json` | `Optional[str]` | `None` | Path to template JSON file for AF3 input. If not provided, create input from scratch. |
| `add_ptm_at_mapped_residues` | `Optional[dict]` | `None` | Map of {ptmType: original_residue_key} for covalent modifications. The residue key (e.g. 'A10') is looked up in the R... |
| `add_ptm_at_residues` | `Optional[list[dict]]` | `None` | Direct list of AF3 modification dicts to add to the protein sequence. Each entry should have 'ptmType' and 'ptmPositi... |
| `consolidate_intermediates` | `bool` | `True` | After compiling, pack each design's redundant per-sample subdirs (outputs/<design>/seed-*_sample-*/) into a single un... |
| `buckets` | `Optional[list[int]]` | `None` | Token-count buckets for JIT compilation. None uses the inner script's smart default (fine-grained at small token coun... |
| `flash_attention_implementation` | `str` | `'triton'` | 'triton' (fastest, Ampere+), 'cudnn', or 'xla' (portable). |
| `conformer_max_iterations` | `Optional[int]` | `None` | RDKit conformer search max iterations. None uses RDKit default. |
| `mean_pae_contact_distance` | `float` | `5.0` | Contact cutoff (Å) for the `interface_mean_pae` metric: only protein residues with any atom within this distance of t... |
| `mean_pae_core_atoms` | `Optional[list[str]]` | `None` | Ligand atom names defining the buried/core substructure for the `interface_mean_pae` and `full_mean_pae` metrics. Whe... |
| `mean_pae_sidechain_angle` | `Optional[float]` | `90.0` | Max angle (degrees) between a pocket residue's CA→sidechain-centroid vector and its CA→nearest-ligand-atom vector for... |
| `model_seeds` | `list[int]` | `<computed at runtime>` | Independent AF3 model seeds written into each input JSON. Multiple seeds rerun trunk and diffusion inference and are ... |

---

### af3_with_target_template

*AF3 with the design's target chain attached as a structural template.*

> [Upstream code / documentation](https://github.com/google-deepmind/alphafold3) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/af3_with_target_template.py)

Class: `AF3WithTargetTemplate` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `alphafold3`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `jax_cache_dir` | `Optional[str]` | `<computed at runtime>` | Writable compilation cache mounted directly into the container |
| `input_mode` | `Literal` | `'json'` | — |
| `use_local_msas` | `bool` | `False` | Inject each protein chain's precomputed a3m from the on-disk MSA stores into the AF3 input. Works with both native mm... |
| `msa_dirs` | `Optional[list]` | `None` | Sharded MSA roots searched in order, as paths or as {"dir", "extension", "directory_depth"} dicts. None uses PROTEIN_... |
| `max_msa_sequences` | `Optional[int]` | `10000` | Truncate injected MSAs to this depth. |
| `raise_if_missing_msa_for_protein_of_length_n` | `Optional[int]` | `None` | Fail when a protein at least this many residues long has no on-disk MSA, instead of silently falling back to single-s... |
| `template_json` | `Optional[str]` | `None` | Path to template JSON file for AF3 input. If not provided, create input from scratch. |
| `add_ptm_at_mapped_residues` | `Optional[dict]` | `None` | Map of {ptmType: original_residue_key} for covalent modifications. The residue key (e.g. 'A10') is looked up in the R... |
| `add_ptm_at_residues` | `Optional[list[dict]]` | `None` | Direct list of AF3 modification dicts to add to the protein sequence. Each entry should have 'ptmType' and 'ptmPositi... |
| `consolidate_intermediates` | `bool` | `True` | After compiling, pack each design's redundant per-sample subdirs (outputs/<design>/seed-*_sample-*/) into a single un... |
| `buckets` | `Optional[list[int]]` | `None` | Token-count buckets for JIT compilation. None uses the inner script's smart default (fine-grained at small token coun... |
| `flash_attention_implementation` | `str` | `'triton'` | 'triton' (fastest, Ampere+), 'cudnn', or 'xla' (portable). |
| `conformer_max_iterations` | `Optional[int]` | `None` | RDKit conformer search max iterations. None uses RDKit default. |
| `mean_pae_contact_distance` | `float` | `5.0` | Contact cutoff (Å) for the `interface_mean_pae` metric: only protein residues with any atom within this distance of t... |
| `mean_pae_core_atoms` | `Optional[list[str]]` | `None` | Ligand atom names defining the buried/core substructure for the `interface_mean_pae` and `full_mean_pae` metrics. Whe... |
| `mean_pae_sidechain_angle` | `Optional[float]` | `90.0` | Max angle (degrees) between a pocket residue's CA→sidechain-centroid vector and its CA→nearest-ligand-atom vector for... |
| `model_seeds` | `list[int]` | `<computed at runtime>` | Independent AF3 model seeds written into each input JSON. Multiple seeds rerun trunk and diffusion inference and are ... |
| `target_chain` | `str` | `'B'` | Chain ID of the receptor / target whose structure should be templated. |
| `template_kalign_matches` | `Optional[int]` | `None` | If set, override AF3's `templates[].kalign_matches`. Default lets AF3 decide. |
| `full_receptor_pdb` | `Optional[str]` | `None` | Optional path to a PDB/CIF containing the full target chain. When set, AF3's chain `target_chain` is replaced with th... |

---

### colabfold

*ColabFold (AlphaFold2) monomer and multimer structure prediction.*

> [Upstream code / documentation](https://github.com/sokrypton/ColabFold) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/colabfold.py)

Class: `ColabFoldBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `colabfold_portable`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `5` | Number of diffusion samples (prediction structures) per input. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `num_diffusion_steps` | `int` | `200` | Number of diffusion sampling steps. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `num_models` | `int` | `5` | Number of AF2 models (1-5). |
| `num_seeds` | `int` | `1` | Number of random seeds per model. |
| `random_seed` | `Optional[int]` | `None` | Base random seed. None = ColabFold default (0). |
| `msa_mode` | `str` | `'single_sequence'` | MSA databases: single_sequence (no MSA, offline), mmseqs2_uniref_env, mmseqs2_uniref, mmseqs2_uniref_env_envpair. |
| `use_msa_server` | `bool` | `False` | Query the ColabFold MMseqs2 server for MSAs. When True and msa_mode is still single_sequence, switches to mmseqs2_uni... |
| `use_local_msas` | `bool` | `False` | Look each folded sequence up in the precomputed on-disk MSA stores and stage it as an a3m input, which overrides msa_... |
| `msa_dirs` | `Optional[list]` | `None` | Sharded MSA roots searched in order, as paths or as {"dir", "extension", "directory_depth"} dicts. None uses the site... |
| `max_msa_sequences` | `Optional[int]` | `10000` | Truncate staged MSAs to this depth. None keeps the full store entry. |
| `raise_if_missing_msa_for_protein_of_length_n` | `Optional[int]` | `None` | Fail when a protein at least this many residues long has no on-disk MSA. None (default) uses a query-only alignment. ... |
| `model_type` | `str` | `'auto'` | AF2 model set (auto, alphafold2_ptm, alphafold2_multimer_v3, ...). |
| `use_templates` | `bool` | `False` | Query PDB templates from the MSA server (needs internet). |
| `amber_relax` | `bool` | `False` | Run AMBER relaxation (CPU only; --use-gpu-relax is never emitted). |
| `rank` | `str` | `'auto'` | Model ranking metric (auto, plddt, ptm, iptm, multimer). |
| `data_dir` | `Optional[str]` | `<computed at runtime>` | Writable dir for AF2 params (--data). ~3-4GB downloaded on first run. Bound into the container automatically when the... |

---

### bindcraft

*Design protein binders against each input target structure.*

> Pacesa et al. (2024) — BindCraft: one-shot design of functional protein binders
> [Paper](https://www.biorxiv.org/content/10.1101/2024.09.30.615802) · [Upstream code / documentation](https://github.com/martinpacesa/BindCraft) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/bindcraft.py)

Class: `BindCraftBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`large`, mem=`48GB`, time=`24:00:00` · Container: `bindcraft`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`chains`</span> | `str` | **required** | BindCraft target-chain selector, e.g. 'A' or 'A,B'. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`target_hotspot_residues`</span> | `Optional[str]` | `None` | Hotspot selection such as 'A56,A60-64'; None auto-selects a site. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`lengths`</span> | `tuple` | `(65, 150)` | Inclusive minimum and maximum binder length. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`number_of_final_designs`</span> | `int` | `1` | Accepted designs requested per input target. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `max_trajectories` | `Optional[int]` | `100` | Maximum hallucination trajectories per target. None honors the advanced preset and may run until the accepted-design ... |
| `binder_name` | `str` | `'binder'` | Prefix for generated designs. |
| `filter_preset` | `str` | `'default_filters'` | Bundled BindCraft filter preset basename. |
| `advanced_preset` | `str` | `'default_4stage_multimer'` | Bundled BindCraft advanced-settings preset basename. |
| `filters_path` | `Optional[str]` | `None` | Optional custom filters JSON; overrides filter_preset. |
| `advanced_path` | `Optional[str]` | `None` | Optional custom advanced JSON; overrides advanced_preset. |
| `filter_overrides` | `dict` | `<computed at runtime>` | Shallow overrides applied to the selected filter JSON. |
| `advanced_overrides` | `dict` | `<computed at runtime>` | Shallow overrides applied to the selected advanced JSON. |

---

### boltzgen

*BoltzGen block for protein design.*

> Stark et al. (2025) — BoltzGen: Toward Universal Binder Design
> [Paper](https://www.biorxiv.org/content/10.1101/2025.11.20.689494v1) · [Upstream code / documentation](https://github.com/baker-laboratory/BoltzGen) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/boltzgen.py)

Class: `BoltzGenBlock` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `boltzgen`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`config`</span> | `str` | **required** | Path to BoltzGen design specification YAML file |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`protocol`</span> | `str` | `'protein-anything'` | Design protocol (protein-anything, peptide-anything, protein-small_molecule, nanobody-anything, antibody-anything, pr... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_designs`</span> | `int` | `100` | Total number of designs to generate |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`budget`</span> | `int` | `30` | Number of final designs after filtering |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `steps` | `Optional[list[str]]` | `None` | Pipeline steps to run (default: all). Options: design, inverse_folding, design_folding, folding, affinity, analysis, ... |
| `diffusion_batch_size` | `Optional[int]` | `None` | Designs per trunk run (default: auto) |
| `step_scale` | `Optional[float]` | `None` | Fixed diffusion step scale |
| `noise_scale` | `Optional[float]` | `None` | Fixed noise scale |
| `skip_inverse_folding` | `bool` | `False` | Skip the inverse folding step |
| `inverse_fold_num_sequences` | `int` | `1` | Number of sequences per backbone |
| `alpha` | `Optional[float]` | `None` | Quality vs diversity (0.0=quality, 1.0=diversity) |
| `devices` | `Optional[int]` | `None` | Number of GPUs (default: all available) |
| `reuse` | `bool` | `False` | Reuse existing results from previous runs |
| `n_batches` | `int` | `1` | Number of independent BoltzGen runs |

---

### disco

*DISCO: Diffusion for Sequence-Structure Co-design.*

> [Upstream code / documentation](https://github.com/DISCO-design/DISCO) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/disco/disco.py)

Class: `DISCOBlock` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `disco`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`inputs`</span> | `str` | **required** | Path to DISCO input JSON file (AF3 Server-like format) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_seeds`</span> | `int` | `5` | Number of random seeds (samples per job in the input JSON) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`experiment`</span> | `Literal` | `'designable'` | Experiment preset: 'designable' (higher designability) or 'diverse' (greater structural variety) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`effort`</span> | `Literal` | `'fast'` | Compute preset: 'fast' (~4x faster, default) or 'max' (full quality). Use 'max' for conditional generation. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_batches` | `int` | `1` | Number of parallel chunks to split seeds across |
| `use_deepspeed_evo_attention` | `bool` | `True` | Use DeepSpeed EvoformerAttention (requires CUTLASS + Ampere+ GPU) |
| `n_diffusion_steps` | `Optional[int]` | `None` | Override number of diffusion steps (default: set by effort preset) |
| `n_cycles` | `Optional[int]` | `None` | Override number of recycling cycles (default: set by effort preset) |
| `noisy_guidance` | `Optional[bool]` | `None` | Enable/disable noisy guidance (default: set by experiment preset) |
| `ckpt_path` | `Optional[str]` | `None` | Path to custom model checkpoint (default: auto-download from HuggingFace) |
| `cache_dir` | `str` | `<computed at runtime>` | Hugging Face model cache mounted into the container |

---

### e1

*E1 block for masked language modeling on protein sequences.*

> Jain et al. (2025) — E1: Retrieval-Augmented Protein Encoder Models
> [Paper](https://www.biorxiv.org/content/10.1101/2025.11.12.688125v1) · [Upstream code / documentation](https://github.com/Profluent-AI/E1) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/plm/e1.py)

Class: `E1Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`16GB`, time=`5:00:00` · Container: `e1`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model_name`</span> | `Literal \| str` | `'150m'` | Model size (150m, 300m, 600m) or full HuggingFace path |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`max_batch_tokens`</span> | `int` | `65536` | Maximum number of tokens to batch in a single forward pass |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `save_masked_positions_only` | `bool` | `False` | Whether to save logits/embeddings for only the masked positions |
| `fields_to_save` | `list[str]` | `['logits', 'token_embeddings', 'mean_token_embeddings']` | Fields to save in output safetensors |
| `context_fasta` | `Optional[str]` | `None` | Optional path to context FASTA for retrieval-augmented prediction |
| `cache_dir` | `str` | `<computed at runtime>` | Hugging Face model cache, mounted directly into the container |

---

### esm2

*ESM-2 block for protein sequence embeddings.*

> Lin et al. (2023) — Evolutionary-scale prediction of atomic-level protein structure with a language model
> [Paper](https://doi.org/10.1126/science.ade2574) · [Upstream code / documentation](https://github.com/facebookresearch/esm) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/plm/esm2.py)

Class: `ESM2Block` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`16GB`, time=`5:00:00` · Container: `esm2`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model_name`</span> | `Literal \| str` | `'650m'` | Model size (8m, 35m, 150m, 650m, 3b) or full HuggingFace path |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`max_batch_tokens`</span> | `int` | `4096` | Maximum number of tokens to batch in a single forward pass |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `fields_to_save` | `list[str]` | `['token_embeddings', 'mean_token_embeddings']` | Fields to save in output safetensors |

---

### foundry_mpnn

*Foundry Protein/LigandMPNN for inverse folding.*

> Foundry Protein/LigandMPNN implementation; paper describes the underlying LigandMPNN method
> [Paper](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1) · [Upstream code / documentation](https://github.com/RosettaCommons/foundry) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/mpnn.py)

**Validation status:** Wrapper/integration coverage does not establish scientific equivalence to the `mpnn` block (LigandMPNN). The [PPI benchmark scripts](https://github.com/RosettaCommons/chaperone/tree/main/examples/benchmarks/ppi) provide a starting point for a controlled comparison; no equivalence result is asserted here.

Class: `FoundryMPNNBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00` · Container: `foundry`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model_type`</span> | `Literal` | `'ligand_mpnn'` | Model type: protein_mpnn or ligand_mpnn. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_seq_per_pdb`</span> | `int` | `8` | Number of sequences to generate per structure. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`output_mode`</span> | `Literal` | `'sequences'` | Return compact sequence rows or legacy generated CIF structures. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`temperature`</span> | `float` | `0.1` | Sampling temperature. Lower = more conservative. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `checkpoint_path` | `Optional[str]` | `None` | Path to model checkpoint. None uses foundry default. |
| `seed` | `Optional[int]` | `None` | Random seed for reproducibility. |
| `fixed_residues` | `Optional[list[str]]` | `None` | Residues to keep fixed, e.g. ['A12', 'A13']. |
| `designed_residues` | `Optional[list[str]]` | `None` | Residues to redesign (all others fixed). |
| `fixed_chains` | `Optional[list[str]]` | `None` | Chain IDs to keep fixed. |
| `designed_chains` | `Optional[list[str]]` | `None` | Chain IDs to redesign. |
| `keep_motif_sequences_fixed` | `bool` | `True` | Keep sequences of motifs fixed (from RFD3 metadata). |

---

### manifest_val_structures

*Materialize the parquet's PDB assemblies and carry its scoring schema.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/validation/manifest.py)

Class: `ManifestValStructuresBlock` · Input: `False` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `parquet_path` | `str` | **required** | Entry-level validation table (.parquet or .csv). |
| `pdb_mirror` | `str` | `<computed at runtime>` | Local PDB mirror root, sharded by the middle two PDB-ID characters. |
| `bucket` | `str` | `'all'` | Validation composition bucket to select. 'all' keeps every row; individual buckets are: protein, protein-protein, pro... |

---

### mpnn

*Inverse folding with LigandMPNN.*

> Dauparas et al. (2023) — Atomic context-conditioned protein sequence design using LigandMPNN
> [Paper](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1) · [Upstream code / documentation](https://github.com/dauparas/LigandMPNN) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/mpnn.py)

**Backend:** `mpnn` runs LigandMPNN (`dauparas/LigandMPNN`) from the `ligandmpnn` image, with every official checkpoint baked in; install it with `chaperone install ligandmpnn`. No settings are needed. `model_type` selects the model (`protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`, `per_residue_label_membrane_mpnn` or `global_label_membrane_mpnn`) and `noise_level` selects the checkpoint version; the [container guide](containers.md) lists the valid combinations.

**Packed structures:** Use `output_mode="structures"` and, for one structure per sampled sequence, `args={"number_of_packs_per_design": 1}`. The worked examples set these explicitly. The block default remains compact sequence output. With packed output and `args.repack_everything=0`, Chaperone restores observed coordinates of fixed protein residues after packing: the upstream packer can reconstruct and move these atoms even when their sequences are fixed. Newly completed atoms remain packed. Set `args.repack_everything=1` to allow fixed residues to be repacked, or `preserve_fixed_coordinates=False` to retain native packer reconstructions. Sequence fixing and coordinate preservation are distinct from subsequent refolding, which can move the structure.

Class: `ClassicMPNNBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00` · Container: `ligandmpnn`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model_type`</span> | `Literal` | `'ligand_mpnn'` | Model family: protein_mpnn, ligand_mpnn, soluble_mpnn, or a membrane model (per_residue_label_membrane_mpnn, global_l... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`noise_level`</span> | `Optional[Literal]` | `None` | Checkpoint version: backbone noise (Angstrom) the model was trained with. None uses the upstream default. Valid value... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_seq_per_pdb`</span> | `int` | `8` | Number of sequences to generate per structure |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`output_mode`</span> | `Literal` | `'sequences'` | Return compact sequence rows (default) or side-chain-packed structures. Sequence mode disables packing. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `preserve_fixed_coordinates` | `bool` | `True` | For packed structures with repack_everything=0, restore observed finite heavy-atom coordinates of fixed protein resid... |
| `skip_existing` | `bool` | `True` | Skip processing if outputs already exist |
| `device` | `Optional[str]` | `None` | Force device for MPNN inference (e.g. 'cpu'). Default: auto-detect. |
| `keep_motif_sequences_fixed` | `bool` | `True` | Keep sequences of motifs fixed (if read into the dataframe) |
| `fix_interface_residues` | `bool` | `False` | Fix the sequence at the interfaces |
| `hbond_fix` | `bool` | `False` | Fix residues involved in hydrogen bonds |
| `aromatic_fix` | `bool` | `False` | Fix aromatic residues near ligand |
| `component_remap` | `Optional[dict]` | `None` | Remap residue names in CIF files before MPNN. Maps old -> new (e.g., {'LIG_B': 'B;N'}). |
| `args` | `MPNNArgs` | `<computed at runtime>` | LigandMPNN/ProteinMPNN CLI arguments. Partial overrides keep defaults. |

---

### protein_hunter

*Protein Hunter: de novo protein binder design via structure hallucination with Boltz.*

> Cho et al. (2025) — Protein Hunter: exploiting structure hallucination within diffusion for protein design
> [Paper](https://www.biorxiv.org/content/10.1101/2025.10.10.681530v1) · [Upstream code / documentation](https://github.com/yehlincho/Protein-Hunter) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/hallucination/protein_hunter.py)

Class: `ProteinHunterBlock` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `protein_hunter`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`protein_seqs`</span> | `str` | `''` | Target protein sequence(s). Separate multiple chains with ':'. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_designs`</span> | `int` | `10` | Number of independent design trajectories. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_cycles`</span> | `int` | `5` | Number of predict-redesign cycles per trajectory. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`min_protein_length`</span> | `int` | `100` | Minimum binder length (residues). |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`max_protein_length`</span> | `int` | `150` | Maximum binder length (residues). |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffuse_steps`</span> | `int` | `200` | Number of diffusion steps. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`recycling_steps`</span> | `int` | `3` | Number of recycling steps. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`temperature`</span> | `float` | `0.1` | LigandMPNN sampling temperature. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_ccd` | `str` | `''` | CCD code for small molecule target (e.g., 'SAM'). |
| `ligand_smiles` | `str` | `''` | SMILES string for small molecule target. |
| `nucleic_seq` | `str` | `''` | DNA/RNA target sequence. |
| `nucleic_type` | `str` | `'dna'` | Nucleic acid type: 'dna' or 'rna'. |
| `emit_all_cycles` | `bool` | `False` | Emit one handler row per predicted cycle instead of one multi-structure lineage row per trajectory. Use this when eve... |
| `percent_X` | `int` | `90` | Percentage of X (unknown) residues in initial sequence. |
| `seq` | `str` | `''` | Starting binder sequence (for refiner mode). Empty = random. |
| `refiner_mode` | `bool` | `False` | Refine the sequence supplied by seq instead of initializing a de novo binder trajectory. |
| `msa_mode` | `str` | `'mmseqs'` | MSA mode: 'mmseqs' or 'single'. |
| `boltz_model_version` | `str` | `'boltz2'` | Boltz model version. |
| `high_iptm_threshold` | `float` | `0.8` | ipTM threshold for saving designs. |
| `high_plddt_threshold` | `float` | `0.8` | pLDDT threshold for saving designs. |
| `cyclic` | `bool` | `False` | Enable cyclic peptide design. |
| `alanine_bias` | `bool` | `False` | Apply negative alanine bias during design. |
| `randomly_kill_helix_feature` | `bool` | `False` | Disrupt binder i/i+4 pair features during diffusion to reduce alpha-helical collapse and explore alternative topologies. |
| `negative_helix_constant` | `float` | `0.2` | Magnitude of the random negative i/i+4 pair-feature scaling used when randomly_kill_helix_feature is enabled. |
| `omit_AA` | `str` | `'C'` | Amino acids to omit from design (e.g., 'C'). |
| `contact_residues` | `str` | `''` | Target chain residue positions for contact specification. Comma-separated, pipe-separated for multiple chains (e.g., ... |
| `template_path` | `str` | `''` | PDB code or path(s) for template conditioning. |
| `template_cif_chain_id` | `str` | `''` | Chain ID(s) for mmCIF templates. |
| `ccd_path` | `str` | `<computed at runtime>` | Path to CCD molecule data directory (.pkl files) for Boltz. |
| `boltz_model_path` | `str` | `<computed at runtime>` | Path to Boltz model checkpoint. |
| `gpu_id` | `int` | `0` | GPU device ID. |

---

### proteina_complexa

*Co-design all-atom protein binders around a fixed protein target.*

> Proteina-Complexa: Scaling Atomistic Protein Binder Design with Generative Pretraining and Test-Time Compute
> [Paper](https://arxiv.org/abs/2603.27950) · [Upstream code / documentation](https://github.com/NVIDIA-BioNeMo/Proteina-Complexa) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/proteina_complexa/proteina_complexa.py)

**Inputs and outputs:** [Binder configuration guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/design/proteina_complexa/input.md). Single-pass all-atom protein binder generation from a target PDB, target selection and hotspots. Outputs include target and designed binder. Run a downstream structure predictor to evaluate binding geometry and confidence; generation alone is not a binding-quality result.

Class: `ProteinaComplexaBlock` · Input: `False` · Resources: partition=`gpu`, gpu_class=`large`, mem=`48GB`, time=`5:00:00` · Container: `proteina_complexa`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`target_pdb`</span> | `str` | **required** | Target PDB; copied into each job sandbox |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`target_selection`</span> | `str` | **required** | Upstream target input specification, e.g. A1-115 |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`hotspots`</span> | `list[str]` | **required** | Target residue identifiers, e.g. ['A37', 'A39'] |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `binder_length` | `tuple` | `(80, 120)` | Inclusive minimum and maximum binder residue count |
| `num_designs` | `int` | `1` | Designs generated per batch |
| `n_batches` | `int` | `1` | Independent engine jobs, each with a distinct seed |
| `batch_size` | `int` | `1` | Simultaneous samples on one GPU |
| `num_steps` | `int` | `400` | Flow integration steps; upstream default is 400 |
| `seed` | `int` | `0` | Base seed, incremented by engine chunk ID |
| `checkpoint_dir` | `str` | **required** | Directory containing public complexa.ckpt and complexa_ae.ckpt checkpoints |

---

### rfd1

*RFdiffusion v1 block for backbone-level protein design.*

> [Upstream code / documentation](https://github.com/RosettaCommons/RFdiffusion) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/rfdiffusion1.py)

Class: `RFD1Block` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`16GB`, time=`5:00:00` · Container: `rfdiffusion1`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`contigs`</span> | `str` | **required** | Contig specification. Segments separated by /, chains by spaces. E.g. '150-150' for unconditional, '5-15/A10-25/30-40... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_designs`</span> | `int` | `10` | Number of designs to generate per chunk |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`input_pdb`</span> | `Optional[str]` | `None` | Input PDB for motif scaffolding or binder design |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`config_name`</span> | `str` | `'base'` | Hydra config name (e.g. 'base', 'symmetry') |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `symmetry` | `Optional[str]` | `None` | Symmetry type for oligomer design (e.g. 'c3', 'c5', 'd3', 'tetrahedral'). Sets config_name='symmetry' automatically. |
| `diffuser_T` | `int` | `50` | Number of denoising timesteps |
| `partial_T` | `Optional[int]` | `None` | Partial diffusion timesteps (for design diversification) |
| `noise_scale_ca` | `Optional[float]` | `None` | Translation noise scaling (denoiser.noise_scale_ca) |
| `noise_scale_frame` | `Optional[float]` | `None` | Rotation noise scaling (denoiser.noise_scale_frame) |
| `ckpt_override_path` | `Optional[str]` | `None` | Custom checkpoint path. Defaults to container's Base_ckpt.pt. Common options: Complex_base_ckpt.pt, Complex_Fold_base... |
| `hotspot_res` | `Optional[list[str]]` | `None` | Hotspot residues for binder design, e.g. ['A30', 'A33', 'A34'] |
| `inpaint_seq` | `Optional[list[str]]` | `None` | Inpaint sequence positions, e.g. ['A1', 'A30-40'] |
| `inpaint_str` | `Optional[list[str]]` | `None` | Inpaint structure positions (flexible regions), e.g. ['B165-178'] |
| `scaffold_guided` | `bool` | `False` | Enable fold-conditioned design with ss/adj constraints |
| `target_pdb` | `Optional[str]` | `None` | Target PDB for scaffold-guided binder design |
| `scaffold_dir` | `Optional[str]` | `None` | Directory of _ss.pt and _adj.pt files for scaffold guidance |
| `extra_hydra_args` | `Optional[list[str]]` | `None` | Additional Hydra CLI overrides passed verbatim, e.g. ['diffuser.chi_type=interp'] |
| `raise_on_missing_trbs` | `bool` | `True` | Raise error if TRB file is missing for a design |

---

### rfd2

*RFdiffusion2 (rf_diffusion) block for atomic-level protein design.*

> RFdiffusion2
> [Upstream code / documentation](https://github.com/RosettaCommons/RFdiffusion2) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/rfdiffusion2.py)

Class: `RFD2Block` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`5:00:00` · Container: `rfdiffusion2_portable`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`inputs`</span> | `str \| list[str]` | **required** | Hydra config name to use (references a yaml/json file in benchmark configs) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_designs`</span> | `int` | `1` | Number of designs to generate per benchmark case |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`ckpt_path`</span> | `str` | `<computed at runtime>` | Path to RFdiffusion2 checkpoint file. The portable image defaults to the RFD_173.pt baked into it. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `raise_on_missing_trbs` | `bool` | `True` | Raise error if TRB file is missing for a design |

---

### rfd3

*RFdiffusion3 block for backbone generation.*

> Butcher, Krishna et al. (2025) — De novo Design of All-atom Biomolecular Interactions with RFdiffusion3
> [Paper](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2) · [Upstream code / documentation](https://github.com/RosettaCommons/foundry) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfdiffusion3.py)

Class: `RFD3Block` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`16GB`, time=`5:00:00` · Container: `foundry`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`inputs`</span> | `str \| dict \| None` | `None` | Path to input json/yaml file, or dict of design specifications. JSON inputs take the following nested top-level struc... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`ckpt_path`</span> | `Optional[str]` | `<computed at runtime>` | Optional checkpoint path (foundry will manage checkpoints itself) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`batch_size`</span> | `int` | `8` | Number of designs to diffuse per batch |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`n_batches`</span> | `int` | `2` | Number of diffusion batches to run (in total) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`seed_offset`</span> | `Optional[int]` | `42` | Increments seed by 1 from offset for every chunk |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `compile_model` | `Optional[bool]` | `None` | Compile hot diffusion modules; None inherits the Foundry runtime default. |
| `json_keys_subset` | `list` | `<computed at runtime>` | Keys to fetch from json |
| `jitter_ori_sigma` | `float` | `0` | Jitter origin of the noise cloud by this standard deviation |
| `convert_config_to_test_config` | `bool` | `False` | When True, shorten generated protein length to 50 residues and use fast inference settings for testing. Only shortens... |
| `inference_sampler` | `dict` | `<computed at runtime>` | Sampler parameters to override (not configurable via CLI) |

---

### rfd3_partial

*RFdiffusion3 partial diffusion. Useful for diversifying structures.*

> [Upstream code / documentation](https://github.com/RosettaCommons/foundry) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfdiffusion3.py)

Class: `RFD3PartialBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`small`, mem=`16GB`, time=`5:00:00` · Container: `foundry`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`inputs`</span> | `str \| dict \| None` | `None` | Path to input json/yaml file, or dict of design specifications. JSON inputs take the following nested top-level struc... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`ckpt_path`</span> | `Optional[str]` | `<computed at runtime>` | Optional checkpoint path (foundry will manage checkpoints itself) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`batch_size`</span> | `int` | `8` | Number of designs to diffuse per batch |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`n_batches`</span> | `int` | `2` | Number of diffusion batches to run (in total) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`seed_offset`</span> | `Optional[int]` | `42` | Increments seed by 1 from offset for every chunk |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `compile_model` | `Optional[bool]` | `None` | Compile hot diffusion modules; None inherits the Foundry runtime default. |
| `json_keys_subset` | `list` | `<computed at runtime>` | Keys to fetch from json |
| `jitter_ori_sigma` | `float` | `0` | Jitter origin of the noise cloud by this standard deviation |
| `convert_config_to_test_config` | `bool` | `False` | When True, shorten generated protein length to 50 residues and use fast inference settings for testing. Only shortens... |
| `inference_sampler` | `dict` | `<computed at runtime>` | Sampler parameters to override (not configurable via CLI) |
| `partial_t` | `int` | `5` | Timestep for partial diffusion (e.g., 5 = ~80% diffused). |
| `partial_diffusion_mode` | `str` | `'indexed'` | Mode for partial diffusion |
| `ligand_name` | `Optional[str]` | `None` | Ligand name for partial diffusion specs (e.g., 'LIG_B'). Overrides ligand from metadata when set. |

---

### rfd3na

*RFdiffusion3NA block for nucleic-acid and mixed-polymer design.*

> Butcher, Krishna et al. (2025) — RFdiffusion3NA multipolymer design
> [Paper](https://www.biorxiv.org/content/10.1101/2025.09.18.676967v2) · [Upstream code / documentation](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3na) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfd3na.py)

Class: `RFD3NABlock` · Input: `False` · Resources: partition=`gpu`, gpu_class=`small`, mem=`32GB`, time=`6:00:00` · Container: `foundry`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`inputs`</span> | `str \| dict \| None` | `None` | Path to input json/yaml file, or dict of design specifications. JSON inputs take the following nested top-level struc... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`ckpt_path`</span> | `Optional[str]` | `<computed at runtime>` | RFD3NA checkpoint path. Set to 'rfd3na' or None to use Foundry's checkpoint registry/search path. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`batch_size`</span> | `int` | `8` | Number of designs to diffuse per batch |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`n_batches`</span> | `int` | `2` | Number of diffusion batches to run (in total) |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`seed_offset`</span> | `Optional[int]` | `42` | Increments seed by 1 from offset for every chunk |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `compile_model` | `Optional[bool]` | `None` | Compile hot diffusion modules; None inherits the Foundry runtime default. |
| `json_keys_subset` | `list` | `<computed at runtime>` | Keys to fetch from json |
| `jitter_ori_sigma` | `float` | `0` | Jitter origin of the noise cloud by this standard deviation |
| `convert_config_to_test_config` | `bool` | `False` | When True, shorten generated protein length to 50 residues and use fast inference settings for testing. Only shortens... |
| `inference_sampler` | `dict` | `<computed at runtime>` | RFD3NA inference-sampler overrides. |
| `read_sequence_from_sequence_head` | `bool` | `False` | Upstream README recommends False as the global setting for RFD3NA. |
| `prevalidate_inputs` | `bool` | `True` | Validate every RFD3NA input specification before loading the model. |
| `low_memory_mode` | `bool` | `False` | Use memory-efficient tokenization when GPU memory is constrained. |
| `dump_trajectories` | `bool` | `False` | Write large noisy and denoised diffusion trajectory files. |
| `skip_existing` | `bool` | `True` | Skip designs whose output file already exists. |

---

### rfoptimization

*Refine chain A of an existing complex with RF3/Boltz and MPNN.*

> RFOptimization: Guiding Design Optimization with All-Atom Structure Prediction
> [Paper](https://www.biorxiv.org/content/10.64898/2026.09.04.749184v2) · [Upstream code / documentation](https://github.com/RosettaCommons/foundry/tree/69d5f735f272b78b6b23c230abba849f18315234/models/rfo) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfoptimization/block.py)

**Inputs and outputs:** [Configuration guide](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/foundry/rfoptimization/input.md). Refines an existing complex with designed protein chain A. The final MPNN redesign has incomplete side chains at mutations; pack/refold it before evaluating that sequence. Intermediate prediction confidence is retained in the upstream records, not attached as a score for the final redesigned structure.

Class: `RFOptimizationBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`large`, mem=`64GB`, time=`04:00:00` · Container: `rfoptimization`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `total_cycles` | `int` | `10` | Number of RF3/Boltz + MPNN cycles per input. |
| `backprop_fraction` | `float` | `0.5` | Probability of RF3 gradients per cycle; otherwise Boltz-2. |
| `mpnn_model_type` | `Literal` | `'protein_mpnn'` | — |
| `mpnn_temperature` | `float` | `0.1` | — |
| `seed` | `int` | `42` | — |
| `rf3_checkpoint` | `Optional[str]` | `None` | Public confidence-enabled RF3 checkpoint; required when RF3 is used. Defaults to configured Foundry checkpoint direct... |
| `boltz_cache` | `str` | `<computed at runtime>` | — |
| `rf3_config` | `Optional[str]` | `None` | Optional upstream optimizer YAML. Omit to use the pinned cycle_ppi preset. |
| `msa_paths` | `dict` | `<computed at runtime>` | Optional chain to A3M path mapping; no remote MSA service is used. |
| `templates` | `dict` | `<computed at runtime>` | Optional chain to template path mapping for the RF3 branch. |
| `cyclic` | `bool` | `False` | Mark chain A cyclic in Boltz only; RF3 does not constrain its closure. |

---

### rpxdock

*Dock a z-oriented C3 asymmetric unit into an I3 cage and emit C3 trimers*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/rpxdock.py)

Input geometry must use the C3 symmetry axis as its z axis and retain the asymmetric unit's radial displacement. Outputs retain motif correspondence, input lineage and each model's RPX score. Changing scoring residue selection requires checking score thresholds; the score alone does not validate a complete cage or experimental assembly.

Class: `RPXDockBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`02:00:00` · Container: `rpxdock`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `nout_top` | `int` | `1` | Number of highest scoring C3 trimers per input |
| `beam_size` | `int` | `10000` | RPXdock hierarchical search beam size |
| `ignored_aas` | `str` | `'CGP'` | Residue types excluded from RPX scoring; use an empty string for glycine-placeholder backbones. |

---

### score_val_structures

*Score the last (predicted) structure against the first (reference) one.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/validation/score.py)

Class: `ScoreValStructuresBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `metrics` | `list[ValidationMetric]` | `<computed at runtime>` | Scoring backends to run: rf3_allatom_lddt and/or peppr. |
| `remap_outputs_to_input` | `bool` | `False` | Map a renumbered monomer prediction onto the single scored protein pn-unit before scoring. Disabled by default so dir... |
| `min_atom_coverage` | `float` | `0.95` | Minimum fraction of resolved reference atoms that a remapped oracle output must represent. Used only when remap_outpu... |

---

### simplefold

*Apple SimpleFold single-sequence, protein-monomer structure prediction.*

> [Upstream code / documentation](https://github.com/apple/ml-simplefold) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/prediction/simplefold.py)

Class: `SimpleFoldBlock` · Input: `True` · Resources: partition=`gpu`, gpu_class=`large`, mem=`96GB`, time=`24:00:00` · Container: `simplefold`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`diffusion_batch_size`</span> | `int` | `1` | Number of independently sampled conformers per protein. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`read_most_confident_model`</span> | `bool` | `True` | If True, read only the highest-confidence model; False, read all samples. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`num_diffusion_steps`</span> | `int` | `500` | Number of Euler-Maruyama flow-sampling steps. |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`model`</span> | `Literal` | `'simplefold_100M'` | Released SimpleFold folding checkpoint. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `n_recycles` | `Optional[int]` | `None` | Number of trunk recycling steps. None uses the model default. |
| `ckpt_path` | `Optional[str]` | `None` | Optional checkpoint path for model weights. |
| `compile_as_metric` | `bool` | `False` | If True, compile outputs as metrics with path_handle instead of path. |
| `remove_ligands` | `bool` | `False` | If True, skip ligands in input preparation. |
| `specified_smiles` | `dict` | `<computed at runtime>` | Map ligand name → SMILES string. |
| `specified_ccd` | `dict` | `<computed at runtime>` | Map ligand name → CCD code. |
| `disable_ccd_detection` | `bool` | `False` | If True, disable automatic CCD code detection. |
| `which` | `list[int] \| str` | `'last'` | Which structure(s) in each row's lineage to predict. 'last' (default) uses the most recent; a list of ints selects by... |
| `sort_by_length` | `bool` | `True` | If True (default), sort the input handler by `sort_length_key` before chunking so each chunk hits a narrow JIT-bucket... |
| `sort_length_key` | `str` | `'rfd3.num_residues'` | Metric column to sort on when sort_by_length=True (dot-path into metrics JSON, as exposed by handler.unpack()). If mi... |
| `tau` | `float` | `0.01` | Diffusion coefficient scaling factor. |
| `seed` | `int` | `42` | Random seed; chunk id is added to keep chunks independent. |
| `predict_plddt` | `bool` | `False` | Run the optional pLDDT head. This additionally loads a fixed 1.6B folding model and is substantially more memory inte... |
| `raw_output_format` | `Literal` | `'pdb'` | SimpleFold's raw output format before compile-time conversion. |
| `convert_outputs_to_cif` | `bool` | `True` | Parse raw predictions into AtomArrays and write canonical CIF outputs. |
| `weights_dir` | `str` | `<computed at runtime>` | Shared directory for folding, ESM2, and CCD downloads. |

---

### tmol_interface

*Protein<->ligand interface energetics (single-point, beta_nov2016).*

> [Upstream code / documentation](https://github.com/uw-ipd/tmol) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/tmol.py)

Class: `TmolInterfaceBlock` · Input: `True` · Resources: partition=`cpu`, mem=`16GB`, time=`5:00:00` · Container: `tmol`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_smiles` | `Optional[str]` | `None` | SMILES of the ligand. Used to generate a tmol .tmol params file (typing only — input ligand coordinates are scored as... |
| `ligand_params_files` | `Optional[list[str]]` | `None` | Pre-made tmol .tmol params file(s) for the ligand(s). Most deterministic typing route; bypasses SMILES generation. |

---

### tmol_minimize

*Cartesian energy minimisation (beta_nov2016_cart), all atoms free.*

> [Upstream code / documentation](https://github.com/uw-ipd/tmol) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/networks/tmol.py)

Class: `TmolMinimizeBlock` · Input: `True` · Resources: partition=`cpu`, mem=`16GB`, time=`5:00:00` · Container: `tmol`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_smiles` | `Optional[str]` | `None` | SMILES of the ligand. Used to generate a tmol .tmol params file (typing only — input ligand coordinates are scored as... |
| `ligand_params_files` | `Optional[list[str]]` | `None` | Pre-made tmol .tmol params file(s) for the ligand(s). Most deterministic typing route; bypasses SMILES generation. |
| `fix_backbone` | `bool` | `False` | Freeze backbone and minimise sidechains only (default: all free). |

---

## Metric Blocks

*Blocks that compute metrics on existing structures — RMSD, contacts, energetics, clustering.*

### aromatic_neighbors

*Count aromatic residues near ligand atoms.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/aromatic_neighbors.py)

**Outputs and interpretation:** Counts aromatic residues near selected ligand atoms under the configured distance cutoff. This is a geometric neighborhood count, not a pi-stacking energy or proof of a favorable interaction.

Class: `AromaticNeighborsBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `distance_threshold` | `float` | `6` | Distance threshold in Angstroms for finding aromatic neighbors |
| `ligand_atom_names` | `Optional[list[str]]` | `None` | Optional canonical ligand atom names to use as the distance query. None counts neighbors around the complete ligand. |

---

### chirality_check

*Per-row chirality check vs a reference ligand.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/chirality.py)

**Outputs and interpretation:** Compares assigned per-atom CIP stereochemical codes with a reference ligand. A matching code requires a resolvable atom correspondence; missing or unassigned centers should not be assumed correct.

Class: `ChiralityCheckBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `reference_cif` | `str` | **required** | Path to reference ligand structure (CIF/PDB) with bond records. Atom names in this reference must match those in the ... |
| `ligand_chain` | `Optional[str]` | `None` | Chain ID for the ligand atoms in row structures. None auto-detects via the `hetero` annotation (any atom with hetero=... |

---

### clash

*Count non-bonded backbone heavy-atom clashes in a structure.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/clash.py)

**Outputs and interpretation:** Counts backbone-heavy-atom pairs (N/CA/C/O) closer than the cutoff in Å, excluding atoms in the same residue and consecutive residues of one chain. This is a geometric clash screen, not a full steric energy calculation.

Class: `ClashMetricsBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `cutoff` | `float` | `1.5` | Distance (A) below which a non-bonded backbone atom pair counts as a clash. |
| `backbone_only` | `bool` | `True` | Restrict to backbone heavy atoms (N/CA/C/O). Set False to use all heavy atoms — note this is not comparable across ge... |
| `sequence_separation` | `int` | `1` | Residues this far apart or closer in the same chain are treated as bonded and excluded. 1 excludes intra-residue and ... |
| `exclude_hetero` | `bool` | `True` | Drop hetero (ligand/cofactor) atoms before scoring. Required for correctness whenever a ligand is a canonical amino a... |

---

### contact_ms

*Compute contact molecular surface between binder and target chains.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/contact_ms.py)

**Outputs and interpretation:** Distance-weighted contact molecular surface in Å²: the target surface buried by the selected binder chain under the py_contact_ms algorithm. It measures interface contact extent, not binding energy.

Class: `ContactMSBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `binder_chains` | `list[str]` | **required** | Chain IDs for the binder |
| `target_chains` | `list[str]` | **required** | Chain IDs for the target |
| `target_atom_names` | `Optional[list[str]]` | `None` | Restrict CMS aggregation to these target atom names (e.g. ['C12', 'O3']). The molecular surface is still generated fr... |
| `target_residues` | `Optional[list[str]]` | `None` | Restrict CMS aggregation to these target residues, as chain+number identifiers (e.g. ['B56', 'B121']). Use for epitop... |

---

### dssr_motif_base_pairing

*DSSR pairing and native classification retention for a mapped motif.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/dssr_snap_metrics.py)

**Outputs and interpretation:** Compares DSSR base-pair assignments and native pairing classifications for motif residues mapped by `diffused_index_map`. Per-residue pairing and aggregate retention diagnostics preserve the full native/predicted pair annotations; SNAP is not used.

Class: `DSSRMotifBasePairingBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `x3dna_dssr_bin` | `Optional[str]` | `<computed at runtime>` | Path to x3dna-dssr. Defaults to $X3DNA_BIN_DIR/x3dna-dssr. |
| `x3dna_home` | `Optional[str]` | `<computed at runtime>` | Value exported as $X3DNA while invoking x3dna-dssr. |
| `reference` | `str` | **required** | Native RNA/DNA structure used to derive partner annotations. |
| `residues` | `list[str]` | **required** | Reference motif residue ids to report, e.g. ['A10', 'B4']. |
| `continue_on_error` | `bool` | `False` | Return per-row status/error diagnostics instead of aborting the metric block when predicted-structure DSSR processing... |
| `which` | `list[int]` | `[0, -1]` | Structures carrying diffused_index_map and predicted coordinates. |

---

### dssr_snap_metrics

*Run DSSR + SNAP on each row's last structure and attach 11 metrics.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/dssr_snap_metrics.py)

**Outputs and interpretation:** Counts protein–DNA hydrogen bonds classified as major-groove, minor-groove, or backbone contacts, plus hydrogen-bond support and SNAP interaction categories. SNAP-derived fields are missing if SNAP fails; missing values are not zero interactions. See the implementation for all 11 keys.

Class: `DSSRSnapMetricsBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `x3dna_dssr_bin` | `Optional[str]` | `<computed at runtime>` | Path to the x3dna-dssr binary. Defaults to ``$X3DNA_BIN_DIR/x3dna-dssr`` from the environment (configure these in .en... |
| `x3dna_snap_bin` | `Optional[str]` | `<computed at runtime>` | Path to the x3dna-snap binary. Defaults to ``$X3DNA_BIN_DIR/x3dna-snap`` from the environment. |
| `x3dna_home` | `Optional[str]` | `<computed at runtime>` | Value to export as $X3DNA when invoking the binaries (the 3DNA install root containing config/, lib/, etc.). Defaults... |
| `skip_snap` | `bool` | `False` | If True, skip SNAP entirely (fast path) — SNAP-derived metrics will be NaN. |

---

### foldseek

*Cluster structures by structural similarity using Foldseek.*

> van Kempen et al. (2024) — Fast and accurate protein structure search with Foldseek
> [Paper](https://doi.org/10.1038/s41587-023-01773-0) · [Upstream code / documentation](https://github.com/steineggerlab/foldseek) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/foldseek.py)

**Outputs and interpretation:** Groups structures by structural similarity at the requested Foldseek thresholds. Cluster counts describe diversity under that clustering protocol, rather than a per-design correctness or activity score.

Class: `FoldseekBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `foldseek_exec` | `str` | `<computed at runtime>` | Path to foldseek binary |
| `tm_thresholds` | `list[float]` | `[0.5, 0.6, 0.7, 0.8]` | TM-score thresholds for clustering. Results are output as cluster_id.<threshold> and n_clusters.<threshold> for each. |
| `alignment_type` | `int` | `1` | 1=structure only, 2=structure+sequence |
| `min_seq_id` | `float` | `0.0` | Min sequence identity for clustering (alignment_type=2 only) |
| `multimer_tm_threshold` | `float` | `0.65` | TM threshold for multimer clustering |
| `interface_lddt_threshold` | `float` | `0.65` | Interface lDDT threshold for multimer clustering |
| `chains` | `Optional[list[str]]` | `None` | Chain IDs to cluster on. If None, all chains are used. |
| `group_by_task` | `bool` | `True` | If True, cluster within each task group separately (uses the first structure's metadata.task field). |

---

### hbond_metrics

*Compute hydrogen bond metrics between ligand and protein.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/hbond.py)

**Outputs and interpretation:** HBPLUS identifies hydrogen bonds between ligand/motif and surrounding protein atoms. Outputs count the selected interactions, optionally by ligand atom group. Counts depend on the input protonation and HBPLUS geometry rules and are not hydrogen-bond energies.

Class: `HBondBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `atom_groups` | `Optional[dict]` | `None` | Mapping of group name to list of ligand atom names. When provided, per-group h-bond counts are included in the output. |
| `verbose` | `bool` | `False` | When True, output also includes a `pairs` list with per-hbond atom-level detail (protein_residue, protein_atom, ligan... |
| `hbplus_path` | `Optional[str]` | `None` | Explicit HBPLUS executable. Resolved once during setup and propagated to workers so they do not depend on worker-loca... |

---

### ipsae

*Compute ipSAE between a binder chain and one or more target chains.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ipsae.py)

**Outputs and interpretation:** Interface predicted Score from Aligned Errors summarizes inter-chain confidence from a predictor's PAE matrix using the configured distance/PAE cutoffs. Higher confidence does not prove binding. Missing PAE cannot be treated as a passing score. See the [upstream implementation](https://github.com/flyark42/ipsae).

Class: `IpSAEBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `binder_chain` | `str` | `'A'` | Chain ID of the designed binder. Default 'A' matches RFD3 output. |
| `target_chains` | `Optional[list[str]]` | `None` | Chain IDs of the target. If None, all non-binder chains in the prediction are used. |
| `pae_cutoff` | `float` | `10.0` | PAE threshold for inclusion in the score (Å). Default 10.0 per Park & Cong. |

---

### ligand_clash

*Reject severe protein-ligand heavy-atom penetrations cheaply.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ligand_clash.py)

**Outputs and interpretation:** `n_clashes` counts selected protein–ligand heavy-atom pairs separated by less than `cutoff` Å; `minimum_distance` and `closest_pair` identify the nearest contact. This simple distance test does not distinguish intentional covalent bonds.

Class: `LigandClashMetricsBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_resnames` | `list[str]` | **required** | Ligand residue names to score, e.g. ['ADN']. |
| `cutoff` | `float` | `2.0` | Distance (A) below which a severe penetration is counted. |

---

### ligand_ddg

*Compute Rosetta DDG (binding energy) for protein-ligand complexes.*

> Rosetta DDG binding energy calculation
> [Upstream code / documentation](https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/filter_pages/DdgFilter) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ligand_ddg.py)

**Outputs and interpretation:** `ddg_norepack` is Rosetta's bound-minus-separated score across jump 1, in Rosetta energy units; more negative values favor the bound state under that score function. The bundled protocol uses `beta_nov16`, minimizes before scoring, and disables repacking in the Ddg filter (`repack=0`, one repeat). It is not an experimental binding free energy or kcal/mol estimate. Check that jump 1 separates the intended partners. See the [exact protocol](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/scripts/ddg_protocol.xml) and [flags](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/scripts/flags_rescore_genpot); overriding either changes the interpretation.

Class: `LigandDDGBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `rosetta_bin` | `str` | `<computed at runtime>` | Path to rosetta_scripts binary |
| `protocol` | `str` | `'<checkout>/src/blocks/scripts/ddg_protocol.xml'` | Path to Rosetta XML protocol file |
| `flags` | `str` | `'<checkout>/src/blocks/scripts/flags_rescore_genpot'` | Path to Rosetta flags file |
| `dalphaball_bin` | `Optional[str]` | `<computed at runtime>` | DAlphaBall executable required by the bundled Holes filter; also searches PATH. |
| `override_param_file` | `Optional[str]` | `None` | Override path to ligand .params file for -extra_res. Bypasses auto-resolution from ligand name. |
| `override_ligand_names` | `Optional[list[str]]` | `None` | Override ligand 3-letter codes (e.g. ['FAD', 'IAI']). Bypasses auto-detection from structure metadata. The first matc... |
| `ligand_map` | `Optional[dict]` | `None` | Rename ligand residues in the atom array before detection and PDB writing. Maps old res_name to new res_name, e.g. {'... |

---

### ligand_ddg_soft

*Soft-repulsive ddG (pack hard, score soft, trimmed mean over replicates).*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ligand_ddg_soft.py)

**Outputs and interpretation:** `ddg_soft` is the mean bound-minus-separated Rosetta score after packing with the hard beta_nov16 function and scoring with beta_nov16_soft. With at least three replicates the best and worst are dropped; `ddg_soft_std` reports the retained replicate spread. Values are Rosetta energy units, not measured binding free energies; do not compare numerically with the hard-score `ligand_ddg` protocol.

Class: `LigandDDGSoftBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `replicates` | `int` | `5` | Pack+score cycles. ≥3 → trimmed mean (drop best+worst). |
| `binder_chain` | `str` | `'A'` | Chain ID of the binder. |
| `ligand_chain` | `str` | `'B'` | Chain ID of the ligand. |
| `interesting_atoms` | `Optional[list[str]]` | `None` | Ligand heavy-atom names to keep (plus their bonded H). Params is generated from the subset. |
| `ph` | `float` | `7.4` | Target pH for Dimorphite-DL protonation of H-less ligands. |
| `override_params_file` | `Optional[str]` | `None` | Pre-made .params file. Bypasses protonation and params generation — ligand H must already match. |
| `ref_smiles` | `Optional[str]` | `None` | Reference SMILES for rebuilding the ligand bond graph when atom names don't follow CCD convention (common in diffusio... |
| `cleanup_params_cache` | `bool` | `True` | Delete the shared params cache when the last chunk exits. |

---

### ligand_pocket

*Quantify ligand CMS, enclosure, shell depth, and contact distribution.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ligand_pocket.py)

**Outputs and interpretation:** Combines contact molecular surface, heavy-atom contact coverage, pocket shell/enclosure, and polar-atom neighborhood counts for the selected ligand. Surface uses Å² and distance cutoffs use Å; contact counts and coverage summarize geometry, not interaction energy.

Class: `LigandPocketMetricsBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_resnames` | `list[str]` | **required** | Ligand residue names to measure, e.g. ['ADN']. |
| `core_atom_names` | `list[str]` | `<computed at runtime>` | Ligand core atoms that must all receive protein contact. |
| `tail_polar_atom_names` | `list[str]` | `<computed at runtime>` | Ligand tail atoms requiring explicit polar support. |
| `contact_distance` | `float` | `5.0` | Protein-heavy-atom distance defining coverage and shell. |
| `polar_distance` | `float` | `3.6` | Maximum protein N/O/S distance defining polar support. |
| `clash_distance` | `float` | `2.0` | Protein-ligand heavy-atom distance below which a severe clash is counted. |

---

### metal3d

*Score each row's structure for transition-metal binding sites.*

> [Upstream code / documentation](https://github.com/lcbc-epfl/metal-site-prediction) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/metal3d.py)

**Outputs and interpretation:** A learned three-dimensional probability field for transition-metal binding sites. Predicted site confidence and position describe the model's output, not experimentally confirmed metal occupancy.

Class: `Metal3DBlock` · Input: `True` · Resources: partition=`cpu`, mem=`16GB`, time=`5:00:00` · Container: `metal3d`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`metalbinding`</span> | `bool` | `True` | If True, only voxelize residues whose sidechains can coordinate metals (HIS/CYS/ASP/GLU/ASN/GLN/MET). If False, scan ... |
| <span style="color: var(--claude-accent, #da7756); font-weight: 600">`pthreshold`</span> | `float` | `0.1` | Minimum probability for a voxel to seed a clustered site. Lower values produce more (noisier) sites. |
| <em style="color: var(--claude-text-secondary, #6b6560)">Additional parameters</em> | | | |
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `cluster_threshold` | `float` | `7.0` | Maximum distance (Å) between probe voxels in the same cluster. Larger values merge nearby sites into one. |
| `batch_size` | `int` | `128` | Inference batch size (number of voxel boxes). |
| `write_cube` | `bool` | `False` | If True, also write the volumetric probability map as a .cube file per input. Off by default — files can be large. |
| `compile_as_metric` | `bool` | `True` | If True (default), attach Metal3D metrics to the parent row without adding the probefile to the lineage — same conven... |

---

### motif_confidence

*Motif vs scaffold pLDDT and PAE of an oracle refold.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/motif_confidence.py)

**Outputs and interpretation:** Summarizes predictor pLDDT and PAE separately for the mapped motif and scaffold. Higher pLDDT and lower PAE indicate model confidence, not experimentally validated accuracy; the upstream confidence scale and residue mapping must be preserved.

Class: `MotifConfidenceBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `which` | `list[int]` | `[0, -1]` | Two indices into each row's structures list: the structure carrying diffused_index_map, and the prediction to score. |
| `residues` | `Optional[list[str]]` | `None` | Reference residue ids to treat as the motif, e.g. ['A10', 'A11']. Defaults to every polymer entry in diffused_index_map. |
| `chain` | `Optional[str]` | `None` | Designed chain to score. Defaults to the chain the mapped motif residues land in. |

---

### motif_geometry

*Named distances and angles between motif atoms, addressed by reference id.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/motif_reference.py)

**Outputs and interpretation:** Reports named distances in Å and angles between selected motif atoms addressed using reference residue identifiers. Compare each quantity against the intended geometry rather than treating all distances as smaller-is-better.

Class: `MotifGeometryBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `measurements` | `Dict` | **required** | Name -> atom specs as '<reference residue id>.<atom name>', e.g. {'contact': ["A24.N1", "B4.O2'"]}. Two specs give a ... |
| `which` | `list[int]` | `[0, -1]` | Two indices into each row's structures list: the structure carrying diffused_index_map, and the structure to measure. |

---

### mpnn_score

*Attach MPNN per-sequence confidence (from FASTA) as metrics to each row.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/mpnn_score.py)

**Outputs and interpretation:** Copies per-sequence `overall_confidence`, `seq_rec`, and optional `ligand_confidence` from MPNN outputs. Confidence is an upstream model ranking signal and sequence recovery measures agreement with the input sequence; neither establishes folding or binding success. Backend definitions can differ.

Class: `MpnnScoreBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |

---

### peppr

*Compute structural evaluation metrics using peppr.*

> PePPr — Package for Evaluation of Predicted Poses
> [Upstream code / documentation](https://github.com/aivant/peppr) · [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/peppr.py)

**Outputs and interpretation:** PEPPR computes the requested structure and interface comparison metrics against a reference; there is no single universal PEPPR score. Interpret each returned metric using its upstream definition and retain the atom/chain mapping and reference used for the calculation.

Class: `PepprBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `metrics` | `list[PepprMetric]` | `[<PepprMetric.RMSD: 'rmsd'>, <PepprMetric.TM_SCORE: 'tm_score'>, <PepprMetric.LDDT: 'lddt'>, <PepprMetric.LDDT_PLI: 'lddt_pli'>, <PepprMetric.FNAT: 'fnat'>, <PepprMetric.IRMSD: 'irmsd'>, <PepprMetric.LRMSD: 'lrmsd'>, <PepprMetric.DOCKQ: 'dockq'>]` | Which metrics to compute. Options: rmsd, tm_score, lddt, lddt_pli, fnat, irmsd, lrmsd, dockq |
| `rmsd_threshold` | `float` | `2.0` | RMSD threshold (Angstroms) for MonomerRMSD classification |
| `tolerate_exceptions` | `bool` | `True` | If True, continue when individual metrics fail (e.g., lDDT-PLI with no ligand). Failed metrics are stored as None. |

---

### plip

*Profile protein-ligand interactions using PLIP.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/plip.py)

**Outputs and interpretation:** PLIP classifies protein–ligand contacts into interaction types such as hydrogen bonds, hydrophobic contacts, and salt bridges. Returned counts and optional details depend on PLIP's geometric and chemical rules; they are not binding energies. See [PLIP](https://github.com/pharmai/plip).

Class: `PLIPBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ligand_ids` | `Optional[list[str]]` | `None` | Filter to specific binding sites by PLIP ID (format: 'HetID:Chain:Position', e.g. 'LIG:A:301'). When None, all detect... |
| `per_residue` | `bool` | `False` | Include per-residue total interaction counts (e.g. per_res.A42 = 3). |
| `include_details` | `bool` | `False` | Include per-ligand interaction breakdown when multiple binding sites are present. |
| `verbose` | `bool` | `False` | When True, output also includes `per_interaction_details`: for each interaction type, a list of dicts with atom-level... |

---

### pocket_coulomb

*Electrostatic complementarity between charged ligand atoms and pocket residues.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/pocket_coulomb.py)

**Outputs and interpretation:** An approximate charge-based pocket interaction score using the configured charge and distance treatment. Its scale depends on those settings and is not an experimental affinity.

Class: `PocketCoulombBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `pocket_radius` | `float` | `8.0` | CB distance cutoff (A) for defining pocket residues around the ligand |
| `distance_floor` | `float` | `1.5` | Minimum distance (A) to prevent singularities from clashes |
| `distance_cutoff` | `float` | `12.0` | Ignore charged protein atoms beyond this distance (A) from any charged ligand atom |
| `power` | `float` | `2.0` | Distance exponent for coulomb-like decay (2.0 = implicit solvent screening) |
| `ref_smiles` | `Optional[str]` | `None` | Reference SMILES for rebuilding ligand bond graph when atom names don't follow CCD convention (diffusion model outputs) |
| `per_residue` | `bool` | `False` | Include per-residue score breakdown in output metrics |

---

### ppi_ddg

*Protein-protein binding energy via FastRelax + ddG.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/ppi_ddg.py)

**Outputs and interpretation:** Protein–protein binding score computed with FastRelax and a bound-versus-separated ddG calculation. These are Rosetta energy units, not experimental affinities. Partner selections, relaxation, and score-function settings must match for comparisons; consult the implementation for the returned energy terms.

Class: `PPIDDGBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `binder_chain` | `str` | `'A'` | Chain ID of the binder (translated far away for unbound score). |
| `target_chain` | `str` | `'B'` | Chain ID of the target (stays in place). |
| `relax_rounds` | `int` | `1` | FastRelax standard_repeats. 1 is fast (~1–3 min/structure on CPU); 3–5 gives lower, more converged energies. |
| `replicates` | `int` | `1` | Independent relax + ddG cycles. Mean and std are reported. Each additional replicate adds one full relax run. |
| `constrain_to_start_coords` | `bool` | `True` | Harmonic coordinate constraints during FastRelax — keeps the backbone close to the AF3-predicted geometry. |
| `constraint_weight` | `float` | `1.0` | Coordinate constraint weight. Higher = stays closer to input coords; lower = more relaxation freedom. |

---

### rasa

*Compute solvent accessible surface area (SASA/RASA) for ligand atoms.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rasa.py)

**Outputs and interpretation:** Per-atom relative solvent accessibility is computed as solvent-accessible area divided by an element-specific reference area. `ligand_rasa` is the **sum** of these fractions, not a fraction bounded by one; `mean_ligand_atom_rasa` is their mean. Smaller values indicate burial. Keep probe radius, atom groups, and backbone-only settings fixed when comparing designs.

Class: `RASABlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `atom_groups` | `Optional[dict]` | `None` | Mapping of group name to list of ligand atom names. When provided, per-group RASA aggregates are included in the output. |
| `probe_radius` | `float` | `1.4` | Radius of the solvent probe in Angstroms. |
| `backbone_only` | `bool` | `False` | If True, strip protein sidechains before SASA so ligand accessibility is measured against backbone atoms (N, CA, C, O... |

---

### reference_motif_rmsd

*RMSD of the designed motif against the structure it was taken from.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/motif_reference.py)

**Outputs and interpretation:** Motif atom RMSD in Å against the original input reference, with correspondence resolved through `diffused_index_map`. Smaller values indicate closer motif geometry; mapping coverage and atom selection matter.

Class: `ReferenceMotifRMSDBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `reference` | `str` | **required** | Path to the canonical structure the motif was taken from. |
| `which` | `list[int]` | `[0, -1]` | Two indices into each row's structures list: the structure carrying diffused_index_map, and the structure to score. D... |
| `residues` | `Optional[list[str]]` | `None` | Reference residue ids to score, e.g. ['A10-12', 'B4']. Defaults to every entry in diffused_index_map. |
| `ligands` | `list[str]` | `<computed at runtime>` | Ligand/ion residue names to include, e.g. ['MN']. RFD3NA leaves non-polymer components out of diffused_index_map, so ... |

---

### rmsd

*Compute alignment and RMSD metrics between two designs in the lineage.*

> Root-mean-square deviation between aligned structures
> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rmsd.py)

**Outputs and interpretation:** RMSD is the root mean square atom displacement in Å after fitting the selected alignment atoms. Output names such as `bkbn_align.motif_allatom_rmsd` identify the fit selection and the measured selection. Lower means closer geometry; compare only matched atoms and selections.

Class: `RMSDBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `which` | `list[int]` | `[0, -1]` | Indices into each row's structures list selecting the two structures to compare. Default [0, -1] compares first vs la... |
| `alignment_methods` | `list[SELECTIONS]` | `[<SELECTIONS.BKBN: 'bkbn'>, <SELECTIONS.MOTIF_BKBN: 'motif_bkbn'>]` | List of selections to use for computing alignment transformations |
| `rmsd_targets` | `list[SELECTIONS]` | `[<SELECTIONS.BKBN: 'bkbn'>, <SELECTIONS.MOTIF_ALLATOM: 'motif_allatom'>, <SELECTIONS.LIGANDS: 'ligand'>]` | List of selections to compute RMSD on after alignment |
| `resolve_symmetries` | `bool` | `True` | Whether to resolve automorphisms for all-atom RMSD |
| `select_mask_by_name_pairing` | `list[SELECTIONS]` | `[<SELECTIONS.LIGANDS: 'ligand'>]` | List of RMSD targets where atom masks are paired by finding common atoms (via chain_id, res_id, atom_name) instead of... |
| `component_remap` | `Optional[dict]` | `None` | Remap component names for ligand lookup (e.g., {'LIG_B': 'B;N'}). Needed when upstream tools rename ligands and RMSD ... |
| `atom_groups` | `Optional[dict]` | `None` | Mapping of group name to list of ligand atom names. When provided, per-group ligand RMSD values are included in the o... |

---

### rna_connectivity

*Real O3'-P backbone connectivity of a designed nucleic-acid chain.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rna_connectivity.py)

**Outputs and interpretation:** Detects nucleic-acid backbone breaks from the O3′ of residue i to the P atom of residue i+1 using the configured distance criterion. Missing atoms and residue mapping affect which links can be evaluated.

Class: `RNAConnectivityBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `chain` | `Optional[str]` | `None` | Chain to walk. Defaults to every chain carrying C3' atoms. |
| `max_bond_distance` | `float` | `2.0` | O3'(i)-P(i+1) distance above which the junction is a break. The bond is ~1.6 A. |
| `which` | `str \| list[int]` | `'last'` | Which structure(s) of each row to measure. Measure the diffusion output, not a refold: an oracle rebuilds a continuou... |

---

### rna_ss_f1_mcc

*Motif-restricted secondary-structure F1 / MCC against reference pairs.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rna_ss_f1_mcc.py)

**Outputs and interpretation:** F1 and Matthews correlation coefficient compare predicted/reference RNA secondary-structure pair assignments. The assigned base pairs and matching scheme define positives and negatives; inspect unresolved residues before comparing results.

Class: `RNASecondaryStructureF1Block` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `predictions_csv` | `str` | **required** | RNet2 runner output CSV. Must carry 'predicted_dbn' and the key column named by 'join_on'. |
| `join_on` | `str` | `'design_path'` | Column shared with the handler used to match a prediction to its design. Row position is NOT used - the runner preser... |
| `reference_pairs` | `list[list[str]]` | `<computed at runtime>` | Canonical (WC + G.U) reference pairs as reference residue ids, e.g. [["A5", "A10"], ["A12", "A25"]]. Resolved through... |
| `pseudoknot_pairs` | `list[list[str]]` | `<computed at runtime>` | Reference pairs excluded from R because they are pseudoknotted. |
| `compare_to_vienna` | `bool` | `True` | Also report whole-sequence F1 between RNet and ViennaRNA's MFE on the same design. Cross-method 2D disagreement is a ... |

---

### rna_thermo

*ViennaRNA Boltzmann scoring of a required base-pair set.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rna_thermo.py)

**Outputs and interpretation:** ViennaRNA estimates `p_motif = exp(-(G_constrained - G_unconstrained) / RT)`, the ensemble probability of the required motif base pairs under its thermodynamic model. Pseudoknots are excluded and reported with partial status; the model lacks an Mg²⁺ term. Use it to compare designs within a target, not as an absolute measure of full-complex folding.

Class: `RNAThermoBlock` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `required_pairs` | `list[list[str]]` | `<computed at runtime>` | Base pairs the motif requires, as reference residue ids resolved through diffused_index_map, e.g. [["A10", "A24"], ["... |
| `pseudoknot_pairs` | `list[list[str]]` | `<computed at runtime>` | Pairs that form pseudoknots. Excluded from the constraint because ViennaRNA's model cannot represent them; reported a... |
| `chain` | `Optional[str]` | `None` | Restrict to one chain id. None uses every RNA residue. |

---

### rotamer_metrics

*Compute rotamer quality metrics for protein structures.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/rotamer.py)

**Outputs and interpretation:** Reports sidechain rotamer quality against the selected library/model. Percentile and probability fields are different quantities; compare using the same backend and residue selection.

Class: `RotamerMetricsBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `backend` | `Literal` | `'dunbrack'` | Scoring backend: 'peppr' (Top8000 library via peppr) or 'dunbrack' (backbone-dependent Dunbrack library). |
| `motif_only` | `bool` | `True` | Only score residues annotated as motif (is_motif_atom). If the structure has no motif annotation, all residues are sc... |
| `include_rama` | `bool` | `True` | Also compute Ramachandran scores (peppr backend only). |
| `detailed_output` | `bool` | `False` | Include per-residue-type breakdown of scores. |
| `per_residue_output` | `bool` | `True` | Include per-original-residue scores using diffused_index_map metadata to map back to original residue identifiers (e.... |

---

### secondary_structure

*Per-fold SS composition + loop/pLDDT disorder, globally and around the ligand.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/secondary_structure.py)

**Outputs and interpretation:** Reports secondary-structure composition and loop/confidence disorder summaries globally or around a ligand pocket. Confidence interpretation requires the input B-factor field to carry the intended pLDDT values.

Class: `SecondaryStructureBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `pocket_cutoff` | `float` | `8.0` | Å cutoff for a residue to be considered in the pocket (any non-H protein atom within this distance of any ligand atom). |
| `plddt_threshold` | `float` | `70.0` | Per-residue mean b_factor below this counts as low-confidence (pLDDT semantics; b_factor in AF3/Boltz/RF3 CIFs carrie... |
| `plddt_min_signal` | `float` | `50.0` | Auto-scaling rules for b_factor → pLDDT (0-100):   - max(bf) ≤ 1.5         → assume 0-1 scale (RF3); multiply by 100.... |
| `verbose` | `bool` | `False` | When True, also emit `ss_string`, segment topology, and a `residue_records` list with per-residue {residue_id, resnam... |
| `chains` | `Optional[list[str]]` | `None` | Restrict SS computation to these chain IDs (e.g. ['B']). When None, all chains are included. |

---

### sequence_identity_gate

*Hard novelty gate on scaffold-region global sequence identity.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/sequence_identity_gate.py)

**Outputs and interpretation:** Checks identity against a reference sequence collection and reports the nearest match/gate verdict. Identity thresholds describe sequence similarity, not structural or functional novelty by themselves.

Class: `SequenceIdentityGateBlock` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `references` | `str` | **required** | FASTA of natural sequences to compare against - the native construct plus Rfam/RNAcentral homologs. |
| `threshold` | `float` | `0.4` | Maximum scaffold identity a design may have and still pass. Set per campaign; 0.40 is the RNA motif-scaffolding bench... |
| `exclude_motif` | `bool` | `True` | Score the scaffold region only. Leave True - motif identity is 100% by construction and including it makes the gate u... |
| `chain` | `Optional[str]` | `None` | Restrict to one chain id. None uses every RNA residue. |
| `compute_intra_identity` | `bool` | `False` | Also report the maximum identity to any OTHER design in this chunk, which measures whether the designer explored or e... |

---

### symmetry

*Measure how closely an assembly obeys its declared cyclic point group.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/symmetry.py)

**Outputs and interpretation:** Measures deviation of a homo-oligomeric assembly from cyclic point-group symmetry. Atom selection and subunit correspondence define the geometric comparison; symmetry alone does not establish stability.

Class: `SymmetryMetricsBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `point_group` | `str` | `'C3'` | Expected cyclic point group, e.g. 'C3'. Only Cn is supported; dihedral and polyhedral groups need an explicit subunit... |
| `chains` | `Optional[list[str]]` | `None` | Protomer chain IDs. Defaults to every protein chain in the structure, alphabetically. Name them explicitly to score o... |
| `backbone_only` | `bool` | `True` | Superimpose on backbone N/CA/C/O only. Set False to include side chains — only comparable when every protomer is full... |
| `exclude_hetero` | `bool` | `True` | Drop hetero (ligand/cofactor) atoms before scoring, so a bound cofactor never enters the protomer superposition. |

---

### tm_score

*TM-score against an external reference structure.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/tm_score.py)

**Outputs and interpretation:** TM-score compares a structure with an external reference fold using a length-normalized structural similarity score. Higher values indicate greater similarity; reference choice, aligned chains, and normalization determine comparability.

Class: `TMScoreBlock` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `reference` | `str` | **required** | Path to the reference structure (PDB/CIF). |
| `reference_chain` | `Optional[str]` | `None` | Restrict the reference to one chain id. |
| `chain` | `Optional[str]` | `None` | Restrict the target to one chain id. |

---

### tm_score_rna

*TM-score_RNA against a reference fold, computed by US-align.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/metrics/tm_score_rna.py)

**Outputs and interpretation:** RNA structural similarity against a reference, using the block's selected nucleotide anchor atoms and alignment/normalization settings. Higher TM-score means closer overall fold; it does not certify base-pair or catalytic geometry.

Class: `TMScoreRNABlock` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `reference` | `str` | **required** | Path to the reference structure (the native fold). |
| `reference_chain` | `Optional[str]` | `None` | Restrict the reference to one chain. None keeps every chain and aligns them as one unit (-ter 0). |
| `chain` | `Optional[str]` | `None` | Restrict the design to one chain. None uses all. |
| `expected_reference_length` | `Optional[int]` | `None` | Nucleotide count the reference should parse to. Checked at setup - a mis-parsed reference silently changes d0 and eve... |
| `fixed_correspondence` | `bool` | `False` | Score the residue-index correspondence as given (-TMscore 1) instead of searching for an alignment. Only meaningful w... |
| `mol` | `str` | `'RNA'` | US-align molecule type: RNA or prot. |
| `usalign_binary` | `Optional[str]` | `None` | US-align executable name or path. Defaults to the USALIGN_BIN setting, the venv's bin, then PATH. |
| `min_reference_length` | `int` | `30` | Refuse a reference shorter than this. d0 goes to zero near L=18, so TM-score_RNA is undefined on short selections - u... |

---

## Utility Blocks

*Blocks for data manipulation — ingestion, filtering, format conversion, annotation.*

### annotate

*Annotate molecular structures with interface, hbond, and ligand info.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/annotate.py)

Class: `AnnotateBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `annotations` | `list[str]` | `['interface', 'hbond', 'aromatics_near_ligand', 'ligand']` | — |
| `overwrite` | `bool` | `True` | Whether to overwrite existing annotations |
| `interface_config` | `InterfaceConfig` | `<computed at runtime>` | Configuration for interface residue annotation |

---

### filter

*Filter pipeline rows by metric range filters or a predicate function.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/filter.py)

Class: `FilterBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `expr` | `Optional[str]` | `None` | Boolean filter expression with and/or/not and parentheses over metric keys, e.g. 'af3.min_pae_interaction <= 10 and (... |
| `filters` | `Optional[dict]` | `None` | Metric key -> MetricFilter(min/max). Uses Polars-native fast path. |
| `filter_groups` | `Optional[dict]` | `None` | Named groups of filters combined with OR logic. Filters within each group use 'filter_groups_inner_operator'; groups ... |
| `filter_groups_inner_operator` | `Literal` | `'and'` | How filters within each filter_group are combined before groups are OR'd together. |
| `function` | `Optional[Callable]` | `None` | Predicate function: metrics dict -> bool (True = keep) |
| `check_column_exists_only` | `bool` | `False` | When True, validate that filter columns exist and contain valid floats (not NaN) but do not prune rows. Useful for te... |

---

### ingest

*Ingest data from various sources (dir, csv, dataframe, sequences, or compile).*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/ingest.py)

Class: `IngestBlock` · Input: `False` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `source` | `str \| PipelineDataHandler` | **required** | Data source to ingest. When compile_as is provided, this should be a comma-separated list of directory paths. Otherwi... |
| `compile_as` | `Optional[str]` | `None` | If provided, treat source as comma-separated directories and compile using the specified block (e.g., 'rfd3', 'mpnn') |

---

### materialize

*Write stored structures to disk.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/materialize.py)

Class: `MaterializeBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `output_dir` | `Optional[str]` | `None` | Target directory for materialized files. Defaults to sandbox_dir/materialized. |
| `nest_by_step` | `bool` | `True` | Organize files into subdirectories by step name (e.g. output_dir/rfd3/, output_dir/af3/). |

---

### mutate_residues

*Expand each input structure into named residue-mutation variants.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/mutate_residues.py)

Class: `MutateResiduesBlock` · Input: `True` · Resources: partition=`cpu`, mem=`4GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `variants` | `dict` | **required** | Named mutation sets, e.g. {'a73f': {'A73': 'PHE'}, 'double': {'A38': 'PHE', 'A73': 'PHE'}}. |

---

### nanoparticle_prepare

*Align antigen clash reference and preserve a C3 scaffold's symmetry axis*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/nanoparticle.py)

Requires RFdiffusion motif correspondence. Emits an asymmetric unit with motif metadata and preserved radial placement; rows exceeding the clash limit are removed. Invalid symmetry or missing correspondence fails explicitly.

Class: `NanoparticlePrepareBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `motif_pdb` | `str` | **required** | The original RFdiffusion C3 antigen stub |
| `clash_ref_pdb` | `str` | **required** | Full antigen stalk in the stub's coordinate frame |
| `max_clashes` | `int` | `8` | — |
| `max_c3_rmsd` | `float` | `2.0` | — |

---

### order_collection

*Extract sequences, check ordered status, and collect prediction metrics.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/order_collection.py)

Class: `OrderCollectionBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `ordered_sequences` | `list[str]` | `<computed at runtime>` | Ordered sequence fingerprints (from final filter) |
| `prediction_step` | `str` | `''` | Prediction step name to extract metrics from |

---

### package_submission

*Write a validated ``designs.csv`` (and optional ``run_meta.json``).*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/package_submission.py)

Class: `PackageSubmissionBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `output_dir` | `str` | **required** | Directory to write designs.csv (and run_meta.json) into |
| `expected_count` | `Optional[int]` | `None` | Exact number of designs required; None accepts any count |
| `sequence_key` | `str` | `'sequence'` | Column holding the amino-acid sequence. The default resolves standard sequence_<CHAIN> columns and structure-backed p... |
| `allow_duplicates` | `bool` | `False` | Permit repeated sequences (off: duplicates are a violation) |
| `metadata` | `Optional[dict]` | `None` | Written verbatim to run_meta.json; omitted when None |
| `extra_columns` | `Optional[list[str]]` | `None` | Additional columns to carry into designs.csv alongside sequence |

---

### remap_ligand

*Rename predicted ligand atom/residue names to match a reference CIF.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/remap_ligand.py)

Class: `RemapLigandBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `reference_cif` | `str` | **required** | Path to a ligand-only CIF with canonical atom names and _chem_comp_bond records. |
| `target_name` | `Optional[str]` | `None` | Desired residue name in outputs (e.g. 'B;N'). Read from the reference CIF if None. |
| `atom_name_mapping` | `Optional[dict]` | `None` | Optional explicit predicted-to-reference atom-name mapping. When provided, bypass graph discovery and validate each r... |
| `core_atoms` | `Optional[list[str]]` | `None` | Reference atom names of the rigid core. When given, the RMSD used for the substructure-match tiebreak is computed onl... |
| `buried_atoms` | `Optional[list[str]]` | `None` | Backward-compatible alias for core_atoms. |
| `sample_size` | `int` | `64` | Scan this many structures first before streaming the rest of the chunk. |
| `rmsd_ok` | `float` | `2.0` | Stop on the first mapping with RMSD below this threshold (Å). |
| `num_workers` | `Optional[int]` | `0` | Parallel workers for the atom-mapping scan. 0 (default) runs serial — avoids a fork() deadlock when the scan runs ins... |
| `emit_metric` | `bool` | `True` | Store the match RMSD under '<name>.match_rmsd' on every remapped row. |

---

### restore_sidechains

*Restore explicitly selected side-chain conformers from an earlier step.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/restore_sidechains.py)

Class: `RestoreSidechainsBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `positions_json` | `str` | **required** | JSON mapping reference structure paths to residue positions. |
| `reference_index` | `int` | `0` | Index of the reference structure in each lineage row. |
| `align_backbone` | `bool` | `False` | Rigidly align each reference residue's backbone onto the current backbone before copying its side chain. Use for pred... |

---

### rotamer_sweep

*Mutate named residues and enumerate explicit chi-angle combinations.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/rotamer_sweep.py)

Class: `SidechainRotamerSweepBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `mutations` | `dict` | **required** | Residue identities to install, e.g. {'A12': 'PHE'}. |
| `rotamers` | `dict` | **required** | Per-position chi vectors in degrees. Every Cartesian combination is emitted as a separate structure. |
| `max_combinations` | `int` | `1000` | Safety cap on emitted combinations per input structure. |

---

### select_chains

*Keep specified chains, preserving residue IDs and parent lineage*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/select_chains.py)

Missing chains raise an error. The output is a new CIF; motif residues, atom selections and diffusion correspondence are restricted to retained chains. This supports a true protein-only prediction branch when a complex also contains a protein or peptide partner, which ligand removal alone would retain.

Class: `SelectChainsBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `chains` | `list[str]` | **required** | — |

---

### select_diverse

*Keep a subset of rows no two of which exceed the similarity cutoffs.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/select_diverse.py)

Class: `SelectDiverseBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `max_identity` | `float` | `0.5` | Reject a design sharing more than this identity with a kept one |
| `target_count` | `Optional[int]` | `None` | Stop once this many are selected; None keeps every non-conflicting design |
| `rank_by` | `Optional[str]` | `None` | Metric column to prefer when breaking conflicts (best first) |
| `rank_ascending` | `bool` | `True` | True when a lower rank_by value is better (e.g. PAE) |
| `sequence_key` | `str` | `'sequence'` | Column holding the amino-acid sequence. The default resolves standard sequence_<CHAIN> columns and structure-backed p... |
| `require_target` | `bool` | `False` | Raise if target_count cannot be met, rather than warning |
| `required_structure_steps` | `list[str]` | `<computed at runtime>` | Only select rows containing outputs from every named structure step (for example ['rf3']); use this to prevent unscre... |

---

### select_motif

*Apply a user-defined function across structures to update motif annotations.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/select_motif.py)

Class: `SelectMotifBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `selector` | `Optional[Callable]` | `None` | Function mapping LoadedStructure -> list of residue IDs to fix |
| `metadata_key` | `str` | `'motif'` | Metadata key to write selected residues into |
| `append` | `bool` | `True` | Append to existing metadata key (True) or overwrite (False) |

---

### selective_sidechain_repack

*Resolve local protein clashes without moving the backbone or ligand.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/repack_sidechains.py)

Class: `SelectiveSidechainRepackBlock` · Input: `True` · Resources: partition=`cpu`, mem=`8GB`, time=`5:00:00`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `positions` | `list[str]` | `<computed at runtime>` | Protein residue IDs to repack, such as ['A42', 'A:73']. |
| `clash_distance` | `Optional[float]` | `None` | When set, also detect and repack each structure's residues in nonbonded protein-heavy-atom contacts below this distance. |
| `sequence_separation` | `int` | `1` | Same-chain residue separation excluded from clash detection. |
| `scorefxn_name` | `str` | `'beta_nov16'` | Rosetta score function used to select fixed-identity rotamers. |
| `extra_rotamers` | `bool` | `True` | Include extra chi1-chi4 sampling at the selected positions. |

---

### to_cif

*Parse input structures and re-save as clean CIF files.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/parse.py)

Class: `ParseToCifBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |

---

### to_pdb

*Parse input structures and re-save as clean PDB files.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/parse.py)

Class: `ParseToPdbBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |

---

### transform

*Apply an atomworks Transform to each row of the pipeline.*

> [Chaperone implementation](https://github.com/RosettaCommons/chaperone/blob/main/src/blocks/util_blocks/transform.py)

Class: `TransformBlock` · Input: `True` · Resources: Not specified

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `store_cleanup` | `bool` | `False` | When ``store='catcif'``, delete sandbox output files after they have been consolidated into the archive. Ignored othe... |
| `transform` | `Any` | `None` | An atomworks Transform instance to apply |
| `extract_metrics` | `Optional[list[str]]` | `None` | Keys to extract from the transform output as metrics |
| `extract_metadata` | `Optional[list[str]]` | `None` | Keys to extract from the transform output as metadata updates |
| `write_structure` | `bool` | `False` | If True, write result['atom_array'] back as a new CIF file |

---

## Standard Parameters

The following parameters are shared across most blocks and are omitted from the per-block tables above for brevity.

| Parameter | Description |
|-----------|-------------|
| `store` | Store structure file contents in a Delta Lake table for portable/archival runs. Default: `False`. Available on all blocks. |
| `command_template` | Override the container/binary command template. Pass a list of strings to run against a local install instead of the default container. Available on compiled blocks. |
| `cloud` | Set `True` to request Modal execution for a compiled block. Default: `False`; the engine backend can also select Modal. See [cloud setup](cloud.md). |
