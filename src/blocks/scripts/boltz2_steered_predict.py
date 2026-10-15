#!/usr/bin/env python
"""Boltz2 steered prediction script using the Protein Hunter fork.

Runs Boltz2 structure prediction with helix-steering support via
the PH fork's modified forward() method. Produces output in the
same directory layout as `boltz predict` so the standard
Boltz2Block.compile() can collect results.

Expected output structure:
    <out_dir>/boltz_results_<stem>/predictions/<name>/<name>_model_0.{cif,pdb}
    <out_dir>/boltz_results_<stem>/predictions/<name>/confidence_<name>_model_0.json
"""

import argparse
import json
import gc
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

# PH fork imports (available inside the PH container)
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.mol import load_molecules, load_canonicals
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import MSA, Coords, Ensemble, Input, StructureV2
from boltz.data.write.pdb import to_pdb
from boltz.data.write.mmcif import to_mmcif
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2

warnings.filterwarnings("ignore", message=".*requires_grad=True.*")

# Chain ID -> internal index mapping (same as PH constants)
CHAIN_TO_NUMBER = {chr(i): i - ord("A") for i in range(ord("A"), ord("K"))}


def parse_args():
    p = argparse.ArgumentParser(description="Boltz2 steered prediction")
    p.add_argument("input_dir", type=str, help="Directory of YAML input files")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument(
        "--cache", default=os.path.expanduser("~/.boltz"), help="Boltz cache dir"
    )
    p.add_argument("--output_format", default="mmcif", choices=["mmcif", "pdb"])

    # Standard boltz prediction params
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=200)
    p.add_argument("--diffusion_samples", type=int, default=1)
    p.add_argument("--max_parallel_samples", type=int, default=1)
    p.add_argument("--step_scale", type=float, default=1.638)

    # Steering params (PH fork)
    p.add_argument("--randomly_kill_helix_feature", action="store_true", default=False)
    p.add_argument("--negative_helix_constant", type=float, default=0.2)
    p.add_argument(
        "--binder_chain",
        type=str,
        default="A",
        help="Chain ID of the binder (helix steering applied to this chain)",
    )

    # Potentials
    p.add_argument("--use_potentials", action="store_true", default=False)

    return p.parse_args()


def load_model(args, device):
    """Load the PH-forked Boltz2 model."""
    checkpoint = os.path.join(args.cache, "boltz2_conf.ckpt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Boltz2 checkpoint not found at {checkpoint}. "
            f"Set --cache to the directory containing boltz2_conf.ckpt"
        )

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": args.max_parallel_samples,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = args.step_scale

    steering_args = BoltzSteeringParams()
    if not args.use_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False

    pairformer_args = PairformerArgsV2()
    pairformer_args.v2 = True
    pairformer_args.activation_checkpointing = True

    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
    msa_args.activation_checkpointing = True

    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False,
        use_templates=True,
        use_templates_v2=True,
        use_trifast=False,
        max_parallel_samples=args.max_parallel_samples,
        steering_args=asdict(steering_args),
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
    )
    model.eval()
    model.to(device)
    return model


def process_yaml(yaml_path, ccd_path, ccd_lib):
    """Parse a Boltz YAML input into a batch and structure."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    name = Path(yaml_path).stem
    target = parse_boltz_schema(name, data, ccd_lib, ccd_path, boltz_2=True)

    structure = target.structure
    coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
    ensemble = np.array([(0, len(coords))], dtype=Ensemble)

    structure = StructureV2(
        atoms=structure.atoms,
        bonds=structure.bonds,
        residues=structure.residues,
        chains=structure.chains,
        interfaces=structure.interfaces,
        mask=structure.mask,
        coords=coords,
        ensemble=ensemble,
    )

    msas = {}
    for chain in target.record.chains:
        msa_id = chain.msa_id
        if msa_id != -1:
            msa = np.load(msa_id)
            msas[chain.chain_id] = MSA(**msa)

    input_data = Input(
        structure,
        msas,
        record=target.record,
        residue_constraints=target.residue_constraints,
        templates=target.templates,
        extra_mols=target.extra_mols,
    )

    tokenizer = Boltz2Tokenizer()
    featurizer = Boltz2Featurizer()

    tokenized = tokenizer.tokenize(input_data)
    random = np.random.default_rng()

    molecules = {}
    molecules.update(ccd_lib)
    molecules.update(input_data.extra_mols)
    mol_names = set(tokenized.tokens["res_name"].tolist())
    mol_names = mol_names - set(molecules.keys())
    molecules.update(load_molecules(ccd_path, mol_names))

    batch = featurizer.process(
        tokenized,
        random=random,
        molecules=molecules,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=4096,
        pad_to_max_seqs=False,
        compute_symmetries=False,
        single_sequence_prop=0.0,
        compute_frames=True,
        inference_pocket_constraints=None,
        compute_constraint_features=True,
        compute_affinity=False,
    )

    return batch, structure, target.record


def write_outputs(output, structure, record, out_dir, output_format, model_idx=0):
    """Write structure and confidence JSON in boltz predict format."""
    name = record.id
    struct_dir = out_dir / name
    struct_dir.mkdir(parents=True, exist_ok=True)

    # Extract coords and plddt
    coords = output["coords"]
    plddts = output["plddt"].detach().cpu().numpy()[0] if "plddt" in output else None

    # Update structure coords
    structure.atoms["coords"] = (
        coords[0].detach().cpu().numpy()[: structure.atoms["coords"].shape[0]]
    )

    # Write structure file
    outname = f"{name}_model_{model_idx}"
    if output_format == "pdb":
        path = struct_dir / f"{outname}.pdb"
        with open(path, "w") as f:
            f.write(to_pdb(structure, plddts=plddts, boltz2=True))
    else:
        path = struct_dir / f"{outname}.cif"
        with open(path, "w") as f:
            f.write(to_mmcif(structure, plddts=plddts, boltz2=True))

    # Write confidence JSON
    if "plddt" in output:
        conf = {}
        for key in [
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
        ]:
            if key in output:
                val = output[key]
                if hasattr(val, "item"):
                    conf[key] = val.item()
                elif isinstance(val, torch.Tensor):
                    conf[key] = val.detach().cpu().item()
                else:
                    conf[key] = float(val)

        # pair_chains_iptm
        if "pair_chains_iptm" in output:
            pair_iptm = output["pair_chains_iptm"]
            conf["pair_chains_iptm"] = {
                str(i): {
                    str(j): (
                        pair_iptm[i][j].item()
                        if hasattr(pair_iptm[i][j], "item")
                        else float(pair_iptm[i][j])
                    )
                    for j in pair_iptm[i]
                }
                for i in pair_iptm
            }

        conf_path = struct_dir / f"confidence_{name}_model_{model_idx}.json"
        with open(conf_path, "w") as f:
            json.dump(conf, indent=4, fp=f)

    print(f"  -> {path}")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Collect YAML inputs
    input_dir = Path(args.input_dir)
    yamls = sorted(input_dir.glob("*.yaml"))
    if not yamls:
        print(f"No YAML files found in {input_dir}")
        sys.exit(1)
    print(f"Found {len(yamls)} input(s)")

    # Setup output directory to match boltz predict layout
    stem = input_dir.stem
    results_dir = Path(args.out_dir) / f"boltz_results_{stem}"
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD library
    ccd_path = os.path.join(args.cache, "ccd")
    if not os.path.exists(ccd_path):
        # Fallback: try mols/ (PH convention)
        ccd_path = os.path.join(args.cache, "mols")
    ccd_lib = load_canonicals(ccd_path)

    # Load model
    print("Loading Boltz2 model...")
    model = load_model(args, device)
    print("Model loaded")

    if args.randomly_kill_helix_feature:
        print(
            f"Helix steering ENABLED: binder_chain={args.binder_chain}, "
            f"negative_helix_constant={args.negative_helix_constant}"
        )

    # Process each input
    for yaml_path in yamls:
        name = yaml_path.stem
        print(f"Processing: {name}")

        try:
            batch, structure, record = process_yaml(yaml_path, ccd_path, ccd_lib)
        except Exception as e:
            print(f"  ERROR parsing {name}: {e}")
            continue

        # Move batch to device
        batch = {k: v.unsqueeze(0).to(device) for k, v in batch.items()}

        # Run prediction with steering
        try:
            output = model.predict_step(
                batch,
                batch_idx=0,
                dataloader_idx=0,
                randomly_kill_helix_feature=args.randomly_kill_helix_feature,
                negative_helix_constant=args.negative_helix_constant,
                binder_chain=args.binder_chain,
            )
        except Exception as e:
            print(f"  ERROR predicting {name}: {e}")
            continue

        if output.get("exception", False):
            print(f"  WARNING: OOM or exception for {name}, skipping")
            continue

        # Write outputs
        write_outputs(output, structure, record, predictions_dir, args.output_format)

        # Cleanup
        del batch, output
        gc.collect()
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    main()
