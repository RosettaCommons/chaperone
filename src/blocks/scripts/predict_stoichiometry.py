#!/usr/bin/env python
"""Predict protein stoichiometry from sequence using E1 embeddings and a trained classifier.

This script predicts the likely oligomeric state (stoichiometry) of a protein sequence
by using E1 embeddings at multiple orders and a trained MLP classifier.

USAGE (with container - recommended):
=====================================
    The recommended way to run this script is with apptainer using an E1 image.
    This ensures all dependencies (E1, torch, safetensors) are available.
    ``--weights-path`` (the trained classifier checkpoint) is required.

    # Basic usage with container
    apptainer run --nv /path/to/e1.sif \
        python /path/to/chaperone/src/blocks/scripts/predict_stoichiometry.py \
        --input-path sequences.fasta \
        --output-path predictions.csv \
        --weights-path /path/to/stoichiometry_classifier.pt

    # With directory bindings (for accessing files outside container)
    apptainer run --nv \
        --bind /path/to/data:/path/to/data \
        /path/to/e1.sif \
        python /path/to/chaperone/src/blocks/scripts/predict_stoichiometry.py \
        --input-path /path/to/data/sequences.fasta \
        --output-path /path/to/data/predictions.csv \
        --weights-path /path/to/data/stoichiometry_classifier.pt

    # Using a larger E1 model for better accuracy
    apptainer run --nv /path/to/e1.sif \
        python /path/to/chaperone/src/blocks/scripts/predict_stoichiometry.py \
        --input-path sequences.fasta \
        --output-path predictions.csv \
        --weights-path /path/to/stoichiometry_classifier.pt \
        --e1-model 600m

    # Skip E1 inference if embeddings already exist
    apptainer run --nv /path/to/e1.sif \
        python /path/to/chaperone/src/blocks/scripts/predict_stoichiometry.py \
        --input-path sequences.fasta \
        --output-path predictions.csv \
        --weights-path /path/to/stoichiometry_classifier.pt \
        --skip-e1 \
        --embeddings-dir /path/to/existing/embeddings

USAGE (without container):
==========================
    If running outside the container, ensure E1 and dependencies are installed:
        pip install E1 torch safetensors click

    python predict_stoichiometry.py --input-path sequences.fasta --output-path predictions.csv \
        --weights-path /path/to/stoichiometry_classifier.pt

OUTPUT:
=======
    CSV file with columns:
    - sequence_id: ID from the FASTA file
    - predicted_copies: Most likely number of copies (1, 2, 3, 4, 5, or "6+")
    - copies_1, copies_2, ..., copies_6plus: Probability for each stoichiometry class

MODEL ARCHITECTURE:
===================
    The classifier uses a 6-order embedding approach:
    1. For each input sequence, E1 embeddings are computed for orders 1-6
       (monomer, dimer, trimer, tetramer, pentamer, hexamer+)
    2. Each order's mean embedding (768-dim) is projected to 64-dim
    3. The 6 projected embeddings are concatenated (6*64 = 384-dim)
    4. A final linear layer predicts 6 class probabilities

    Training used balanced sampling (beta=0.5) to handle class imbalance.

DEPENDENCIES:
=============
    - torch
    - safetensors
    - click
    - E1 (Profluent-Bio/E1)

AUTHORS:
========
    Chaperone framework - https://github.com/RosettaCommons/chaperone
"""

import logging
import os
from pathlib import Path
from typing import Optional

import click
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

DEFAULT_E1_MODEL = "150m"

# Stoichiometry class labels
STOICHIOMETRY_CLASSES = [
    "copies_1",
    "copies_2",
    "copies_3",
    "copies_4",
    "copies_5",
    "copies_6plus",
]
STOICHIOMETRY_LABELS = ["1", "2", "3", "4", "5", "6+"]


# =============================================================================
# Model Architecture
# =============================================================================


class OrderEmbeddingMLP(nn.Module):
    """MLP for 6-order embeddings with per-order projection.

    Architecture:
        Input: (batch, n_orders, embed_dim) e.g. (N, 6, 768)
        -> Per-order projection: (N, 6, 768) -> (N, 6, proj_dim) e.g. (N, 6, 64)
        -> Flatten: (N, 6*proj_dim) e.g. (N, 384)
        -> MLP: (N, 384) -> (N, n_classes)
    """

    def __init__(
        self,
        n_orders: int = 6,
        embed_dim: int = 768,
        proj_dim: int = 64,
        n_classes: int = 6,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_orders = n_orders
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        # Shared projection layer (applied to each order), no bias
        self.order_proj = nn.Linear(embed_dim, proj_dim, bias=False)

        # MLP after flattening
        flat_dim = n_orders * proj_dim
        layers = []
        in_dim = flat_dim

        if hidden_dims:
            for h_dim in hidden_dims:
                layers.extend(
                    [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
                )
                in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_classes))
        layers.append(nn.Softmax(dim=-1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_orders, embed_dim)

        Returns:
            (batch, n_classes) probability distribution
        """
        # Project each order: (N, 6, 768) -> (N, 6, 64)
        x = self.order_proj(x)
        x = torch.relu(x)

        # Flatten: (N, 6, 64) -> (N, 384)
        x = x.view(x.size(0), -1)

        # MLP to output
        return self.mlp(x)


# =============================================================================
# Embedding Loading
# =============================================================================


def load_6order_embeddings(
    safetensor_path: str | Path,
    n_orders: int = 6,
    embed_dim: int = 768,
) -> torch.Tensor | None:
    """Load 6-order mean embeddings from a safetensor file.

    Args:
        safetensor_path: Path to safetensor file with {order}/mean_token_embeddings keys
        n_orders: Number of orders to load (default 6)
        embed_dim: Embedding dimension (default 768, used for zero-filling missing orders)

    Returns:
        Tensor of shape (n_orders, embed_dim), e.g. (6, 768)
        Missing orders are filled with zeros.
        Returns None if file has no order-prefixed keys (not a 6-order file).
    """
    embeddings = []
    found_any = False
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for order in range(1, n_orders + 1):
            key = f"{order}/mean_token_embeddings"
            if key in keys:
                embeddings.append(f.get_tensor(key))
                found_any = True
            else:
                # Fill missing orders with zeros
                embeddings.append(torch.zeros(embed_dim))

    if not found_any:
        return None

    return torch.stack(embeddings, dim=0)  # (n_orders, embed_dim)


# =============================================================================
# E1 Prediction (inline, for containerized use)
# =============================================================================


def run_e1_orders_prediction(
    input_path: str,
    output_dir: str,
    model_name: str = "150m",
    max_orders: int = 6,
    max_batch_tokens: int = 65536,
) -> dict[str, Path]:
    """Run E1 predictions for multiple oligomeric orders.

    Args:
        input_path: Path to input FASTA file
        output_dir: Directory to save output safetensors
        model_name: E1 model name (150m, 300m, 600m)
        max_orders: Maximum oligomeric order to predict
        max_batch_tokens: Maximum tokens per batch

    Returns:
        Dict mapping sequence_id to output safetensor path
    """
    from E1 import dist
    from E1.io import read_fasta_sequences
    from E1.modeling import E1ForMaskedLM
    from E1.predictor import E1Predictor

    # Resolve model name
    shortcuts = {
        "150m": "Profluent-Bio/E1-150m",
        "300m": "Profluent-Bio/E1-300m",
        "600m": "Profluent-Bio/E1-600m",
    }
    full_model_name = shortcuts.get(model_name, model_name)

    os.makedirs(output_dir, exist_ok=True)

    # Read sequences
    input_sequences = read_fasta_sequences(input_path)
    logger.info(f"Loaded {len(input_sequences)} sequences from {input_path}")

    # Check which sequences need processing
    sequences_to_process = {}
    output_paths = {}
    for seq_id, seq in input_sequences.items():
        output_path = Path(output_dir) / f"{seq_id}_orders.safetensors"
        output_paths[seq_id] = output_path
        if not output_path.exists():
            sequences_to_process[seq_id] = seq

    if not sequences_to_process:
        logger.info("All sequences already processed")
        return output_paths

    # Load model
    logger.info(f"Loading E1 model: {full_model_name}")
    dist.setup_dist()
    model = E1ForMaskedLM.from_pretrained(
        full_model_name, device_map=dist.get_device(), dtype=torch.float
    )
    model.eval()

    # Process each sequence
    for seq_id, sequence in sequences_to_process.items():
        logger.info(f"Processing {seq_id} (length {len(sequence)})")

        tensors = {}
        successful_orders = []

        for order in range(1, max_orders + 1):
            # Create multi-chain sequence
            repeated_seq = ",".join([sequence] * order)
            token_count = len(repeated_seq) + 2

            if token_count > max_batch_tokens:
                logger.warning(f"Order {order} exceeds max tokens, stopping")
                break

            try:
                predictor = E1Predictor(
                    model,
                    max_batch_tokens=max_batch_tokens,
                    save_masked_positions_only=False,
                    fields_to_save=["mean_token_embeddings"],
                )

                predictions = list(
                    predictor.predict(
                        sequences=[repeated_seq],
                        sequence_ids=[f"{seq_id}_order{order}"],
                        context_seqs=None,
                    )
                )

                if predictions:
                    pred = predictions[0]
                    if "mean_token_embeddings" in pred:
                        tensors[f"{order}/mean_token_embeddings"] = pred[
                            "mean_token_embeddings"
                        ]
                        successful_orders.append(order)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM at order {order}, stopping")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                raise

        if tensors:
            metadata = {
                "id": seq_id,
                "orders": ",".join(str(o) for o in successful_orders),
                "base_sequence_length": str(len(sequence)),
            }
            save_file(tensors, str(output_paths[seq_id]), metadata=metadata)
            logger.info(f"Saved embeddings for orders {successful_orders}")

    dist.destroy_process_group()
    return output_paths


# =============================================================================
# Stoichiometry Prediction
# =============================================================================


def load_classifier(weights_path: str | Path) -> OrderEmbeddingMLP:
    """Load the trained stoichiometry classifier.

    Args:
        weights_path: Path to the saved model weights (.pt file)

    Returns:
        Loaded OrderEmbeddingMLP model in eval mode
    """
    model = OrderEmbeddingMLP(
        n_orders=6,
        embed_dim=768,
        proj_dim=64,
        n_classes=6,
        hidden_dims=None,  # No hidden layers, direct projection
        dropout=0.3,
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def predict_stoichiometry(
    embedding_paths: dict[str, Path],
    weights_path: str | Path,
) -> dict[str, dict]:
    """Predict stoichiometry from E1 6-order embeddings.

    Args:
        embedding_paths: Dict mapping sequence_id to safetensor path
        weights_path: Path to classifier weights

    Returns:
        Dict mapping sequence_id to prediction dict with:
        - predicted_copies: Most likely copy number
        - probabilities: Dict of class -> probability
    """
    # Load classifier
    classifier = load_classifier(weights_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)

    results = {}

    for seq_id, embedding_path in embedding_paths.items():
        if not embedding_path.exists():
            logger.warning(f"Embedding file not found for {seq_id}: {embedding_path}")
            continue

        # Load embeddings
        embeddings = load_6order_embeddings(embedding_path)
        if embeddings is None:
            logger.warning(f"Could not load embeddings for {seq_id}")
            continue

        # Predict
        with torch.no_grad():
            x = embeddings.unsqueeze(0).to(device)  # (1, 6, 768)
            probs = classifier(x).cpu().numpy()[0]  # (6,)

        predicted_class = int(np.argmax(probs))
        predicted_label = STOICHIOMETRY_LABELS[predicted_class]

        results[seq_id] = {
            "predicted_copies": predicted_label,
            "probabilities": {
                label: float(probs[i]) for i, label in enumerate(STOICHIOMETRY_LABELS)
            },
        }

    return results


def write_predictions_csv(results: dict[str, dict], output_path: str | Path) -> None:
    """Write predictions to CSV file.

    Args:
        results: Dict from predict_stoichiometry
        output_path: Output CSV path
    """
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        header = ["sequence_id", "predicted_copies"] + [
            f"prob_{label}" for label in STOICHIOMETRY_LABELS
        ]
        writer.writerow(header)

        # Data
        for seq_id, pred in results.items():
            row = [
                seq_id,
                pred["predicted_copies"],
            ] + [pred["probabilities"][label] for label in STOICHIOMETRY_LABELS]
            writer.writerow(row)


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option("--input-path", type=str, required=True, help="Path to input FASTA file")
@click.option("--output-path", type=str, required=True, help="Path to output CSV file")
@click.option(
    "--weights-path",
    type=str,
    required=True,
    help="Path to the trained classifier weights (.pt). No default is shipped.",
)
@click.option(
    "--e1-model",
    type=str,
    default=DEFAULT_E1_MODEL,
    help="E1 model name (150m, 300m, 600m) or HuggingFace path",
)
@click.option(
    "--embeddings-dir",
    type=str,
    default=None,
    help="Directory for E1 embeddings (default: output_path parent / embeddings)",
)
@click.option(
    "--max-batch-tokens",
    type=int,
    default=65536,
    help="Maximum tokens per E1 batch",
)
@click.option(
    "--skip-e1/--run-e1",
    default=False,
    help="Skip E1 inference (use existing embeddings)",
)
def predict(
    input_path: str,
    output_path: str,
    weights_path: str,
    e1_model: str,
    embeddings_dir: Optional[str],
    max_batch_tokens: int,
    skip_e1: bool,
) -> None:
    """Predict protein stoichiometry from FASTA sequences.

    This tool uses E1 embeddings at multiple oligomeric orders and a trained
    classifier to predict the most likely stoichiometry (1, 2, 3, 4, 5, or 6+
    copies) for each input sequence.

    Example:
        python predict_stoichiometry.py --input-path seqs.fasta --output-path preds.csv \
            --weights-path /path/to/stoichiometry_classifier.pt
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up embeddings directory
    if embeddings_dir is None:
        embeddings_dir = output_path.parent / "embeddings"
    else:
        embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Check classifier weights exist
    if not Path(weights_path).exists():
        raise click.ClickException(
            f"Classifier weights not found at {weights_path}. "
            "Pass the path to the trained classifier checkpoint with --weights-path."
        )

    # Run E1 predictions
    if not skip_e1:
        logger.info("Running E1 6-order embedding prediction...")
        embedding_paths = run_e1_orders_prediction(
            input_path=input_path,
            output_dir=str(embeddings_dir),
            model_name=e1_model,
            max_orders=6,
            max_batch_tokens=max_batch_tokens,
        )
    else:
        # Find existing embeddings
        from E1.io import read_fasta_sequences

        input_sequences = read_fasta_sequences(input_path)
        embedding_paths = {
            seq_id: embeddings_dir / f"{seq_id}_orders.safetensors"
            for seq_id in input_sequences.keys()
        }

    # Predict stoichiometry
    logger.info("Running stoichiometry prediction...")
    results = predict_stoichiometry(embedding_paths, weights_path)

    # Write output
    write_predictions_csv(results, output_path)
    logger.info(f"Wrote predictions to {output_path}")

    # Print summary
    print(f"\nPredicted stoichiometry for {len(results)} sequences:")
    for seq_id, pred in results.items():
        probs_str = ", ".join(f"{k}:{v:.2f}" for k, v in pred["probabilities"].items())
        print(f"  {seq_id}: {pred['predicted_copies']} copies ({probs_str})")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    predict()
