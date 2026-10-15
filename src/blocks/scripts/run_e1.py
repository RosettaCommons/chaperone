#!/usr/bin/env python
"""Fixed E1 predict script that handles None context_id.

This is a patched version of E1.tools.predict that fixes a bug where
context_id=None causes safetensors to fail when saving metadata.

The original bug: https://github.com/Profluent-AI/E1/issues/XXX
When no --context-path is provided, context_id is None, and the line:
    save_file(..., metadata={"id": _id, "context_id": context_id})
fails because safetensors cannot convert None to a string.

This script fixes it by only including context_id in metadata when it's not None.

Additional feature: --predict-orders N
Predicts embeddings for multiple oligomeric orders (1 to N chains of the sequence).
Chains are comma-separated (e.g., "ABC,ABC" for a dimer). Each FASTA entry is
processed independently. Saves embeddings indexed by order in safetensors.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import torch
from safetensors.torch import save_file

from E1 import dist
from E1.io import read_fasta_sequences
from E1.modeling import E1ForMaskedLM
from E1.predictor import E1Prediction, E1Predictor

logger = logging.getLogger(__name__)


def get_output_filename(sequence_id: str, context_id: str | None = None) -> str:
    """Get the output filename for a prediction."""
    return (
        f"{context_id}+{sequence_id}.safetensors"
        if context_id
        else f"{sequence_id}.safetensors"
    )


def save_prediction(prediction: E1Prediction, output_dir: str) -> None:
    """Save prediction to safetensors file with proper metadata handling."""
    _id = prediction["id"]
    context_id = prediction["context_id"]

    tensors = {}
    if "logits" in prediction:
        tensors["logits"] = prediction["logits"]
    if "token_embeddings" in prediction:
        tensors["token_embeddings"] = prediction["token_embeddings"]
    if "mean_token_embeddings" in prediction:
        tensors["mean_token_embeddings"] = prediction["mean_token_embeddings"]

    filename = get_output_filename(_id, context_id)

    # Fix: Only include context_id in metadata if it's not None
    # safetensors requires all metadata values to be strings
    metadata = {"id": _id}
    if context_id is not None:
        metadata["context_id"] = context_id

    save_file(tensors, os.path.join(output_dir, filename), metadata=metadata)


def get_orders_output_filename(sequence_id: str) -> str:
    """Get the output filename for orders mode predictions."""
    return f"{sequence_id}_orders.safetensors"


def _predict_orders_impl(
    model: E1ForMaskedLM,
    sequence: str,
    sequence_id: str,
    max_orders: int,
    max_batch_tokens: int,
    fields_to_save: list[str],
    output_dir: str,
    skip_existing: bool = True,
) -> None:
    """Predict embeddings for multiple oligomeric orders of a single sequence.

    For each order k (1 to max_orders), creates a multi-chain sequence with k
    copies of the input sequence separated by commas (e.g., "ABC,ABC,ABC" for
    order 3). Saves all successful predictions to a single safetensors file
    with keys indexed by order number.

    Args:
        model: The E1 model to use for predictions.
        sequence: The input protein sequence.
        sequence_id: Identifier for the sequence.
        max_orders: Maximum number of repeats to predict (1 to max_orders).
        max_batch_tokens: Maximum tokens allowed per batch.
        fields_to_save: Which embedding fields to save.
        output_dir: Directory to save the output file.
        skip_existing: If True, skip prediction if output file already exists.
    """
    # Check if output already exists
    filename = get_orders_output_filename(sequence_id)
    output_path = os.path.join(output_dir, filename)
    if skip_existing and os.path.exists(output_path):
        logger.info(f"Skipping '{sequence_id}': output already exists at {output_path}")
        return

    tensors: dict[str, torch.Tensor] = {}
    successful_orders: list[int] = []

    for order in range(1, max_orders + 1):
        # Join chains with comma separator (e.g., "ABC,ABC,ABC" for order 3)
        repeated_seq = ",".join([sequence] * order)
        seq_length = len(repeated_seq)

        # Check if sequence fits within token budget (add buffer for special tokens)
        # E1 adds BOS/EOS tokens, so actual token count is seq_length + 2
        token_count = seq_length + 2
        if token_count > max_batch_tokens:
            logger.warning(
                f"Order {order} sequence length ({seq_length}) exceeds max_batch_tokens "
                f"({max_batch_tokens}). Skipping orders {order} to {max_orders}."
            )
            break

        logger.info(f"Predicting order {order} (sequence length: {seq_length})")

        try:
            # Create predictor with conservative batch size for this order
            predictor = E1Predictor(
                model,
                max_batch_tokens=max_batch_tokens,
                save_masked_positions_only=False,
                fields_to_save=fields_to_save,
            )

            # Predict for this order
            predictions = list(
                predictor.predict(
                    sequences=[repeated_seq],
                    sequence_ids=[f"{sequence_id}_order{order}"],
                    context_seqs=None,
                )
            )

            if not predictions:
                logger.warning(f"No prediction returned for order {order}. Skipping.")
                continue

            prediction = predictions[0]
            order_key = str(order)

            # Store tensors with order-prefixed keys
            if "logits" in prediction and "logits" in fields_to_save:
                tensors[f"{order_key}/logits"] = prediction["logits"]
            if (
                "token_embeddings" in prediction
                and "token_embeddings" in fields_to_save
            ):
                tensors[f"{order_key}/token_embeddings"] = prediction[
                    "token_embeddings"
                ]
            if (
                "mean_token_embeddings" in prediction
                and "mean_token_embeddings" in fields_to_save
            ):
                tensors[f"{order_key}/mean_token_embeddings"] = prediction[
                    "mean_token_embeddings"
                ]

            successful_orders.append(order)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"Out of memory at order {order}. Stopping prediction of higher orders."
                )
                # Clear CUDA cache and break
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            raise

    if not tensors:
        logger.error(
            "No embeddings could be computed for any order. Output file not created."
        )
        return

    # Save all orders to a single file
    filename = f"{sequence_id}_orders.safetensors"
    metadata = {
        "id": sequence_id,
        "orders": ",".join(str(o) for o in successful_orders),
        "base_sequence_length": str(len(sequence)),
    }

    save_file(tensors, os.path.join(output_dir, filename), metadata=metadata)
    logger.info(f"Saved embeddings for orders {successful_orders} to {filename}")


def run_standard_prediction(
    model: E1ForMaskedLM,
    input_path: str,
    max_batch_tokens: int,
    save_masked_positions_only: bool,
    fields_to_save: list[str],
    context_path: str | None,
    output_dir: str,
    skip_existing: bool = True,
) -> None:
    """Run standard E1 predictions on multiple sequences."""
    context_seqs: dict[str, str] | None = None
    if context_path is not None:
        context_seqs = read_fasta_sequences(context_path)

    input_sequences = read_fasta_sequences(input_path)

    # Filter out sequences that already have outputs
    if skip_existing:
        filtered_sequences = {}
        skipped_count = 0
        for seq_id, seq in input_sequences.items():
            # For context mode, we'd need to check all context combinations
            # For simplicity, check if any output for this sequence exists
            filename = get_output_filename(seq_id, context_id=None)
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                logger.info(
                    f"Skipping '{seq_id}': output already exists at {output_path}"
                )
                skipped_count += 1
            else:
                filtered_sequences[seq_id] = seq

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} sequences with existing outputs")

        if not filtered_sequences:
            logger.info("All sequences already processed. Nothing to do.")
            return

        input_sequences = filtered_sequences

    sequence_ids, sequences = zip(*input_sequences.items())

    predictor = E1Predictor(
        model,
        max_batch_tokens=max_batch_tokens,
        save_masked_positions_only=save_masked_positions_only,
        fields_to_save=fields_to_save,
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for prediction in predictor.predict(
            sequences=list(sequences),
            sequence_ids=list(sequence_ids),
            context_seqs=context_seqs,
        ):
            futures.append(executor.submit(save_prediction, prediction, output_dir))

        for future in as_completed(futures):
            future.result()


@click.command()
@click.option("--model-name", type=str, required=True, help="Name of the model to use")
@click.option(
    "--input-path", type=str, required=True, help="Path to the input fasta file"
)
@click.option(
    "--output-dir",
    type=str,
    default="predictions",
    help="Directory to save the predictions",
)
@click.option("--context-path", type=str, help="Path to the context fasta file")
@click.option(
    "--max-batch-tokens",
    type=int,
    default=65536,
    help="Maximum number of tokens to batch in a single forward pass",
)
@click.option(
    "--save-masked-positions-only",
    type=bool,
    default=False,
    help="Whether to save logits/embeddings for only the masked positions in each sequence",
)
@click.option(
    "--fields-to-save",
    type=str,
    multiple=True,
    default=["logits", "token_embeddings", "mean_token_embeddings"],
    help="Fields to save",
)
@click.option(
    "--predict-orders",
    type=int,
    default=None,
    help=(
        "Predict embeddings for oligomeric orders 1 to N (e.g., monomer, dimer, trimer). "
        "Chains are comma-separated. Each FASTA entry is processed independently. "
        "Incompatible with --context-path."
    ),
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip sequences that already have output files (default: True)",
)
def predict(
    model_name: str,
    input_path: str,
    max_batch_tokens: int,
    save_masked_positions_only: bool,
    fields_to_save: list[str],
    context_path: str | None,
    output_dir: str,
    predict_orders: int | None,
    skip_existing: bool,
) -> None:
    """Run E1 masked language model predictions on protein sequences.

    Standard mode: Predicts embeddings for all sequences in the input FASTA.

    Orders mode (--predict-orders N): For each sequence in the input FASTA,
    predicts embeddings for 1 to N chains (comma-separated, e.g., "ABC,ABC,ABC"
    for order 3). Each sequence produces a separate output file with keys
    indexed by order number (1, 2, 3, ...).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Early exit check: see if all outputs already exist before loading model
    if skip_existing:
        input_sequences = read_fasta_sequences(input_path)
        sequences_to_process = []

        if predict_orders is not None:
            # Orders mode: check for _orders.safetensors files
            for seq_id in input_sequences.keys():
                filename = get_orders_output_filename(seq_id)
                if not os.path.exists(os.path.join(output_dir, filename)):
                    sequences_to_process.append(seq_id)
        else:
            # Standard mode: check for individual safetensors files
            for seq_id in input_sequences.keys():
                filename = get_output_filename(seq_id, context_id=None)
                if not os.path.exists(os.path.join(output_dir, filename)):
                    sequences_to_process.append(seq_id)

        if not sequences_to_process:
            logger.info(
                "All outputs already exist. Skipping model loading and exiting early."
            )
            return

        logger.info(
            f"Will process {len(sequences_to_process)}/{len(input_sequences)} sequences "
            f"({len(input_sequences) - len(sequences_to_process)} already exist)"
        )

    # Model is loaded in fp32, the actual predictions is made under torch.autocast(..., torch.bfloat16)
    # context manager
    model = E1ForMaskedLM.from_pretrained(
        model_name, device_map=dist.get_device(), dtype=torch.float
    )
    model.eval()

    if predict_orders is not None:
        # Orders mode: predict embeddings for multiple oligomeric states
        if context_path is not None:
            raise click.UsageError(
                "--predict-orders is incompatible with --context-path"
            )
        if predict_orders < 1:
            raise click.UsageError("--predict-orders must be at least 1")

        input_sequences = read_fasta_sequences(input_path)
        logger.info(
            f"Running orders mode: predicting orders 1 to {predict_orders} "
            f"for {len(input_sequences)} sequence(s)"
        )

        for sequence_id, sequence in input_sequences.items():
            logger.info(f"Processing sequence '{sequence_id}' (length {len(sequence)})")
            _predict_orders_impl(
                model=model,
                sequence=sequence,
                sequence_id=sequence_id,
                max_orders=predict_orders,
                max_batch_tokens=max_batch_tokens,
                fields_to_save=list(fields_to_save),
                output_dir=output_dir,
                skip_existing=skip_existing,
            )
    else:
        # Standard mode
        run_standard_prediction(
            model=model,
            input_path=input_path,
            max_batch_tokens=max_batch_tokens,
            save_masked_positions_only=save_masked_positions_only,
            fields_to_save=list(fields_to_save),
            context_path=context_path,
            output_dir=output_dir,
            skip_existing=skip_existing,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dist.setup_dist()
    try:
        predict()
    finally:
        dist.destroy_process_group()
