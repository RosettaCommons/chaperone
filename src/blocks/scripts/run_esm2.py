#!/usr/bin/env python
"""ESM-2 inference script for protein sequence embeddings.

This script runs ESM-2 (Evolutionary Scale Modeling 2) on protein sequences
to generate per-residue and sequence-level embeddings. It uses HuggingFace
transformers for model loading and inference.

Model sizes available:
- 8m:   facebook/esm2_t6_8M_UR50D    (320 hidden, 6 layers)
- 35m:  facebook/esm2_t12_35M_UR50D  (480 hidden, 12 layers)
- 150m: facebook/esm2_t30_150M_UR50D (640 hidden, 30 layers)
- 650m: facebook/esm2_t33_650M_UR50D (1280 hidden, 33 layers)
- 3b:   facebook/esm2_t36_3B_UR50D   (2560 hidden, 36 layers)
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import click
import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# Model shortcuts to full HuggingFace paths
MODEL_SHORTCUTS = {
    "8m": "facebook/esm2_t6_8M_UR50D",
    "35m": "facebook/esm2_t12_35M_UR50D",
    "150m": "facebook/esm2_t30_150M_UR50D",
    "650m": "facebook/esm2_t33_650M_UR50D",
    "3b": "facebook/esm2_t36_3B_UR50D",
}


def parse_fasta(path: str) -> dict[str, str]:
    """Parse FASTA file into id->sequence dict.

    Args:
        path: Path to FASTA file

    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    sequences = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            elif current_id is not None:
                current_seq.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    return sequences


def batch_sequences(
    sequences: dict[str, str],
    max_batch_tokens: int,
) -> Iterator[list[tuple[str, str]]]:
    """Batch sequences by token count for memory-efficient processing.

    Args:
        sequences: Dict mapping sequence IDs to sequences
        max_batch_tokens: Maximum tokens per batch

    Yields:
        Lists of (seq_id, sequence) tuples forming a batch
    """
    current_batch = []
    current_tokens = 0

    for seq_id, seq in sequences.items():
        # Estimate tokens (sequence length + special tokens)
        seq_tokens = len(seq) + 2  # +2 for BOS and EOS tokens

        if current_batch and current_tokens + seq_tokens > max_batch_tokens:
            yield current_batch
            current_batch = []
            current_tokens = 0

        current_batch.append((seq_id, seq))
        current_tokens += seq_tokens

    if current_batch:
        yield current_batch


def get_output_filename(seq_id: str) -> str:
    """Get the output filename for a sequence."""
    return f"{seq_id}.safetensors"


def save_prediction(
    tensors: dict[str, torch.Tensor],
    seq_id: str,
    output_dir: str,
) -> None:
    """Save prediction to safetensors file.

    Args:
        tensors: Dict of tensor name to tensor
        seq_id: Sequence identifier
        output_dir: Output directory path
    """
    filename = get_output_filename(seq_id)
    metadata = {"sequence_id": seq_id}
    save_file(tensors, os.path.join(output_dir, filename), metadata=metadata)


@click.command()
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="Model size (8m, 35m, 150m, 650m, 3b) or full HuggingFace path",
)
@click.option(
    "--input-path",
    type=str,
    required=True,
    help="Path to input FASTA file",
)
@click.option(
    "--output-dir",
    type=str,
    default="predictions",
    help="Directory to save output safetensors files",
)
@click.option(
    "--max-batch-tokens",
    type=int,
    default=4096,
    help="Maximum number of tokens per batch for memory efficiency",
)
@click.option(
    "--fields-to-save",
    type=str,
    multiple=True,
    default=["token_embeddings", "mean_token_embeddings"],
    help="Fields to save (token_embeddings, mean_token_embeddings)",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    help="Skip sequences that already have output files (default: True)",
)
def predict(
    model_name: str,
    input_path: str,
    output_dir: str,
    max_batch_tokens: int,
    fields_to_save: tuple[str, ...],
    skip_existing: bool,
) -> None:
    """Run ESM-2 inference on protein sequences.

    Generates per-residue (token) embeddings and optionally mean-pooled
    sequence embeddings. Outputs are saved as safetensors files.
    """
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    sequences = parse_fasta(input_path)
    logger.info(f"Loaded {len(sequences)} sequences from {input_path}")

    # Early exit check: see if all outputs already exist before loading model
    if skip_existing:
        sequences_to_process = {}
        for seq_id, seq in sequences.items():
            filename = get_output_filename(seq_id)
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                sequences_to_process[seq_id] = seq

        if not sequences_to_process:
            logger.info(
                "All outputs already exist. Skipping model loading and exiting early."
            )
            return

        skipped = len(sequences) - len(sequences_to_process)
        if skipped > 0:
            logger.info(
                f"Will process {len(sequences_to_process)}/{len(sequences)} sequences "
                f"({skipped} already exist)"
            )
        sequences = sequences_to_process

    model_path = MODEL_SHORTCUTS.get(model_name, model_name)
    logger.info(f"Loading model: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    fields_to_save_set = set(fields_to_save)

    total_processed = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []

        for batch in batch_sequences(sequences, max_batch_tokens):
            seq_ids = [s[0] for s in batch]
            seqs = [s[1] for s in batch]

            inputs = tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with (
                torch.no_grad(),
                torch.autocast(device_type=device, dtype=torch.bfloat16),
            ):
                outputs = model(**inputs)

            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            for i, seq_id in enumerate(seq_ids):
                tensors = {}
                seq_len = attention_mask[i].sum().item()

                # ESM-2 wraps residues in <cls> and <eos> tokens.
                if "token_embeddings" in fields_to_save_set:
                    token_emb = last_hidden_state[i, 1 : seq_len - 1, :].float().cpu()
                    tensors["token_embeddings"] = token_emb

                if "mean_token_embeddings" in fields_to_save_set:
                    residue_embeddings = last_hidden_state[i, 1 : seq_len - 1, :]
                    mean_emb = residue_embeddings.mean(dim=0).float().cpu()
                    tensors["mean_token_embeddings"] = mean_emb

                if tensors:
                    futures.append(
                        executor.submit(save_prediction, tensors, seq_id, output_dir)
                    )

            total_processed += len(batch)
            logger.info(f"Processed {total_processed}/{len(sequences)} sequences")

        for future in as_completed(futures):
            future.result()

    logger.info(f"Saved {total_processed} predictions to {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    predict()
