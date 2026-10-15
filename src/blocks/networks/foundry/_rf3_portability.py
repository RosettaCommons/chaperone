"""Disable inherited institutional residue caches in pinned RF3 inference.

Training defaults are untouched. Inference must use the same feature path on
public machines and institutional machines, regardless of mounted caches.
This helper runs standalone when building either Foundry image.
"""

from hashlib import sha256
from pathlib import Path
import sys


_EXPECTED_SHA256 = "7ae6afa48835b6847e9bec9b426bc6dcb3a17a7bec491fad9e5c0278ef789352"
_OLD = '''            transform_overrides={
                "diffusion_batch_size": diffusion_batch_size,
'''
_NEW = '''            transform_overrides={
                # Chaperone: inference never inherits a site-specific residue cache.
                "residue_cache_dir": None,
                "diffusion_batch_size": diffusion_batch_size,
'''


def patch(source: Path) -> None:
    """Apply only to the reviewed upstream source; accept the exact patch again."""
    content = source.read_text()
    original = content.replace(_NEW, _OLD) if content.count(_NEW) == 1 else content
    if (
        sha256(original.encode()).hexdigest() != _EXPECTED_SHA256
        or original.count(_OLD) != 1
    ):
        raise RuntimeError(f"Unexpected pinned Foundry RF3 source: {source}")
    source.write_text(original.replace(_OLD, _NEW))


def check_runtime() -> None:
    """Prove that the real engine overrides a checkpoint-provided cache path."""
    from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData
    from omegaconf import OmegaConf
    from rf3.inference_engines.rf3 import RF3InferenceEngine

    engine = RF3InferenceEngine(ckpt_path="/tmp/unused-rf3-checkpoint.ckpt")
    assert engine.transform_overrides["residue_cache_dir"] is None
    cfg = OmegaConf.create({
        "datasets": {"val": {"probe": {"dataset": {"transform": {
            "_target_": "rf3.data.pipelines.build_af3_transform_pipeline",
            "is_inference": True,
            "protein_msa_dirs": [],
            "rna_msa_dirs": [],
            "residue_cache_dir": "/tmp/legacy-training-residue-cache",
        }}}}},
    })
    engine._construct_pipeline(cfg)
    loaders = [
        transform for transform in engine.pipeline.transforms
        if isinstance(transform, LoadCachedResidueLevelData)
    ]
    assert len(loaders) == 1, loaders
    assert loaders[0].dir is None, loaders[0].dir
    print("RF3 inference overrides checkpoint residue cache: None")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
    if "--check" in sys.argv[2:]:
        check_runtime()
