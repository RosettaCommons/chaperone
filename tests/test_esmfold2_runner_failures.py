"""A failed prediction must fail the chunk even when another prediction succeeds."""

import csv
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_validation_revision_invalidates_old_partial_cache():
    from pydantic import ValidationError

    from blocks.networks.prediction.esmfold2 import ESMFold2Block
    from chaperone import ExecutionEngine
    from chaperone_cli.registry import _extract_field_type

    block = ESMFold2Block()
    config = block.model_dump(mode="json")
    assert config.pop("output_validation_version") == 2
    assert _extract_field_type(type(block).model_fields["output_validation_version"]) is None
    old_payload = [{
        "class": f"{block.__class__.__module__}.{block.__class__.__qualname__}",
        "config": config,
    }]
    old_hash = hashlib.sha256(
        json.dumps(old_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    assert ExecutionEngine._cache_fingerprint([block]) != old_hash
    with pytest.raises(ValidationError):
        ESMFold2Block(output_validation_version=1)
    with pytest.raises(ValidationError):
        block.output_validation_version = 1


@pytest.mark.parametrize("failure", [None, "inference", "malformed"])
def test_runner_exit_status_covers_every_input(monkeypatch, tmp_path, capsys, failure):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    outdir = tmp_path / "outputs"
    for name in ("first", "second"):
        spec = {"name": name, "chains": [{"id": "A", "sequence": "MAAA"}]}
        (inputs / f"{name}.json").write_text(json.dumps(spec))
    if failure == "malformed":
        (inputs / "second.json").write_text("{")

    calls = []

    class Model:
        @classmethod
        def from_pretrained(cls, model):
            return cls()

        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            return self

    class Builder:
        def fold(self, model, prediction_input, **kwargs):
            name = kwargs["complex_id"]
            calls.append(name)
            if failure == "inference" and name == "second":
                raise RuntimeError("mock prediction failed")
            return SimpleNamespace(
                plddt=SimpleNamespace(mean=lambda: 0.8),
                ptm=0.7,
                iptm=None,
                complex=SimpleNamespace(to_mmcif=lambda: f"data_{name}\n"),
            )

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    esm = ModuleType("esm")
    models = ModuleType("esm.models")
    esmfold2 = ModuleType("esm.models.esmfold2")
    esmfold2.EsmFold2Model = Model
    esmfold2.ESMFold2InputBuilder = Builder
    for name in ("LigandInput", "ProteinInput", "StructurePredictionInput"):
        setattr(esmfold2, name, SimpleNamespace)
    esm.models = models
    models.esmfold2 = esmfold2
    for module in (torch, esm, models, esmfold2):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in (
        "XDG_CACHE_HOME", "CUEQ_TRITON_CACHE_DIR", "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR", "ESMCFOLD_CCD_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hf"))
    runner = Path(__file__).parents[1] / "src/blocks/scripts/run_esmfold2_runner.py"
    monkeypatch.setattr(sys, "argv", [str(runner), "--inputs", str(inputs), "--outdir", str(outdir)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(runner), run_name="__main__")

    assert exited.value.code == (1 if failure else 0)
    assert calls == (["first"] if failure == "malformed" else ["first", "second"])
    assert (outdir / "first.cif").read_text() == "data_first\n"
    assert (outdir / "second.cif").exists() is (failure is None)
    with (outdir / "summary.csv").open() as summary:
        rows = list(csv.DictReader(summary))
    assert [row["name"] for row in rows] == ["first", "second"]
    assert rows[0]["length"] == "4"
    assert rows[1]["length"] == ("-1" if failure else "4")
    if failure:
        assert rows[1]["mean_plddt"] == "nan"
        assert "1/2 predictions failed" in capsys.readouterr().err
