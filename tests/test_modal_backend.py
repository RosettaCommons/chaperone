"""Offline regressions and opt-in live Modal lifecycle checks."""

import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from blocks.util_blocks.testing import CopyBlock
from chaperone import ExecutionEngine, JobArgs, PipelineDataHandler, get_settings
from chaperone.block_config.manifest import Manifest
from chaperone.containers import Image
from chaperone.engine.resolve import (
    RuntimeArgs,
    _resolve_default_mode,
    _resolve_job_args,
)
from chaperone.execution_tracking import (
    RemoteAllocation,
    load_execution_records,
    track_execution,
)
from chaperone.orchestration import modal_backend as backend


def test_dockerfile_preserves_multiline_program(tmp_path):
    command = 'python3 -c \'from pathlib import Path\nPath("result.txt").write_text("line1\\nline2")\''
    image = Image("multiline").base_image("ubuntu:24.04").run(command)
    instruction = next(
        line for line in image.to_dockerfile().splitlines() if line.startswith("RUN [")
    )
    argv = json.loads(instruction.removeprefix("RUN "))
    subprocess.run(argv, cwd=tmp_path, check=True)
    assert (tmp_path / "result.txt").read_text() == "line1\nline2"


def test_modal_gpu_mapping_preserves_count():
    settings = SimpleNamespace(
        CHAPERONE_MODAL_GPU="A10G", CHAPERONE_MODAL_GPU_MAP={"large": "A100-80GB"}
    )
    assert backend.gpu_request(settings, JobArgs()) is None
    assert backend.gpu_request(settings, JobArgs(gres="gpu:h200:2")) == "A10G:2"
    assert (
        backend.gpu_request(settings, JobArgs(gpu_class="large", gres="gpu:1"))
        == "A100-80GB"
    )


def test_cloud_preserves_runscript_and_argument_quoting():
    image = (
        Image("entrypoint").base_image("ubuntu:24.04").runscript('printf "%s\\n" "$@"')
    )
    command = backend.CommandContext().build_command(
        image, ["literal; argument"], {}, []
    )
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert result.stdout == "literal; argument\n"


def test_image_context_contains_only_declared_sources(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("print('hello')")
    (tmp_path / ".env").write_text("must not be uploaded")
    image = Image("copy").base_image("ubuntu:24.04")
    image._copies = [(str(script), "/opt/script.py")]
    directory = tmp_path / "build"
    directory.mkdir()
    dockerfile = backend.write_image_context(image, directory)
    assert set(p.name for p in directory.iterdir()) == {"Dockerfile", "copy-0"}
    assert (directory / "copy-0").read_text() == script.read_text()
    assert "COPY copy-0 /opt/script.py" in dockerfile.read_text()
    assert image._copies == [(str(script), "/opt/script.py")]


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "outside" / "design.pdb"
    path.parent.mkdir()
    path.write_text(
        "ATOM      1  CA  GLY A   1       1.000   2.000   3.000  1.00 10.00           C\nEND\n"
    )
    return PipelineDataHandler.from_paths([path], name="input")


@pytest.mark.parametrize("name", ["modal", "cloud"])
def test_cloud_never_falls_back_to_local_or_slurm(name, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/sbatch")
    monkeypatch.setattr(
        "chaperone.orchestration.gpu_local_taskrunner.detect_cuda_gpus", lambda: [0]
    )
    assert _resolve_default_mode(name) == "modal"
    run = RuntimeArgs(mode=name)
    _resolve_job_args({"gres": "gpu:1"}, CopyBlock(), run)
    assert run.mode == "modal"


def test_staging_copies_and_disambiguates(tmp_path):
    first, second = tmp_path / "a" / "x.pdb", tmp_path / "b" / "x.pdb"
    for path, content in [(first, "first"), (second, "second")]:
        path.parent.mkdir()
        path.write_text(content)
    staged = [
        backend.stage_file(p, tmp_path / "chunk" / "inputs") for p in (first, second)
    ]
    assert staged[0] != staged[1]
    assert [p.read_text() for p in staged] == ["first", "second"]
    assert all(not p.is_symlink() for p in staged)


def test_staging_rejects_external_links_and_paths(tmp_path):
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("data")
    (chunk / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        backend.validate_inputs(chunk, ["true"], {})
    (chunk / "link").unlink()
    with pytest.raises(ValueError, match="Unstaged"):
        backend.validate_inputs(chunk, [f"input={outside}"], {})
    backend.validate_inputs(chunk, [f"input={outside}"], {str(outside): "weights"})


@pytest.mark.parametrize("name,link", [("../escape", False), ("link", True)])
def test_download_rejects_unsafe_archive(tmp_path, name, link):
    archive = tmp_path / "output.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo(name)
        if link:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "/etc/passwd"
        tar.addfile(entry, io.BytesIO())
    with pytest.raises((ValueError, tarfile.FilterError)):
        backend._extract_outputs(archive, tmp_path / "output")
    assert not (tmp_path / "escape").exists()


def test_engine_preserves_compile_lineage_cache_and_failures(
    inputs, tmp_path, monkeypatch
):
    calls = []

    def execute(manifest, context, resources):
        calls.append(manifest)
        # Exercise the same native command, without spending cloud compute.
        Manifest.exec(manifest.cwd).check_returncode()

    monkeypatch.setattr(backend, "execute_modal", execute)
    engine = ExecutionEngine(
        tmp_path / "run", run_args_default={"mode": "modal", "max_retries": 0}
    )
    result = engine.run(CopyBlock(n_multiply=2), data=inputs)
    assert len(result) == 2
    assert len(calls) == 1
    assert all(Path(x.path).is_file() for x in result.get_data())
    assert all(len(row["structures"]) == 2 for row in result._df.to_dicts())
    assert len(engine.run(CopyBlock(n_multiply=2), data=inputs)) == 2
    assert len(calls) == 1

    def fail(*args):
        raise subprocess.CalledProcessError(7, "remote")

    monkeypatch.setattr(backend, "execute_modal", fail)
    with pytest.raises(RuntimeError, match="failing"):
        engine.run(CopyBlock(n_multiply=2), data=inputs, run_args={"force_exec": True})


def test_public_mpnn_prepares_without_local_image(inputs, tmp_path, monkeypatch):
    from blocks.networks import mpnn

    monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", mpnn._PUBLIC_MPNN_CONTAINER)
    monkeypatch.setattr(
        Image, "get_image_path", lambda _: pytest.fail("local SIF lookup")
    )
    block = mpnn.ClassicMPNNBlock(cloud=True, device="cpu")
    manifest, context = backend.prepare_modal(block, inputs, tmp_path / "chunk", 0)
    assert context.image.name == "ligandmpnn_public"
    assert manifest.command[0] == "env"
    assert "CUDA_VISIBLE_DEVICES=" in manifest.command
    assert "apptainer" not in manifest.command
    backend.validate_inputs(Path(manifest.cwd), manifest.command, {})


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("gpu", [False, True])
def test_modal_resources_transfer_and_cleanup(tmp_path, monkeypatch, fail, gpu):
    modal = pytest.importorskip("modal")
    chunk = tmp_path / "chunk"
    chunk.mkdir()
    (chunk / "input.txt").write_text("input")
    sandbox = MagicMock()
    sandbox.exec.return_value = SimpleNamespace(
        stdout=io.StringIO(""),
        stderr=io.StringIO(""),
        wait=lambda: None,
        returncode=7 if fail else 0,
    )

    def download(remote, local):
        with tarfile.open(local, "w") as tar:
            entry = tarfile.TarInfo("result.txt")
            entry.size = 6
            tar.addfile(entry, io.BytesIO(b"result"))

    sandbox.filesystem.copy_to_local.side_effect = download
    create = MagicMock(return_value=sandbox)
    monkeypatch.setattr(modal.Sandbox, "create", create)
    monkeypatch.setattr(modal.App, "lookup", MagicMock())
    image = MagicMock()
    monkeypatch.setattr(modal.Image, "from_dockerfile", image)
    monkeypatch.setattr(get_settings(), "CHAPERONE_MODAL_VOLUMES", {})
    context = backend.CommandContext(Image("test").base_image("ubuntu:24.04"))
    manifest = Manifest("test", ["true"], str(chunk))
    resources = JobArgs(gres="gpu:2" if gpu else None, cpus_per_task=2, mem_gb=4)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            backend.execute_modal(manifest, context, resources)
        sandbox.filesystem.copy_to_local.assert_not_called()
    else:
        backend.execute_modal(manifest, context, resources)
        assert (chunk / "result.txt").read_text() == "result"
    assert create.call_args.kwargs["gpu"] == ("A10G:2" if gpu else None)
    assert create.call_args.kwargs["cpu"] == 2
    assert create.call_args.kwargs["memory"] == 4096
    sandbox.terminate.assert_called_once()
    assert context.allocation.backend == "modal"
    assert context.allocation.allocated_gpus == (2 if gpu else 0)
    assert (context.allocation.gpu_seconds > 0) == gpu


def test_modal_accounting_uses_remote_allocation_not_controller(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "controller-job")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    ticks = iter([0, 10, 14, 20])
    monkeypatch.setattr(
        "chaperone.execution_tracking.time.perf_counter", lambda: next(ticks)
    )
    allocation = RemoteAllocation("modal")
    with track_execution(
        step_dir=tmp_path,
        sandbox_dir=tmp_path / "chunk_0",
        block_name="test",
        source="compiled_block",
        allocation=allocation,
    ):
        allocation.start("sb-test", 2)
        allocation.stop()
    (record,) = load_execution_records(tmp_path)
    assert record.backend == "modal"
    assert record.job_id == "sb-test"
    assert record.allocated_gpus == 2
    assert record.gpu_seconds == 8
    assert record.wall_seconds == 20


class GPUCopy(CopyBlock):
    _name = "modal_gpu_copy"
    _resources = JobArgs(partition="gpu", gres="gpu:1", mem_gb=2, time_limit="00:05:00")

    def create_command(self, handler, sandbox_dir, chunk_id):
        return super().create_command(handler, sandbox_dir, chunk_id) + [
            "&&",
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
            ">",
            str(sandbox_dir / "gpu.txt"),
        ]


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("CHAPERONE_RUN_MODAL_TESTS") != "1", reason="Opt-in paid Modal test"
)
@pytest.mark.parametrize("gpu", [False, True])
def test_live_modal_copy_and_cache(tmp_path, inputs, gpu):
    engine = ExecutionEngine(
        tmp_path / "run", run_args_default={"mode": "modal", "max_retries": 0}
    )
    block = GPUCopy() if gpu else CopyBlock(n_multiply=2)
    output = engine.run(block, data=inputs)
    assert len(output) == (1 if gpu else 2)
    for loaded in output.get_data():
        assert (
            Path(loaded.path).read_text()
            == Path(next(inputs.get_data()).path).read_text()
        )
    if gpu:
        assert next((tmp_path / "run").rglob("gpu.txt")).read_text().strip()
    assert len(engine.run(block, data=inputs)) == len(output)


@pytest.mark.cloud
@pytest.mark.skipif(
    os.environ.get("CHAPERONE_RUN_MODAL_MPNN") != "1",
    reason="Opt-in Modal MPNN image build/inference",
)
@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_live_modal_public_mpnn(tmp_path, monkeypatch, gpu):
    from biotite.structure import Atom, array
    from blocks.networks import mpnn
    from chaperone.data.schema import get_sequences_from_row
    from chaperone.io import write_pdb

    class CheckedGPUMPNN(mpnn.ClassicMPNNBlock):
        def create_command(self, handler, sandbox_dir, chunk_id):
            command = super().create_command(handler, sandbox_dir, chunk_id)
            probe = (
                "import torch; from pathlib import Path; "
                "assert torch.cuda.is_available(); "
                "assert torch.ones(1, device='cuda').sum().item() == 1; "
                f"Path({str(sandbox_dir / 'gpu.txt')!r}).write_text(torch.cuda.get_device_name())"
            )
            return ["python", "-c", probe, "&&", *command]

    atoms = []
    for residue in range(1, 9):
        for atom_name, element, offset in [
            ("N", "N", 0),
            ("CA", "C", 1.4),
            ("C", "C", 2.5),
            ("O", "O", 3.0),
        ]:
            atoms.append(
                Atom(
                    [residue * 3.8 + offset, offset % 2, 0],
                    chain_id="Z",
                    res_id=residue,
                    res_name="GLY",
                    atom_name=atom_name,
                    element=element,
                )
            )
    path = tmp_path / "design.pdb"
    write_pdb(array(atoms), path)
    monkeypatch.setattr(mpnn, "MODELHUB_CONTAINER", mpnn._PUBLIC_MPNN_CONTAINER)
    engine = ExecutionEngine(
        tmp_path / "run", run_args_default={"mode": "modal", "max_retries": 0}
    )
    block_type = CheckedGPUMPNN if gpu else mpnn.ClassicMPNNBlock
    block = block_type(device=None if gpu else "cpu", num_seq_per_pdb=2)
    data = PipelineDataHandler.from_paths([path], name="input")
    resources = {"gres": "gpu:1"} if gpu else {}
    result = engine.run(block, data=data, job_args=resources)
    assert len(result) == 2
    for row in result._df.to_dicts():
        sequences = get_sequences_from_row(row)
        assert set(sequences) == {"Z"}
        assert len(sequences["Z"]) == 8
        assert set(sequences["Z"]) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert len(engine.run(block, data=data, job_args=resources)) == 2
    (record,) = load_execution_records(tmp_path / "run" / "mpnn")
    assert record.backend == "modal"
    assert record.allocated_gpus == int(gpu)
    assert record.status == "succeeded"
    if gpu:
        assert next((tmp_path / "run").rglob("gpu.txt")).read_text().strip()
