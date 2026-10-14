"""Private assets round-trip without entering a task's agent environment."""

import io
import json
import tarfile

import pytest

from scripts.private_dataset import export_dataset, restore_dataset


@pytest.fixture
def exported(tmp_path):
    private = tmp_path / "private"
    for folder in ("inputs", "score", "example"):
        (private / "demo" / folder).mkdir(parents=True)
    (private / "demo/prompt.md").write_text("Design a test structure.")
    (private / "demo/inputs/target.txt").write_text("public target")
    (private / "demo/score/score.py").write_text("private scorer")
    (private / "demo/example/answer.txt").write_text("private answer")
    (private / "audit_hints.json").write_text(
        '[{"label":"test","pattern":"private answer"}]'
    )
    bundle = export_dataset(private, tmp_path / "export", "test-org/test-evals")
    return private, bundle


def test_roundtrip_and_model_visible_files(exported, tmp_path):
    from harbor.models.dataset.manifest import DatasetManifest
    from harbor.publisher.packager import Packager

    private, bundle = exported
    restore = restore_dataset(bundle, tmp_path / "restored")
    for file in private.rglob("*"):
        if file.is_file():
            assert (
                restore / file.relative_to(private)
            ).read_bytes() == file.read_bytes()
    model_files = list((bundle / "demo/environment").rglob("*"))
    assert not any(
        "private answer" in p.read_text() or "private scorer" in p.read_text()
        for p in model_files
        if p.is_file()
    )
    manifest = DatasetManifest.from_toml_file(bundle / "dataset.toml")
    assert len(manifest.tasks) == 1
    assert (
        manifest.tasks[0].digest
        == "sha256:" + Packager.compute_content_hash(bundle / "demo")[0]
    )
    assert "exit 2" in (bundle / "demo/tests/test.sh").read_text()


def test_archive_corruption_fails_before_creating_destination(exported, tmp_path):
    _, bundle = exported
    inventory = json.loads((bundle / "SHA256SUMS.json").read_text())
    inventory["demo/prompt.md"] = "0" * 64
    (bundle / "SHA256SUMS.json").write_text(json.dumps(inventory))
    output = tmp_path / "restored"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        restore_dataset(bundle, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
    ],
)
def test_archive_path_escape_is_rejected(exported, tmp_path, name, kind):
    _, bundle = exported
    with tarfile.open(bundle / "private-evals.tar.gz", "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        if kind == tarfile.SYMTYPE:
            member.linkname = "/etc/passwd"
        archive.addfile(member, io.BytesIO(b""))
    with pytest.raises(ValueError, match="Unsafe or unexpected"):
        restore_dataset(bundle, tmp_path / "restored")
