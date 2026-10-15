"""Execute the image's download shell logic with a failing upstream host."""

import hashlib
import os
import subprocess
import sys

import pytest

from blocks.networks import mpnn


@pytest.mark.parametrize("mirror_mode", ["valid", "corrupt", "unavailable"])
def test_checkpoint_fallback_requires_exact_bytes_and_reuses_mirror(
    tmp_path, mirror_mode
):
    payload = b"checkpoint bytes for the shell recovery test"
    checksum = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "ligandmpnn"
    (target / "model_params").mkdir(parents=True)
    downloads = [
        command
        for group in mpnn.LIGANDMPNN_CONTAINER._commands
        for command in group
        if "MPNN_USE_MIRROR" in command
    ][:2]
    for index, (_, expected) in enumerate(mpnn._MPNN_ALL_CHECKPOINTS[:2]):
        downloads[index] = (
            downloads[index]
            .replace(mpnn._MPNN_ROOT, str(target))
            .replace(expected, checksum)
        )
    wget = tmp_path / "wget"
    wget.write_text(f"""#!{sys.executable}
import os, pathlib, sys
url=next(arg for arg in sys.argv if arg.startswith('https://'))
with open(os.environ['REQUEST_LOG'], 'a') as log: log.write(url+'\\n')
if 'files.ipd.uw.edu' in url: sys.exit(1)
if os.environ['MIRROR_MODE']=='unavailable': sys.exit(1)
payload={payload!r} if os.environ['MIRROR_MODE']=='valid' else b'corrupt'
pathlib.Path(sys.argv[sys.argv.index('-O')+1]).write_bytes(payload)
""")
    wget.chmod(0o755)
    log = tmp_path / "requests.log"
    result = subprocess.run(
        ["sh", "-c", "\n".join(downloads) + "\necho COMPLETE"],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "REQUEST_LOG": str(log),
            "MIRROR_MODE": mirror_mode,
        },
        capture_output=True,
        text=True,
    )
    requests = log.read_text().splitlines()
    assert sum("files.ipd.uw.edu" in url for url in requests) == 1
    if mirror_mode == "valid":
        assert result.returncode == 0, result.stderr
        assert "COMPLETE" in result.stdout
        assert len(requests) == 3  # one failed primary, then two pinned mirrors
        assert all(
            path.read_bytes() == payload for path in (target / "model_params").iterdir()
        )
    else:
        assert result.returncode != 0
        assert "COMPLETE" not in result.stdout
        assert len(requests) == 2
