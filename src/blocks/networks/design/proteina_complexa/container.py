"""Pinned, self-contained protein-target Proteina-Complexa runtime."""

import shlex

from chaperone.containers import Image

COMPLEXA_REVISION = "acf55a89a9a7206d5ce9abc0bf2246c3b3829f92"

# Upstream installs AtomWorks in a second uv call, masking its SciPy/Einops conflicts.
# Align its metadata with AtomWorks' supported minimum; no algorithm code changes.
_METADATA_PATCH = """
from pathlib import Path
p = Path('/opt/complexa/pyproject.toml')
s = p.read_text()
for old, new in [('scipy==1.12.0', 'scipy==1.13.1'), ('einops==0.6', 'einops==0.8.0')]:
    if s.count(old) != 1:
        raise RuntimeError('Review Complexa dependency constraint: pinned metadata changed')
    s = s.replace(old, new)
p.write_text(s)
"""

# Import optional AF2/JAX only when hallucination is actually requested.
_OPTIONAL_IMPORT_PATCH = r"""
from pathlib import Path
base = Path('/opt/complexa/src/proteinfoundation/search')
p = base / '__init__.py'
s = p.read_text()
old = 'from proteinfoundation.search.sequence_hallucination import SequenceHallucination\n'
assert s.count(old) == 1
p.write_text(s.replace(old, '').replace('    "SequenceHallucination",\n', ''))
p = base / 'search_factory.py'
s = p.read_text()
assert s.count('    SequenceHallucination,\n') == 1
s = s.replace('    SequenceHallucination,\n', '')
old = '    if algorithm == "sequence_hallucination":\n        return SequenceHallucination(proteina, inf_cfg)'
assert s.count(old) == 1
s = s.replace(old, '    if algorithm == "sequence_hallucination":\n        from proteinfoundation.search.sequence_hallucination import SequenceHallucination\n        return SequenceHallucination(proteina, inf_cfg)')
p.write_text(s)
"""

COMPLEXA_CONTAINER = (
    Image("proteina_complexa")
    .base_image("ubuntu:24.04")
    .packages(
        "build-essential",
        "git",
        "curl",
        "ca-certificates",
        "python3.12",
        "python3.12-dev",
        "python3.12-venv",
        "libgl1",
        "libxrender1",
        "libsm6",
        "libxext6",
        "libglib2.0-0",
    )
    .run(
        "python3.12 -m venv /opt/complexa-env",
        "/opt/complexa-env/bin/pip install --upgrade 'pip>=26.2.2'",
        "/opt/complexa-env/bin/pip install uv==0.13.4",
        "git init /opt/complexa && git -C /opt/complexa remote add origin https://github.com/NVIDIA-BioNeMo/Proteina-Complexa.git && "
        f"git -C /opt/complexa fetch --depth 1 origin {COMPLEXA_REVISION} && git -C /opt/complexa checkout --detach FETCH_HEAD",
        "/opt/complexa-env/bin/uv pip install --python /opt/complexa-env/bin/python torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126",
        "python3.12 -c " + shlex.quote(_METADATA_PATCH),
        "python3.12 -c " + shlex.quote(_OPTIONAL_IMPORT_PATCH),
        "/opt/complexa-env/bin/uv pip install --python /opt/complexa-env/bin/python -e /opt/complexa 'atomworks[ml,openbabel]==2.2.1' biotite==1.4.0 graphein==1.7.7 numpy==1.26.4 transformers==5.3.0 pytorch-lightning==2.5.6",
        "/opt/complexa-env/bin/uv pip install --python /opt/complexa-env/bin/python torch_geometric torch_scatter torch_sparse torch_cluster --find-links https://data.pyg.org/whl/torch-2.7.0+cu126.html",
        "/opt/complexa-env/bin/uv pip check --python /opt/complexa-env/bin/python",
        "PYTHONPATH=/opt/complexa/community_models:/opt/complexa/src /opt/complexa-env/bin/python -c 'import proteinfoundation.generate; import proteinfoundation.datasets.gen_dataset'",
    )
    .env(
        PATH="/opt/complexa-env/bin:$PATH",
        VIRTUAL_ENV="/opt/complexa-env",
        PYTHONPATH="/opt/complexa/community_models:/opt/complexa/src",
        WANDB_MODE="disabled",
        MPLBACKEND="Agg",
    )
    .label(
        Version=COMPLEXA_REVISION,
        CUDA="12.6",
        Scope="protein-target-generation",
        AtomWorks="2.2.1",
    )
    .runscript('"$@"')
    .template_command(exec_mode="exec", binds=[], env={})
)
