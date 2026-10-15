"""Public BindCraft2 runtime; independent of classic BindCraft and Foundry."""

from chaperone.containers import Image

BINDCRAFT2_REVISION = "c0ee638d11bc010817c2c97ee58ed5c323e664e2"
BINDCRAFT2_CONTAINER = (
    Image("bindcraft2")
    .base_image("ubuntu:24.04")
    .packages(
        "python3.12",
        "python3.12-venv",
        "python3.12-dev",
        "build-essential",
        "ca-certificates",
        "git",
    )
    .run(
        "git clone https://github.com/PacesaLab/BindCraft2.git /opt/bindcraft2",
        f"git -C /opt/bindcraft2 checkout --detach {BINDCRAFT2_REVISION}",
        "python3.12 -m venv /opt/bindcraft2-env",
        "/opt/bindcraft2-env/bin/pip install --upgrade pip setuptools wheel",
        "/opt/bindcraft2-env/bin/pip install -e '/opt/bindcraft2[cuda12]' 'jax[cuda12]==0.11.0' biotite==1.6.0 numpy==2.4.3",
        # The JAX wheels carry CUDA libraries; make them visible to the loader.
        "/opt/bindcraft2-env/bin/python -c \"import nvidia, pathlib; print('\\n'.join(sorted(str(p) for root in nvidia.__path__ for p in pathlib.Path(root).glob('*/lib'))))\" > /etc/ld.so.conf.d/bindcraft2-cuda.conf",
        "ldconfig",
        "/opt/bindcraft2-env/bin/pip check",
        "/opt/bindcraft2-env/bin/bindcraft --help >/dev/null",
        "/opt/bindcraft2-env/bin/python -m bindcraft.selfcheck cuda12 --shipped-only",
        "apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip",
    )
    .env(
        PATH="/opt/bindcraft2-env/bin:$PATH",
        VIRTUAL_ENV="/opt/bindcraft2-env",
        NVIDIA_VISIBLE_DEVICES="all",
        NVIDIA_DRIVER_CAPABILITIES="compute,utility",
        PYTHONNOUSERSITE="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    .label(
        Revision=BINDCRAFT2_REVISION,
        Source="https://github.com/PacesaLab/BindCraft2",
        JAX="0.11.0",
        CUDA="12",
    )
    .runscript('"$@"')
    .template_command(
        exec_mode="exec",
        env={"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
)
