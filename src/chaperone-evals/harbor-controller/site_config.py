"""Trusted controller configuration; never read configuration from agent workspaces."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAPABILITIES = (
    "HBPLUS",
    "AlphaFold3 container",
    "Foundry RF3/RFD3 container",
    "LigandMPNN container",
    "LigandMPNN entry point",
    "LigandMPNN weights",
    "RF3 checkpoint",
)
RUNTIME_BINDS = [
    "/dev/fuse",
    "/etc/slurm",
    "/run/munge/munge.socket.2",
    "/usr/lib/slurm",
    "/usr/lib/x86_64-linux-gnu/libmunge.so.2",
    *[
        f"/usr/bin/{name}"
        for name in ("sacct", "scancel", "scontrol", "squeue", "srun", "sstat")
    ],
]
ENV_NAMES = {
    "CHAPERONE_DEFAULT_PARTITION",
    "CHAPERONE_GPU_PARTITION",
    "CHAPERONE_GRES_MAP",
    "CHAPERONE_CONTAINER_DIRS",
    "CHAPERONE_MPNN_CONTAINER",
    "HF_HUB_CACHE",
    "BOLTZ_CACHE_DIR",
    "AF3_JAX_CACHE_DIR",
    "ALPHAFOLD3_WEIGHTS_DIR",
    "OPENFOLD3_CACHE_DIR",
    "HBPLUS_COPY",
    "PYROSETTA_PATH",
    "X3DNA_BIN_DIR",
    "X3DNA_DIR",
    "FOLDSEEK_EXEC",
    "FOUNDRY_CHECKPOINT_DIRS",
    "CCD_MIRROR_PATH",
    "PDB_MIRROR_PATH",
    "DALPHABALL_BIN",
    "ROSETTA_BIN",
    "MPNN_WEIGHTS_DIR",
    "ESM_WEIGHTS_DIR",
    "SIMPLEFOLD_WEIGHTS_DIR",
    "CHAI_DOWNLOADS_DIR",
    "RFDIFFUSION_WEIGHTS_DIR",
    "PROTEIN_MSA_DIRS",
    "PROTENIX_CACHE_DIR",
    "MOLFILE_TO_PARAMS_SCRIPT",
    "DUNBRACK_LIBRARY_COPY",
    "RPXDOCK_CONTAINER",
}


def get_site() -> dict:
    name = os.environ.get("CHAPERONE_EVAL_SITE", "portable")
    if name not in {"portable", "digs"}:
        raise ValueError("CHAPERONE_EVAL_SITE must be portable or digs")
    site = dict(
        name=name,
        agent_image="",
        worker_image="",
        baseline_image="",
        worker_python="/opt/chaperone/.venv/bin/python",
        cpu_partition="",
        gpu_partition="",
        gpu_type="",
        gpu_constraint="",
        account="",
        qos="",
        max_total_gpus=5,
        max_requested_gpus_per_submission=30,
        runtime_binds=list(RUNTIME_BINDS),
        env_file="",
        environment={},
        capabilities={key: [] for key in CAPABILITIES},
    )
    if name == "digs":
        site.update(json.loads((ROOT / "sites/digs.json").read_text()))
        from dotenv import dotenv_values

        preset = ROOT.parents[1] / "chaperone/site_profiles/digs.env"
        supplied = ROOT.parents[2] / ".env.ipd"
        for path in (preset, supplied):
            if path.is_file():
                site["environment"].update(
                    {
                        k: v
                        for k, v in dotenv_values(path).items()
                        if k in ENV_NAMES and v is not None
                    }
                )
    config = os.environ.get("CHAPERONE_EVAL_SITE_CONFIG")
    overrides = {}
    if config:
        overrides = json.loads(Path(config).expanduser().read_text())
        if not isinstance(overrides, dict) or set(overrides) - site.keys():
            raise ValueError("Unknown eval site configuration fields")
        for key, value in overrides.items():
            if key in {"environment", "capabilities"}:
                site[key] = {**site[key], **value}
            else:
                site[key] = value
    for key in (
        "cpu_partition",
        "gpu_partition",
        "gpu_type",
        "gpu_constraint",
        "account",
        "qos",
    ):
        if not isinstance(site[key], str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]*", site[key]
        ):
            raise ValueError(f"Invalid site {key}")
    for key in ("max_total_gpus", "max_requested_gpus_per_submission"):
        if type(site[key]) is not int or site[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("agent_image", "worker_image", "baseline_image"):
        site[key] = os.environ.get("CHAPERONE_EVAL_" + key.upper(), site[key])
        if site[key]:
            site[key] = str(Path(site[key]).expanduser().resolve())
    if (
        not isinstance(site["worker_python"], str)
        or not Path(site["worker_python"]).is_absolute()
    ):
        raise ValueError("worker_python must be an absolute container path")
    for entry in site["runtime_binds"]:
        paths = [entry] if isinstance(entry, str) else entry
        if (
            not isinstance(paths, list)
            or (len(paths) != (1 if isinstance(entry, str) else 2))
            or any(
                not isinstance(p, str)
                or not Path(p).is_absolute()
                or any(c in p for c in ":,\n\x00")
                for p in paths
            )
        ):
            raise ValueError(
                "runtime_binds require absolute paths or [source, target] pairs"
            )
    if set(site["environment"]) - ENV_NAMES:
        raise ValueError("Unapproved scientific environment variable in site config")
    if any(not isinstance(v, str) or "\x00" in v for v in site["environment"].values()):
        raise ValueError("Scientific environment values must be strings without NUL")
    if set(site["capabilities"]) - set(CAPABILITIES):
        raise ValueError("Unknown runtime capability")
    for paths in site["capabilities"].values():
        if not isinstance(paths, list) or any(
            not isinstance(p, str) or not Path(p).is_absolute() for p in paths
        ):
            raise ValueError("Capability paths must be absolute")
    if site["env_file"]:
        from dotenv import dotenv_values

        env_file = Path(site["env_file"]).expanduser()
        if not env_file.is_file():
            raise ValueError(f"Eval env_file does not exist: {env_file}")
        supplied = {
            k: v
            for k, v in dotenv_values(env_file).items()
            if k in ENV_NAMES and v is not None
        }
        site["environment"] = {
            **site["environment"],
            **supplied,
            **overrides.get("environment", {}),
        }
    if site["environment"].get("CHAPERONE_MPNN_CONTAINER", "auto") not in {
        "auto", "portable", "legacy"
    }:
        raise ValueError("CHAPERONE_MPNN_CONTAINER must be auto, portable or legacy")
    site["gpu_partition"] = site["gpu_partition"] or site["cpu_partition"]
    # The scheduler mapping and broker must agree even when the snapshot has no dotenv files.
    gres = "gpu:" + (site["gpu_type"] + ":" if site["gpu_type"] else "") + "1"
    site["environment"].update(
        {
            "CHAPERONE_DEFAULT_PARTITION": site["cpu_partition"],
            "CHAPERONE_GPU_PARTITION": site["gpu_partition"],
            "CHAPERONE_GRES_MAP": json.dumps({"small": gres, "large": gres}),
        }
    )
    return site


def scoring_environment():
    """Host-side scorer receives the same assets/queues as its worker wrapper."""
    site = get_site()
    return {
        **site["environment"],
        "CHAPERONE_DEFAULT_ACCOUNT": site["account"],
        "CHAPERONE_DEFAULT_QOS": site["qos"],
    }


def add_site_arguments(parser):
    parser.add_argument(
        "--site",
        choices=("portable", "digs"),
        default=os.environ.get("CHAPERONE_EVAL_SITE", "portable"),
    )
    parser.add_argument(
        "--site-config",
        type=Path,
        default=os.environ.get("CHAPERONE_EVAL_SITE_CONFIG"),
        help="Trusted JSON site overrides; paths are absolute and shared across nodes",
    )


def configure_site(args):
    os.environ["CHAPERONE_EVAL_SITE"] = args.site
    if args.site_config:
        os.environ["CHAPERONE_EVAL_SITE_CONFIG"] = str(
            Path(args.site_config).expanduser().resolve()
        )
    for attr, key in (
        ("worker_image", "worker_image"),
        (
            "image",
            "baseline_image"
            if getattr(args, "arm", None) == "baseline"
            else "agent_image",
        ),
    ):
        value = getattr(args, attr, None)
        if value:
            os.environ["CHAPERONE_EVAL_" + key.upper()] = str(value)
    site = get_site()
    if hasattr(args, "worker_image"):
        args.worker_image = Path(site["worker_image"]) if site["worker_image"] else None
    if hasattr(args, "image"):
        key = "baseline_image" if args.arm == "baseline" else "agent_image"
        args.image = Path(site[key]) if site[key] else None
        if args.image is None:
            raise ValueError(
                "Specify --image or configure the selected arm's image in --site-config"
            )
    if (
        not getattr(args, "no_score", False)
        and hasattr(args, "worker_image")
        and args.worker_image is None
    ):
        raise ValueError(
            "Scoring requires --worker-image or worker_image in --site-config"
        )
    if hasattr(args, "cpu_partition") and args.cpu_partition is None:
        args.cpu_partition = site["cpu_partition"]
    return site


def policy_instructions(arm):
    site = get_site()
    cpu = (
        f"--partition={site['cpu_partition']}"
        if site["cpu_partition"]
        else "omit --partition (cluster default)"
    )
    gpu = (
        f"--partition={site['gpu_partition']}"
        if site["gpu_partition"]
        else "omit --partition (cluster default)"
    )
    gres = f"gpu:{site['gpu_type'] + ':' if site['gpu_type'] else ''}N"
    return (
        f"CPU jobs: {cpu}, without a GPU directive. GPU jobs: {gpu}, --gres={gres}. "
        f"The controller applies GPU constraint {site['gpu_constraint'] or '(none)'}. "
        "Do not add your own constraint, account, or QoS. "
        f"The trial may reserve at most {site['max_total_gpus']} GPUs across queued/running jobs, "
        "including array concurrency. Each submission may request at most "
        f"{site['max_requested_gpus_per_submission']} GPUs across all array tasks, "
        "regardless of concurrency throttling. "
        + (
            "Use JobArgs(gpu_class='small') for GPU blocks and the configured defaults for CPU blocks. "
            "Set slurm_array_parallelism at or below the remaining trial GPU capacity. "
            "Use partition='local' only for lightweight CPU work alongside the agent."
            if arm == "harness"
            else ""
        )
    )
