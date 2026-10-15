"""Slurm resource resolution: site mapping, new JobArgs fields, precedence, IPD parity.

No cluster is needed. The sbatch text is rendered by submitit with a stub ``srun``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from absl import flags as absl_flags

import chaperone
from blocks import Blocks
from chaperone import _GLOBAL_FLAG_DEFS, JobArgs
from chaperone.engine.resolve import RuntimeArgs, _resolve_job_args
from chaperone.orchestration import SlurmTaskRunner
from chaperone.orchestration.job_args import (
    parse_sbatch_args,
    site_gres,
    site_partition,
)
from chaperone.settings import ChaperoneSettings

REPO = Path(__file__).resolve().parent.parent
SITE_FIELDS = {
    "CHAPERONE_DEFAULT_PARTITION": None,
    "CHAPERONE_GPU_PARTITION": None,
    "CHAPERONE_GRES_MAP": {},
    "CHAPERONE_DEFAULT_ACCOUNT": None,
    "CHAPERONE_DEFAULT_QOS": None,
    "CHAPERONE_DEFAULT_EXTRA_SBATCH": None,
}


@pytest.fixture
def site(monkeypatch):
    """Set the Slurm site settings for one test; everything else is unset."""
    settings = chaperone.get_settings()

    def configure(**values):
        for name, default in SITE_FIELDS.items():
            monkeypatch.setattr(settings, name, values.get(name, default))

    configure()
    return configure


@pytest.fixture
def cli():
    """Parse global flags from an argv list; restore a clean parse afterwards."""
    flag_values = absl_flags.FLAGS
    for name, ftype, default, help_text in _GLOBAL_FLAG_DEFS:
        if hasattr(flag_values, name):
            continue
        if ftype == "string":
            absl_flags.DEFINE_string(name, default, help_text)
        else:
            absl_flags.DEFINE_boolean(name, default, help_text)

    def parse(*argv):
        flag_values.unparse_flags()
        flag_values(["prog", *argv])

    parse()
    yield parse
    flag_values.unparse_flags()
    flag_values(["prog"])


class _Block:
    """Just enough of a block for ``_resolve_job_args``."""

    def __init__(self, resources, name="fake"):
        self._resources = resources
        self.name = name
        self._chunked = True
        self._rows_per_job = 50

    def get_resources(self):
        return self._resources


GPU_BLOCK = _Block(JobArgs(partition="gpu", gpu_class="small", mem_gb=32))
CPU_BLOCK = _Block(JobArgs(partition="cpu", mem_gb=8))


def resolve(block, job_args=None):
    return _resolve_job_args(job_args, block, RuntimeArgs(use_local_gpus=False))


def sbatch_lines(job_args: JobArgs, tmp_path, monkeypatch, via="runner") -> list[str]:
    """The ``#SBATCH`` lines submitit would write for these job args."""
    from submitit.slurm.slurm import _make_sbatch_string

    srun = tmp_path / "bin" / "srun"
    srun.parent.mkdir(exist_ok=True)
    srun.write_text("#!/bin/sh\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{srun.parent}:/usr/bin:/bin")
    if via == "runner":
        with SlurmTaskRunner(job_args=job_args, log_folder=str(tmp_path)) as runner:
            executor = runner._executor
    else:
        executor = job_args.create_executor(log_folder=str(tmp_path))
    text = _make_sbatch_string(
        command="CMD",
        folder=str(tmp_path),
        map_count=2,
        **executor._executor.parameters,
    )
    skip = ("--output", "--error", "--open-mode", "--signal", "--array", "--wckey")
    return [
        line[len("#SBATCH ") :]
        for line in text.splitlines()
        if line.startswith("#SBATCH") and not any(k in line for k in skip)
    ]


# ── the site mapping ────────────────────────────────────────────────────────


def test_unconfigured_site_is_portable(site):
    assert site_gres("small") == "gpu:1"
    assert site_gres(None) == "gpu:1"
    assert site_partition(gpu=True) is None
    assert site_partition(gpu=False) is None


def test_gres_map_with_default_key(site):
    site(CHAPERONE_GRES_MAP={"small": "gpu:a4000:1", "default": "gpu:a100:1"})
    assert site_gres("small") == "gpu:a4000:1"
    assert site_gres("large") == "gpu:a100:1"
    assert site_gres(None) == "gpu:a100:1"


def test_gpu_partition_falls_back_to_default_partition(site):
    site(CHAPERONE_DEFAULT_PARTITION="batch")
    assert site_partition(gpu=True) == "batch"
    assert site_partition(gpu=False) == "batch"
    site(CHAPERONE_DEFAULT_PARTITION="batch", CHAPERONE_GPU_PARTITION="accel")
    assert site_partition(gpu=True) == "accel"
    assert site_partition(gpu=False) == "batch"


def test_gres_map_reads_json_from_the_environment(monkeypatch):
    monkeypatch.setenv("CHAPERONE_GRES_MAP", '{"small": "gpu:t4:1"}')
    assert ChaperoneSettings(_env_file=None).CHAPERONE_GRES_MAP == {"small": "gpu:t4:1"}


# ── block defaults through the mapping ──────────────────────────────────────


def test_block_defaults_on_an_unconfigured_site(site):
    gpu = resolve(GPU_BLOCK)
    assert (gpu.partition, gpu.gres) == (None, "gpu:1")
    assert gpu.mem_gb == 32 and gpu.job_name == "fake"
    cpu = resolve(CPU_BLOCK)
    assert (cpu.partition, cpu.gres) == (None, None)


def test_block_defaults_on_a_configured_site(site):
    site(
        CHAPERONE_DEFAULT_PARTITION="batch",
        CHAPERONE_GPU_PARTITION="accel",
        CHAPERONE_GRES_MAP={"small": "gpu:a4000:1"},
    )
    gpu = resolve(GPU_BLOCK)
    assert (gpu.partition, gpu.gres) == ("accel", "gpu:a4000:1")
    cpu = resolve(CPU_BLOCK)
    assert (cpu.partition, cpu.gres) == ("batch", None)


def test_gpu_block_without_a_size_class_gets_one_generic_gpu(site):
    resolved = resolve(_Block(JobArgs(partition="gpu")))
    assert resolved.gres == "gpu:1"
    assert resolved.wants_gpu


def test_block_with_a_concrete_partition_name_is_not_remapped(site):
    site(CHAPERONE_GPU_PARTITION="accel")
    resolved = resolve(_Block(JobArgs(partition="gpu-bf", gres="gpu:2")))
    assert (resolved.partition, resolved.gres) == ("gpu-bf", "gpu:2")


# ── precedence: CLI > job_args > block > site ───────────────────────────────


def test_job_args_beat_block_and_site(site):
    site(CHAPERONE_GPU_PARTITION="accel", CHAPERONE_GRES_MAP={"small": "gpu:s:1"})
    from_dict = resolve(GPU_BLOCK, {"partition": "mine", "gres": "gpu:x:1"})
    assert (from_dict.partition, from_dict.gres) == ("mine", "gpu:x:1")
    from_obj = resolve(GPU_BLOCK, JobArgs(partition="mine", gres="gpu:x:1"))
    assert (from_obj.partition, from_obj.gres) == ("mine", "gpu:x:1")


def test_job_args_gpu_class_is_mapped_by_the_site(site):
    site(CHAPERONE_GRES_MAP={"small": "gpu:s:1", "large": "gpu:l:1"})
    resolved = resolve(GPU_BLOCK, {"gpu_class": "large"})
    assert resolved.gres == "gpu:l:1"
    # An explicit gres wins over a class in the same job_args.
    both = resolve(GPU_BLOCK, {"gpu_class": "large", "gres": "gpu:x:1"})
    assert both.gres == "gpu:x:1"


def test_job_args_gpu_class_shared_across_blocks_skips_cpu_blocks(site):
    site(CHAPERONE_GRES_MAP={"large": "gpu:l:1"})
    shared = {"gpu_class": "large", "mem_gb": 16}
    assert resolve(GPU_BLOCK, shared).gres == "gpu:l:1"
    cpu = resolve(CPU_BLOCK, shared)
    assert cpu.gres is None and cpu.mem_gb == 16
    # Unless the caller also moves the CPU block onto a GPU partition.
    assert resolve(CPU_BLOCK, {**shared, "partition": "gpu"}).gres == "gpu:l:1"


def test_user_partition_is_used_verbatim_even_if_it_is_a_class_name(site):
    site(CHAPERONE_GPU_PARTITION="accel")
    assert resolve(GPU_BLOCK, {"partition": "gpu"}).partition == "gpu"


def test_cli_beats_job_args(site, cli):
    site(CHAPERONE_GRES_MAP={"small": "gpu:s:1"})
    cli("--partition=fromcli", "--gres=gpu:cli:1")
    resolved = resolve(GPU_BLOCK, {"partition": "mine", "gres": "gpu:x:1"})
    assert (resolved.partition, resolved.gres) == ("fromcli", "gpu:cli:1")


def test_cli_partition_and_gres_leave_cpu_blocks_alone(site, cli):
    cli("--partition=fromcli", "--gres=gpu:cli:1")
    resolved = resolve(CPU_BLOCK)
    assert (resolved.partition, resolved.gres) == (None, None)


def test_cli_partition_drops_the_gpu_type_from_a_mapped_gres(site, cli):
    site(CHAPERONE_GRES_MAP={"small": "gpu:a4000:1"})
    cli("--partition=other")
    assert resolve(GPU_BLOCK).gres == "gpu:1"
    cli("--partition=other", "--gres=gpu:a100:1")
    assert resolve(GPU_BLOCK).gres == "gpu:a100:1"


def test_account_qos_extra_precedence(site, cli, tmp_path, monkeypatch):
    site(
        CHAPERONE_DEFAULT_ACCOUNT="site_acct",
        CHAPERONE_DEFAULT_QOS="site_qos",
        CHAPERONE_DEFAULT_EXTRA_SBATCH="--reservation=site",
    )
    lines = sbatch_lines(resolve(CPU_BLOCK), tmp_path, monkeypatch)
    assert {"--account=site_acct", "--qos=site_qos", "--reservation=site"} <= set(lines)

    lines = sbatch_lines(
        resolve(
            CPU_BLOCK,
            {
                "account": "job_acct",
                "qos": "job_qos",
                "extra_sbatch": ["--reservation=job"],
            },
        ),
        tmp_path,
        monkeypatch,
    )
    assert {"--account=job_acct", "--qos=job_qos", "--reservation=job"} <= set(lines)
    assert "--reservation=site" not in lines

    cli(
        "--account=cli_acct",
        "--qos=cli_qos",
        "--extra_sbatch=--reservation=cli --exclusive",
    )
    lines = sbatch_lines(
        resolve(CPU_BLOCK, {"account": "job_acct", "qos": "job_qos"}),
        tmp_path,
        monkeypatch,
    )
    assert {
        "--account=cli_acct",
        "--qos=cli_qos",
        "--reservation=cli",
        "--exclusive",
    } <= set(lines)


# ── the sbatch text ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("via", ["runner", "executor"])
def test_unconfigured_gpu_block_renders_a_portable_sbatch(
    site, tmp_path, monkeypatch, via
):
    lines = sbatch_lines(resolve(GPU_BLOCK), tmp_path, monkeypatch, via)
    assert "--gres=gpu:1" in lines
    assert not any(
        line.startswith(("--partition", "--account", "--qos")) for line in lines
    )


@pytest.mark.parametrize("via", ["runner", "executor"])
def test_unconfigured_cpu_block_asks_for_no_gpu_and_no_partition(
    site, tmp_path, monkeypatch, via
):
    lines = sbatch_lines(resolve(CPU_BLOCK), tmp_path, monkeypatch, via)
    assert not any(line.startswith(("--partition", "--gres")) for line in lines)


def test_bare_job_args_use_the_default_partition(site, tmp_path, monkeypatch):
    assert not any(
        line.startswith("--partition")
        for line in sbatch_lines(JobArgs(), tmp_path, monkeypatch)
    )
    site(CHAPERONE_DEFAULT_PARTITION="batch", CHAPERONE_GPU_PARTITION="accel")
    assert "--partition=batch" in sbatch_lines(JobArgs(), tmp_path, monkeypatch)
    assert "--partition=accel" in sbatch_lines(
        JobArgs(gres="gpu:1"), tmp_path, monkeypatch
    )
    assert "--partition=mine" in sbatch_lines(
        JobArgs(partition="mine"), tmp_path, monkeypatch
    )


def test_gres_types_are_passed_through_unchecked(site, tmp_path, monkeypatch):
    """chaperone does not validate GPU types; the cluster decides what is valid.

    IPD registers size classes (gpu:small:1) and rejects model names; most other
    clusters register model names (gpu:a100:1). Both reach sbatch verbatim.
    """
    for gres in ("gpu:a4000:1", "gpu:small:1", "gpu:2"):
        lines = sbatch_lines(JobArgs(gres=gres), tmp_path, monkeypatch)
        assert f"--gres={gres}" in lines


def test_extra_sbatch_forms_and_errors():
    assert parse_sbatch_args(
        ["--reservation=a", "--exclusive", "--mail-type", "END"]
    ) == {
        "reservation": "a",
        "exclusive": True,
        "mail-type": "END",
    }
    assert JobArgs(extra_sbatch="--a=1 --b").extra_sbatch == ["--a=1", "--b"]
    with pytest.raises(ValueError, match="long"):
        parse_sbatch_args(["-w", "node1"])
    with pytest.raises(ValueError, match="long"):
        JobArgs(extra_sbatch=["-w", "node1"])


def test_extra_sbatch_coexists_with_nice(site, tmp_path, monkeypatch):
    lines = sbatch_lines(
        JobArgs(nice=5, extra_sbatch=["--reservation=a"]), tmp_path, monkeypatch
    )
    assert {"--nice=5", "--reservation=a"} <= set(lines)


def test_direct_use_without_the_engine_maps_gpu_class(site, tmp_path, monkeypatch):
    site(CHAPERONE_GPU_PARTITION="accel", CHAPERONE_GRES_MAP={"large": "gpu:l:1"})
    for via in ("runner", "executor"):
        lines = sbatch_lines(JobArgs(gpu_class="large"), tmp_path, monkeypatch, via)
        assert {"--gres=gpu:l:1", "--partition=accel"} <= set(lines)


def test_duplicate_keeps_account_qos_and_extra(site, tmp_path, monkeypatch):
    runner = SlurmTaskRunner(
        job_args=JobArgs(account="a", qos="q", extra_sbatch=["--exclusive"]),
        log_folder=str(tmp_path),
    )
    copy = runner.duplicate()
    assert (
        copy._job_args.account,
        copy._job_args.qos,
        copy._job_args.extra_sbatch,
    ) == (
        "a",
        "q",
        ["--exclusive"],
    )


# ── blocks and IPD parity ───────────────────────────────────────────────────

# What every in-tree block requested before the mapping existed. These are the
# IPD values; with .env.ipd loaded the resolved resources must still be these.
LARGE_GPU_BLOCKS = {
    "bindcraft", "bindcraft2", "simplefold", "proteina_complexa", "rfoptimization",
}  # fmt: skip
GPU_BLOCKS = {
    "af3", "af3_with_target_template", "bindcraft", "bindcraft2", "boltz2", "boltz2_steered",
    "boltzgen", "chai1", "colabfold", "disco", "e1", "esm2", "esmfold2",
    "openfold3", "protein_hunter", "protenix", "rf3", "rfd1", "rfd2",
    "rfd3", "rfd3_partial", "rfd3na", "simplefold", "proteina_complexa", "rfoptimization",
}  # fmt: skip


def _blocks_with_resources():
    for name in sorted(Blocks.keys()):
        cls = Blocks[name]
        if getattr(cls, "_resources", None) is not None:
            yield name, cls


def test_gpu_block_list_matches_the_registry():
    declared = {
        name
        for name, cls in _blocks_with_resources()
        if cls._resources.partition == "gpu"
    }
    assert declared == GPU_BLOCKS


def test_blocks_declare_classes_not_cluster_resources():
    for name, cls in _blocks_with_resources():
        resources = cls._resources
        assert resources.partition in ("cpu", "gpu"), name
        assert resources.gres is None, f"{name} pins a gres; declare gpu_class"
        assert (resources.gpu_class is not None) == (name in GPU_BLOCKS), name


def test_no_bare_size_class_gres_literal_in_block_sources():
    pattern = re.compile(r'^\s+gres="gpu:(small|large):\d+",$', re.MULTILINE)
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "blocks").rglob("*.py")
        if pattern.search(path.read_text())
    ]
    assert offenders == []


@pytest.fixture
def ipd_site(site, monkeypatch):
    """Apply the Slurm settings that .env.ipd carries."""
    ipd = ChaperoneSettings(_env_file=REPO / "src/chaperone/site_profiles/digs.env")
    settings = chaperone.get_settings()
    for name in SITE_FIELDS:
        monkeypatch.setattr(settings, name, getattr(ipd, name))
    return ipd


def test_ipd_env_maps_the_size_classes(ipd_site):
    assert ipd_site.CHAPERONE_GRES_MAP == {
        "small": "gpu:small:1",
        "large": "gpu:large:1",
    }
    assert ipd_site.CHAPERONE_GPU_PARTITION == "gpu"
    assert ipd_site.CHAPERONE_DEFAULT_PARTITION == "cpu"


def test_ipd_resolved_resources_are_unchanged(ipd_site):
    for name, cls in _blocks_with_resources():
        block = cls.model_construct(name=name)
        resolved = resolve(block)
        if name in GPU_BLOCKS:
            size = "large" if name in LARGE_GPU_BLOCKS else "small"
            assert (resolved.partition, resolved.gres) == ("gpu", f"gpu:{size}:1"), name
        else:
            assert (resolved.partition, resolved.gres) == ("cpu", None), name


def test_ipd_sbatch_lines_are_unchanged(ipd_site, tmp_path, monkeypatch):
    boltz2 = resolve(Blocks["boltz2"].model_construct(name="boltz2"))
    for via in ("runner", "executor"):
        lines = sbatch_lines(boltz2, tmp_path, monkeypatch, via)
        assert "--partition=gpu" in lines and "--gres=gpu:small:1" in lines
    bindcraft = resolve(Blocks["bindcraft"].model_construct(name="bindcraft"))
    assert "--gres=gpu:large:1" in sbatch_lines(bindcraft, tmp_path, monkeypatch)
    mpnn = resolve(Blocks["mpnn"].model_construct(name="mpnn"))
    lines = sbatch_lines(mpnn, tmp_path, monkeypatch)
    assert "--partition=cpu" in lines and not any(
        line.startswith("--gres") for line in lines
    )


def test_ipd_cli_partition_still_strips_the_size_class(ipd_site, cli):
    cli("--partition=gpu-bf")
    resolved = resolve(Blocks["boltz2"].model_construct(name="boltz2"))
    assert (resolved.partition, resolved.gres) == ("gpu-bf", "gpu:1")
