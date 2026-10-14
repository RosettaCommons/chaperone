#!/usr/bin/env python3
"""CLI for installing containers for pipeline blocks.

Usage:
    chaperone install foundry e1
    chaperone install e1 --dry_run
    chaperone install e1 --force
"""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
from pathlib import Path
from absl import app, flags
import logging

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_boolean("force", False, "Force rebuild even if container exists")
flags.DEFINE_boolean("dry_run", False, "Show what would be done without doing it")
flags.DEFINE_boolean("full", False, "Install the curated core scientific containers")
flags.DEFINE_integer("jobs", None, "Concurrent builds (full: 2; named installs: 1)", lower_bound=1)
flags.DEFINE_boolean("warmup_rfd3", False, "After builds, warm the RFD3 GPU compiler cache")
flags.DEFINE_string("warmup_cache_dir", None, "Persistent writable RFD3 compiler cache")
flags.DEFINE_string("warmup_run_dir", None, "Shared directory for RFD3 warmup outputs")
flags.DEFINE_boolean("warmup_local", False, "Warm up on an already allocated local GPU")

# Names are container registrations, not block aliases. Foundry includes RFD3,
# RFD3NA, RF3 and FoundryMPNN; ligandmpnn includes Protein/Ligand/SolubleMPNN.
CORE_CONTAINERS = (
    "foundry", "ligandmpnn", "alphafold3", "esmfold2", "boltz2", "chai1",
    "colabfold_portable",
)


def get_writable_container_dir() -> Path:
    """Get the writable container directory."""
    from chaperone.containers import get_container_dirs

    dirs = get_container_dirs()
    for d in dirs:
        if d.exists() and d.is_dir():
            try:
                # Concurrent installers must not remove one another's probe.
                with tempfile.NamedTemporaryFile(dir=d, prefix=".chaperone-write-test-"):
                    pass
                return d
            except (PermissionError, OSError) as e:
                logger.warning(
                    "Skipping container directory %s: unable to write test file (%s)",
                    d,
                    e,
                )
                continue
        elif not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                return d
            except (PermissionError, OSError) as e:
                logger.warning(
                    "Skipping container directory %s: unable to create directory (%s)",
                    d,
                    e,
                )
                continue

    user_dir = Path.home() / ".chaperone" / "containers"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_all_containers() -> dict:
    """Get all available containers by importing all blocks (triggering auto-registration)."""
    from blocks import get_all_containers as _get_all

    return _get_all()


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[1;36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _get_block_info(block_cls) -> tuple[str | None, str]:
    """Get container name and status for a block.

    Returns:
        Tuple of (container_name, status)
    """
    image = getattr(block_cls, "_image", None)
    if image is None:
        return None, "N/A"
    try:
        path = image.get_image_path()
        return image.name, f"OK ({path.stat().st_size // 1024 // 1024}MB)"
    except FileNotFoundError:
        return image.name, "Not built"
    except Exception:
        return image.name, "Error"


def _colorize_status(status: str) -> str:
    """Add color to status string based on its value."""
    if status.startswith("OK"):
        return f"{Colors.GREEN}{status}{Colors.RESET}"
    elif status == "Not built":
        return f"{Colors.YELLOW}{status}{Colors.RESET}"
    elif status in ("Deprecated", "Error"):
        return f"{Colors.RED}{status}{Colors.RESET}"
    else:
        return f"{Colors.DIM}{status}{Colors.RESET}"


def list_blocks():
    """List all blocks and their container status, categorized by type."""
    from blocks import Blocks
    from chaperone.block_config.compiled import CompiledBlockFunction

    compiled_blocks = []
    utility_blocks = []

    for name in sorted(Blocks.keys()):
        block_cls = Blocks[name]
        container_name, status = _get_block_info(block_cls)

        if issubclass(block_cls, CompiledBlockFunction):
            compiled_blocks.append((name, container_name, status))
        else:
            utility_blocks.append((name, container_name, status))

    def print_block_table(blocks: list, header: str):
        """Print a table of blocks with header."""
        if not blocks:
            return

        max_container = min(
            20, max((len(c) if c else 1 for _, c, _ in blocks), default=10)
        )
        col_width = max(max_container, 10)

        print(f"\n{Colors.HEADER}{'─' * (20 + col_width + 20)}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD} {header}{Colors.RESET}")
        print(f"{Colors.HEADER}{'─' * (20 + col_width + 20)}{Colors.RESET}")
        print(f" {'Block':<14} {'Container':<{col_width}} Status")
        print(f" {'-' * (14 + col_width + 18)}")

        for name, container_name, status in blocks:
            colored_status = _colorize_status(status)
            container_display = container_name if container_name else "-"
            if len(container_display) > col_width:
                container_display = container_display[: col_width - 2] + ".."
            print(f" {name:<14} {container_display:<{col_width}} {colored_status}")

    print_block_table(compiled_blocks, "Compiled Blocks (external commands)")
    print_block_table(utility_blocks, "Utility Blocks (pure Python)")

    print()


def build_container(
    spec, name: str, output_dir: Path, force: bool = False, dry_run: bool = False
):
    """Build a container from Image spec.

    Uses apptainer or docker based on CONTAINER_BACKEND setting.
    """
    from chaperone import get_settings

    backend = get_settings().CHAPERONE_CONTAINER_BACKEND

    if requirement := spec._labels.get("BuildAccess"):
        print(f"  Build prerequisite: {requirement}")
    if spec.is_existing:
        print(f"  Container '{name}' references existing file: {spec._existing_path}")
        return

    if backend != "docker" and not force:
        try:
            installed = spec.get_image_path()
        except FileNotFoundError:
            pass
        else:
            print(f"  Reusing installed image: {installed}")
            return

    if backend == "docker":
        _build_docker(spec, name, output_dir, force, dry_run)
    else:
        _build_apptainer(spec, name, output_dir, force, dry_run)


def _build_apptainer(
    spec, name: str, output_dir: Path, force: bool = False, dry_run: bool = False
):
    """Build a container using apptainer."""
    import tempfile

    output_path = output_dir / f"{name}.sif"

    if output_path.exists() and not force:
        print(f"  Already exists: {output_path}")
        return

    def_path = output_dir / f"{name}.def"
    if dry_run:
        # Render without cloning or probing the container runtime. Private source
        # access and an installed runtime are not required to inspect a recipe.
        spec.to_singularity()
        print(f"  [DRY RUN] Would write definition to {def_path}")
        print(
            "  [DRY RUN] Would run: apptainer build "
            "--mksquashfs-args '-processors 2 -mem 1G' "
            f"{output_path} {def_path}"
        )
        return

    clone_dir = None
    if spec._git_clones:
        clone_dir = Path(tempfile.mkdtemp(prefix=f"chaperone-{name}-"))
        try:
            spec = spec.prepare_git_clones(clone_dir)
        except Exception:
            shutil.rmtree(clone_dir, ignore_errors=True)
            raise
    def_content = spec.to_singularity()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(def_path, "w") as f:
        f.write(def_content)
    print(f"  Definition written: {def_path}")

    # Build to a temporary file first, then move into place on success.
    tmp_path = output_dir / f"{name}-tmp.sif"

    # mksquashfs can size its buffers/threads from the entire host rather than
    # the Slurm allocation, killing a successful build during final packing.
    cmd = [
        "apptainer", "build", "--force", "--mksquashfs-args",
        "-processors 2 -mem 1G", str(tmp_path), str(def_path),
    ]

    print(f"  Building: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to build {name}: {e}")
    finally:
        if clone_dir:
            shutil.rmtree(clone_dir, ignore_errors=True)

    shutil.move(str(tmp_path), str(output_path))
    print(f"  Built: {output_path}")


def _build_docker(
    spec, name: str, output_dir: Path, force: bool = False, dry_run: bool = False
):
    """Build with an isolated context containing exactly the recipe's sources."""
    import tempfile

    tag = f"chaperone/{name}:latest"
    dockerfile_path = output_dir / f"Dockerfile.{name}"
    if dry_run:
        spec.to_dockerfile()
        print(f"  [DRY RUN] Would write Dockerfile to {dockerfile_path}")
        print(f"  [DRY RUN] Would stage recipe sources and run: docker build -t {tag} .")
        return

    if not force:
        result = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
        if result.returncode == 0:
            print(f"  Already exists: {tag}")
            return

    # Docker cannot COPY absolute host paths or files outside its context.
    # This also avoids exposing the checkout, private data or credentials to
    # the daemon merely because a recipe copies one small runner script.
    with tempfile.TemporaryDirectory(prefix=f"chaperone-{name}-") as temporary:
        workdir = Path(temporary)
        prepared = spec.prepare_git_clones(workdir / "clones")
        context = workdir / "context"
        context.mkdir()
        prepared = prepared._clone()
        staged_copies = []
        for index, (source, destination) in enumerate(prepared._copies):
            source = Path(source).expanduser().resolve(strict=True)
            relative = f"sources/{index}/{source.name}"
            staged = context / relative
            staged.parent.mkdir(parents=True)
            if source.is_dir():
                shutil.copytree(source, staged, symlinks=True)
            else:
                shutil.copy2(source, staged)
            staged_copies.append((relative, destination))
        prepared._copies = staged_copies
        definition = prepared.to_dockerfile()
        (context / "Dockerfile").write_text(definition)
        output_dir.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(definition)
        print(f"  Definition written: {dockerfile_path}")
        command = ["docker", "build", "-t", tag, str(context)]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to build {name}: {exc}") from exc
    print(f"  Built: {tag}")


def install(names: list[str], *, full=False, jobs=1, force=False, dry_run=False):
    """Build each resolved image once, with bounded parallelism and all failures reported."""
    from blocks import get_container

    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    names = list(dict.fromkeys([*(CORE_CONTAINERS if full else ()), *names]))
    requested = [(name, get_container(name)) for name in names]
    unknown = [name for name, spec in requested if spec is None]
    if unknown:
        raise app.UsageError(
            f"Unknown container: {', '.join(unknown)}. "
            f"Available: {sorted(get_all_containers())}"
        )
    # Aliases can resolve to the same output image. Never build it twice at once.
    images = {spec.name: spec for _, spec in requested}
    if dry_run:
        from chaperone.containers import get_container_dirs
        output_dir = get_container_dirs()[0]
    else:
        output_dir = get_writable_container_dir()
    print(f"\nContainer directory: {output_dir}")
    print(f"Images: {', '.join(images)}; concurrent builds: {jobs}\n")
    failures = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {
            executor.submit(build_container, spec, name, output_dir, force=force, dry_run=dry_run): name
            for name, spec in images.items()
        }
        for future in as_completed(pending):
            name = pending[future]
            try:
                future.result()
                print(f"  {name}: done")
            except Exception as exc:
                failures[name] = str(exc)
                print(f"  {name}: FAILED: {exc}")
    if failures:
        raise RuntimeError("Container builds failed: " + "; ".join(
            f"{name}: {error}" for name, error in sorted(failures.items())
        ))


def main(argv):
    names = [a for a in argv[1:] if not a.startswith("-")]

    if (not names and not FLAGS.full) or names == ["list-blocks"]:
        list_blocks()
        return 0

    if names == ["list"]:
        containers = get_all_containers()
        print("\nAvailable containers:")
        for name in sorted(containers.keys()):
            spec = containers[name]
            if spec.is_existing:
                print(f"  {name:15} (existing: {spec._existing_path})")
            else:
                requirement = spec._labels.get("BuildAccess")
                status = f"restricted: {requirement}" if requirement else "source recipe available"
                print(f"  {name:15} ({status})")
        print()
        return 0

    if FLAGS.warmup_rfd3 and not (FLAGS.warmup_cache_dir and FLAGS.warmup_run_dir):
        raise app.UsageError("--warmup_rfd3 requires --warmup_cache_dir and --warmup_run_dir")
    install(names, full=FLAGS.full, jobs=FLAGS.jobs or (2 if FLAGS.full else 1),
            force=FLAGS.force, dry_run=FLAGS.dry_run)
    if FLAGS.warmup_rfd3:
        from chaperone_cli.warmup import main as warmup_main
        warmup_args = ["--cache-dir", FLAGS.warmup_cache_dir, "--rundir", FLAGS.warmup_run_dir]
        if FLAGS.warmup_local:
            warmup_args.append("--local")
        if FLAGS.dry_run:
            warmup_args.append("--dry-run")
        warmup_main(warmup_args)
    print("Done.")


if __name__ == "__main__":
    app.run(main)
