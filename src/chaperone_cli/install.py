#!/usr/bin/env python3
"""CLI for installing containers for pipeline blocks.

Usage:
    chaperone install foundry e1
    chaperone install e1 --dry_run
    chaperone install e1 --force
"""

import shutil
import subprocess
from pathlib import Path
from absl import app, flags
import logging

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_boolean("force", False, "Force rebuild even if container exists")
flags.DEFINE_boolean("dry_run", False, "Show what would be done without doing it")


def get_writable_container_dir() -> Path:
    """Get the writable container directory."""
    from chaperone.containers import get_container_dirs

    dirs = get_container_dirs()
    for d in dirs:
        if d.exists() and d.is_dir():
            try:
                test_file = d / ".write_test"
                test_file.touch()
                test_file.unlink()
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

    if spec.is_existing:
        print(f"  Container '{name}' references existing file: {spec._existing_path}")
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

    # Pre-clone any git repos on the host so they can be injected via %files
    clone_dir = None
    if spec._git_clones:
        clone_dir = Path(tempfile.mkdtemp(prefix=f"chaperone-{name}-"))
        spec = spec.prepare_git_clones(clone_dir)

    def_content = spec.to_singularity()
    def_path = output_dir / f"{name}.def"

    if dry_run:
        print(f"  [DRY RUN] Would write definition to {def_path}")
        print(f"  [DRY RUN] Would run: apptainer build {output_path} {def_path}")
        if clone_dir:
            shutil.rmtree(clone_dir, ignore_errors=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(def_path, "w") as f:
        f.write(def_content)
    print(f"  Definition written: {def_path}")

    # Build to a temporary file first, then move into place on success.
    tmp_path = output_dir / f"{name}-tmp.sif"

    cmd = ["apptainer", "build", "--force", str(tmp_path), str(def_path)]

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
    """Build a container using docker."""
    import tempfile

    tag = f"chaperone/{name}:latest"

    if not force:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"  Already exists: {tag}")
            return

    # Pre-clone any git repos on the host so they can be injected via COPY
    clone_dir = None
    if spec._git_clones:
        clone_dir = Path(tempfile.mkdtemp(prefix=f"chaperone-{name}-"))
        spec = spec.prepare_git_clones(clone_dir)

    dockerfile_content = spec.to_dockerfile()
    dockerfile_path = output_dir / f"Dockerfile.{name}"

    if dry_run:
        print(f"  [DRY RUN] Would write Dockerfile to {dockerfile_path}")
        print(f"  [DRY RUN] Would run: docker build -t {tag} -f {dockerfile_path} .")
        if clone_dir:
            shutil.rmtree(clone_dir, ignore_errors=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)
    print(f"  Dockerfile written: {dockerfile_path}")

    cmd = ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(output_dir)]

    print(f"  Building: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to build {name}: {e}")
    finally:
        if clone_dir:
            shutil.rmtree(clone_dir, ignore_errors=True)

    print(f"  Built: {tag}")


def install(names: list[str]):
    """Install containers by name."""
    from blocks import get_container

    output_dir = get_writable_container_dir()
    print(f"\nContainer directory: {output_dir}\n")

    for name in names:
        spec = get_container(name)
        if spec is None:
            all_containers = get_all_containers()
            print(f"Unknown container: {name}")
            print(f"Available: {sorted(all_containers.keys())}")
            continue

        print(f"Installing {name}...")
        build_container(
            spec, name, output_dir, force=FLAGS.force, dry_run=FLAGS.dry_run
        )
        print("  Done.\n")


def main(argv):
    names = [a for a in argv[1:] if not a.startswith("-")]

    if not names or names == ["list-blocks"]:
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
                print(f"  {name:15} (buildable from Image)")
        print()
        return 0

    install(names)
    print("Done.")


if __name__ == "__main__":
    app.run(main)
