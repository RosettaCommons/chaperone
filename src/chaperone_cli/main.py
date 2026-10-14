#!/usr/bin/env python3
"""Main CLI dispatcher for Chaperone commands.

Usage:
    chaperone install mpnn rfd3 af3
"""

import sys


def _initialize_runtime() -> None:
    """Load the scientific runtime only for commands that need it."""
    import dotenv

    from chaperone.common import setup_logging

    dotenv.load_dotenv()
    setup_logging()


def list_available():
    """List all available containers."""
    _initialize_runtime()
    from blocks import get_all_containers

    registry = get_all_containers()
    print("\n Available containers:\n")
    for name, spec in sorted(registry.items()):
        if spec.is_existing:
            desc = f"existing: {spec._existing_path}"
        else:
            desc = f"Image (base: {spec._base_image})"
        print(f"  {name:15} - {desc}")
    print()


def list_installed():
    """List installed containers."""
    _initialize_runtime()
    from chaperone import get_settings
    from chaperone.containers import get_container_dirs
    from blocks import get_all_containers

    backend = get_settings().CHAPERONE_CONTAINER_BACKEND
    container_names = set(get_all_containers().keys())

    print(f"\n Container backend: {backend}")

    if backend == "docker":
        import subprocess

        print("\n Installed containers:\n")

        installed = []
        for name in sorted(container_names):
            tag = f"chaperone/{name}:latest"
            result = subprocess.run(
                ["docker", "image", "inspect", tag],
                capture_output=True,
            )
            if result.returncode == 0:
                import json

                info = json.loads(result.stdout)
                size_mb = info[0].get("Size", 0) / (1024 * 1024) if info else 0
                print(f"  {name:15} - {tag} ({size_mb:.1f} MB)")
                installed.append(name)
    else:
        search_dirs = get_container_dirs()

        print("\n Container search paths:")
        for dir in search_dirs:
            exists = "exists" if dir.exists() else "not found"
            print(f"  - {dir} ({exists})")

        print("\n Installed containers:\n")

        installed = []
        for name in sorted(container_names):
            for search_dir in search_dirs:
                container_path = search_dir / f"{name}.sif"
                if container_path.exists():
                    size_mb = container_path.stat().st_size / (1024 * 1024)
                    print(f"  {name:15} - {container_path} ({size_mb:.1f} MB)")
                    installed.append(name)
                    break

    if not installed:
        print("  No containers installed yet.")
        print("\n  Install with: chaperone install <container>")
    print()


def find_container(name: str):
    """Find a container by name, returns path/tag or None."""
    _initialize_runtime()
    from chaperone import get_settings

    backend = get_settings().CHAPERONE_CONTAINER_BACKEND

    if backend == "docker":
        import subprocess

        tag = f"chaperone/{name}:latest"
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        )
        if result.returncode == 0:
            return tag
        return None

    from chaperone.containers import get_container_dirs

    for search_dir in get_container_dirs():
        container_path = search_dir / f"{name}.sif"
        if container_path.exists():
            return container_path
    return None


def shell_container(name: str):
    """Open an interactive shell in a container."""
    import subprocess
    from chaperone import get_settings

    container_ref = find_container(name)

    if container_ref is None:
        from blocks import get_all_containers

        registry = get_all_containers()
        print(f"Error: Container '{name}' not found.")
        if name in registry:
            print("  Container is available but not built.")
            print(f"  Install with: chaperone install {name}")
        else:
            print(f"  Available containers: {sorted(registry.keys())}")
        sys.exit(1)

    backend = get_settings().CHAPERONE_CONTAINER_BACKEND

    print(f"Entering container: {container_ref}")
    print("Type 'exit' to leave the container.\n")

    if backend == "docker":
        cmd = ["docker", "run", "-it", "--gpus", "all", str(container_ref), "/bin/bash"]
    else:
        cmd = ["apptainer", "shell", "--nv", str(container_ref)]

    subprocess.run(cmd)


def print_help():
    """Print help message."""
    print("""
chaperone - Protein design pipeline tools

Usage:
    chaperone <command> [options]

Commands:
    list-available    List all available containers
    list-installed    List currently installed containers
    list-blocks       List all blocks and their container status
    install           Install containers for tools (--full --jobs=2)
    warmup-rfd3       Populate a persistent RFD3 GPU compilation cache
    shell             Enter an interactive shell in a container
    artifact          Read a bounded log/artifact excerpt with a continuation cursor
    wait              Sleep until a Slurm job, process, or marker finishes
    monitor-mcp       Serve Chaperone monitoring tools over MCP stdio
    help              Show this help message

Aliases:
    list              Same as list-available

Examples:
    # Container management:
    chaperone list-available          # Show available containers
    chaperone list-installed          # Show installed containers
    chaperone list-blocks             # Show blocks and their containers
    chaperone install foundry e1      # Install containers (--force, --dry_run)
    chaperone shell foundry           # Enter interactive shell in container

For command-specific help:
    chaperone install --help
""")


def main():
    """Main entry point for the Chaperone CLI."""
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_help()
        sys.exit(0)

    command = sys.argv[1]

    if command in ["list", "list-available"]:
        list_available()
    elif command == "list-installed":
        list_installed()
    elif command == "list-blocks":
        _initialize_runtime()
        from chaperone_cli.install import list_blocks

        list_blocks()
    elif command == "install":
        _initialize_runtime()
        from chaperone_cli.install import main as install_main

        # Remove 'install' from argv so absl.app sees the right arguments
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from absl import app

        app.run(install_main)
    elif command == "warmup-rfd3":
        from chaperone_cli.warmup import main as warmup_main

        sys.exit(warmup_main(sys.argv[2:]))
    elif command == "shell":
        if len(sys.argv) < 3:
            print("Usage: chaperone shell <container_name>")
            print("Example: chaperone shell foundry")
            sys.exit(1)
        shell_container(sys.argv[2])
    elif command == "artifact":
        from chaperone_cli.artifact_cmd import main as artifact_main

        sys.exit(artifact_main(sys.argv[2:]))
    elif command == "wait":
        from chaperone_cli.job_wait import wait_cli

        sys.exit(wait_cli(sys.argv[2:]))
    elif command == "monitor-mcp":
        from chaperone_cli.job_wait import mcp_cli

        sys.exit(mcp_cli(sys.argv[2:]))
    else:
        print(f"Unknown command: {command}")
        print("Run 'chaperone help' for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
