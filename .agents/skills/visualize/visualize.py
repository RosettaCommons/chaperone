#!/usr/bin/env -S bash -c '"${CHAPERONE_REPO:-.}/.venv/bin/python" "$0" "$@"'
"""Send structures and commands to PyMOL via its XML-RPC server.

Works with both native PyMOL -R and pymol_remote. Auto-detects port 9123-9127.

Usage:
    ./visualize.py check
    ./visualize.py cmd "show cartoon, all" "color red, chain A"
    ./visualize.py viz /path/to/structure.cif
"""

import pathlib
import sys
import xmlrpc.client

import fire


def _find_pymol(host: str = "localhost"):
    """Find a running PyMOL RPC server on host:9123-9127. Returns (proxy, port)."""
    for port in range(9123, 9128):
        try:
            s = xmlrpc.client.ServerProxy(f"http://{host}:{port}")
            s.get_version()
            return s, port
        except Exception:
            continue
    return None, None


def check(host: str = "localhost"):
    """Check PyMOL RPC connection."""
    s, port = _find_pymol(host)
    if s:
        print(f"Connected to PyMOL on {host}:{port}")
    else:
        print(f"No PyMOL RPC on {host}:9123-9127")
        print("Launch with: pymol -R  (or ./launch_pymol.sh)")


def cmd(*commands: str, host: str = "localhost"):
    """Send arbitrary PyMOL commands."""
    s, port = _find_pymol(host)
    if not s:
        print("No PyMOL RPC found.", file=sys.stderr)
        sys.exit(1)
    for c in commands:
        # s.do() requires trailing semicolons to execute
        c_safe = c if c.rstrip().endswith(";") else c + ";"
        print(f">>> {c_safe}")
        s.do(c_safe)
    print(f"Sent {len(commands)} command(s) to PyMOL on port {port}.")


def viz(
    file_path: str,
    id: str | None = None,
    clear: bool = True,
    style: bool = True,
    host: str = "localhost",
):
    """Send a structure to PyMOL.

    Tries atomworks view_pymol first (richer serialization), falls back to cmd.load.
    """
    path = pathlib.Path(file_path).resolve()
    if not path.exists():
        print(f"Error: {file_path} does not exist", file=sys.stderr)
        sys.exit(1)

    if id is None:
        id = path.stem.replace(".", "_")

    s, port = _find_pymol(host)
    if not s:
        print("No PyMOL RPC found.", file=sys.stderr)
        sys.exit(1)

    if clear:
        s.do("delete all;")

    # Try atomworks view_pymol (handles CIF serialization over RPC)
    try:
        from atomworks.io.utils.visualize import view_pymol

        from chaperone.io import parse_to_atom_array

        atom_array = parse_to_atom_array(str(path))
        import biotite.structure as struc

        print(
            f"Loaded: {atom_array.array_length()} atoms, "
            f"chains: {list(struc.get_chains(atom_array))}"
        )
        view_pymol(atom_array, id=id, hostname=host, port=port)
    except Exception:
        # Fallback: direct load (works if PyMOL can access the file path)
        s.load(str(path), id)

    print(f"Sent to PyMOL as: {id}")

    if style:
        s.do(f"show cartoon, {id};")
        s.do(f"spectrum resi, blue_white_red, {id};")
        print("Styling applied.")


def bfactor(
    selection: str = "all",
    minimum: float = 0,
    maximum: float = 100,
    palette: str = "blue_white_red",
    host: str = "localhost",
):
    """Color selection by B-factor on an absolute scale (default 0–100).

    Uses blue→white→red: low B-factor = blue (ordered), high = red (disordered).
    Pass minimum/maximum to adjust the clamp range.
    """
    s, port = _find_pymol(host)
    if not s:
        print("No PyMOL RPC found.", file=sys.stderr)
        sys.exit(1)
    command = (
        f"spectrum b, {palette}, {selection}, minimum={minimum}, maximum={maximum};"
    )
    print(f">>> {command}")
    s.do(command)
    print(f"B-factor coloring applied ({minimum}–{maximum}) on port {port}.")


if __name__ == "__main__":
    fire.Fire({"check": check, "cmd": cmd, "viz": viz, "bfactor": bfactor})
