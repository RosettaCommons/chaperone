---
name: visualize
description: "Visualize structures (CIF, PDB, pkl) in PyMOL via its XML-RPC server. Use when the user says 'visualize', 'show structure', 'view in pymol', 'viz', 'show cartoon', 'color this', or wants to send a structure or PyMOL commands to their PyMOL session."
allowed-tools: ["Bash", "Read", "Write", "Glob", "Grep", "AskUserQuestion"]
---

# Visualize — Send structures and commands to PyMOL

There are two connection modes. Use whichever the user has set up:

- **Local** (`pymol -R`): No extra deps. `ServerProxy("http://localhost:{PORT}")` → `cmd.do(...)`, `cmd.load(...)`. User runs `pymol -R` (or `launch_pymol.sh`) in a separate terminal.
- **Remote** (`pymol_remote` + SSH): Needs `pymol_remote` installed in PyMOL's Python on the user's laptop + SSH reverse tunnel. `get_pymol_session()` → `pymol.do(...)`, `view_pymol(...)`. See README.md for setup. The `visualize.py` script in this directory wraps this mode.

Both modes use the same port range (9123-9127) and the same `cmd.do("...")` syntax for sending commands. The difference is how structures are loaded (file path vs. serialized over RPC) and what's needed on the user's end.

## Step 0: Find the PyMOL port

PyMOL's RPC server can be on **any port from 9123-9127** depending on how many instances are running. Always detect the port first:

```bash
python3 -c "
from xmlrpc.client import ServerProxy
for port in range(9123, 9128):
    try:
        s = ServerProxy(f'http://localhost:{port}')
        s.do('python pass')
        print(f'PyMOL on port {port}')
    except: pass
"
```

If nothing responds, the user needs to launch PyMOL. Tell them to run in a separate terminal:
```bash
pymol -R          # GUI with RPC
```

OR if they wish to, refer them to the README for details on how to establish the connection from a pymol session on a remote machine.

**Multiple ports:** When multiple PyMOL instances are detected, send to **all** of them — the user may not know which window they're looking at. Use `get_version()` to confirm each is a real session.

## Sending commands

Once you know the port, use `xmlrpc.client.ServerProxy`:

**Important:** Always use bash heredoc syntax (`<< 'PYEOF'`) for Python scripts sent via Bash tool — inline `python3 -c "..."` breaks on PyMOL command strings containing quotes and special characters.

**PREFERRED: Native RPC methods.** These call PyMOL functions directly and are reliable across all setups. Most PyMOL commands have a corresponding RPC method with the same name and arguments as positional strings.

```python
from xmlrpc.client import ServerProxy
s = ServerProxy("http://localhost:{PORT}")

# PREFERRED: Native RPC methods — reliable, no parsing issues
s.show("cartoon", "all")
s.color("red", "chain A")
s.bg_color("white")
s.remove("solvent")
s.spectrum("b", "blue_white_red", "all")
s.ray(1200, 900)
s.delete("all")
s.select("my_sel", "resi 35+52")
s.zoom("my_sel", 8)
s.orient()
s.hide("everything", "my_sel")
s.set("cartoon_transparency", 0.5, "all")

# FALLBACK: s.do() — only if a command has no native RPC equivalent.
# WARNING: s.do() can silently fail depending on PyMOL version and connection
# mode (echoes commands but doesn't execute them). Do NOT use semicolons
# with s.do() — they can cause parse failures.
# s.do("spectrum b, blue_white_red, all")
```

### Loading structures

**Prefer `read_pdbstr`** over `fetch` or `cmd.load`. PyMOL's `fetch` can fail silently if the PyMOL process lacks internet access (common with remote/tunneled sessions). `cmd.load` requires the file path to be accessible to the PyMOL process, which may not hold across SSH tunnels. `read_pdbstr` sends the file content directly over RPC — it always works.

**Critical: `read_pdbstr` only accepts PDB format.** There is no `read_cifstr` RPC method. For CIF files, convert to PDB first (see below). Sending CIF content to `read_pdbstr` will crash PyMOL.

```python
# PDB files: send directly
with open("/path/to/structure.pdb") as f:
    s.read_pdbstr(f.read(), "my_obj")

# CIF files: convert to PDB first via biotite
import io
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

cif = pdbx.CIFFile.read("/path/to/structure.cif")
atom_array = pdbx.get_structure(cif, model=1)
pdb_file = pdb.PDBFile()
pdb_file.set_structure(atom_array)
buf = io.StringIO()
pdb_file.write(buf)
s.read_pdbstr(buf.getvalue(), "my_obj")

# Fetching from RCSB: download PDB format locally, then send
import urllib.request
urllib.request.urlretrieve("https://files.rcsb.org/download/1QYS.pdb", "/tmp/1qys.pdb")
with open("/tmp/1qys.pdb") as f:
    s.read_pdbstr(f.read(), "my_obj")

# AVOID: fetch can fail silently in remote/tunneled PyMOL sessions
# s.do("fetch 1qys;")
```

### Loading from `.catcif` archives

Chaperone can consolidate step outputs into `.catcif` archives (see README → *Storage formats and cleanups*). Handler paths look like `/path/to/archive.catcif:entry_id`.

Resolve each archive entry with Chaperone, then send it to PyMOL via
`read_pdbstr` (convert CIF to PDB as shown above). This uses the installed CatCIF
reader and requires no site-specific PyMOL plugin.

```python
from chaperone.io import parse_to_atom_array
atom_array = parse_to_atom_array("/path/to/archive.catcif:design_0")
# ... serialize to PDB and send via s.read_pdbstr ...
```

## Sending atom arrays from Chaperone

To send a biotite `AtomArray` (e.g. from `parse_to_atom_array` or a block output) directly to PyMOL, serialize it to **PDB format** and send via `read_pdbstr`. Do NOT serialize to CIF — there is no `read_cifstr` RPC method.

```python
import io
import biotite.structure.io.pdb as pdb
from xmlrpc.client import ServerProxy
from chaperone.io import parse_to_atom_array

s = ServerProxy("http://localhost:{PORT}")

atom_array = parse_to_atom_array("/path/to/structure.cif")

# Serialize atom array to PDB string (NOT CIF)
pdb_file = pdb.PDBFile()
pdb_file.set_structure(atom_array)
buf = io.StringIO()
pdb_file.write(buf)

# Send to PyMOL
s.read_pdbstr(buf.getvalue(), "my_design")
```

This works with any `AtomArray` — from `parse_to_atom_array`, from block outputs, or constructed in code. For handler data:

```python
structures, metrics = handler[0]
for struct in structures:
    atom_array = parse_to_atom_array(struct["path"])
    # ... serialize to PDB and send as above
```

## Recommendations for custom visualizations

**Keep it minimal.** Don't over-specify visualizations — avoid setting custom colors, transparency, dash styles, label fonts, label sizes, etc. Trust that the default styling is already set in the user's `.pymolrc`. Only send the structural commands (load, show, select, orient) and let PyMOL's defaults handle the rest. If the user asks for something specific (e.g. "color chain A red"), do that and nothing more.
