---
name: notebook-editing
description: Use when editing, creating, or executing Jupyter notebooks (.ipynb files). Activates for any notebook interaction including reading, modifying cells, converting formats, or running notebooks.
---

# Notebook Editing Skill

**The `.ipynb` is always the source of truth.** Never maintain a separate `.md` file alongside it.

## Editing Notebooks

To edit a notebook:

1. Convert to a **temporary** md: `jupytext --to md notebook.ipynb --output /tmp/notebook.md`
2. Read and edit `/tmp/notebook.md`
3. Convert back with `--update` to preserve cell outputs/ids: `jupytext --to ipynb /tmp/notebook.md --output notebook.ipynb --update`
4. Delete the temp md: `rm /tmp/notebook.md`

### Why This Workflow

- Jupytext markdown is easier to read and edit than raw `.ipynb` JSON
- The `--update` flag preserves cell IDs, outputs, and metadata from the original notebook
- Using `/tmp/` ensures no stale `.md` files linger in the project directory
- The `.ipynb` remains the canonical file checked into version control

### Important Notes

- **Never** commit or keep the temporary `.md` file — always delete it after converting back
- If the notebook doesn't exist yet (creating from scratch), write the `.md` first then convert without `--update`
- For very large notebooks, you can read the `.ipynb` directly first (Claude can parse notebook JSON natively) to understand structure before converting

## Executing Notebooks

Use `papermill` (not `nbconvert --execute`, which has kernel/dependency issues):

```bash
source .venv/bin/activate && papermill notebook.ipynb notebook.ipynb
```

- Papermill executes in-place when input and output paths are the same
- It preserves the notebook structure and adds execution outputs
- Always activate the virtualenv first to ensure correct dependencies
