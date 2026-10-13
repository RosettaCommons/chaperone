# Documentation publishing

The documentation lives in `docs/site/` and is published at
<https://rosettacommons.github.io/chaperone/>.

Build or preview it without installing the scientific runtimes:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The Documentation workflow checks pull requests and publishes documentation
changes merged into `main`. It can also be run manually on `main` from GitHub
Actions. Repository Settings → Pages must use **GitHub Actions** as its source.

`blocks-reference.md` is checked in so publishing does not need GPUs, containers,
private dependencies, or model weights. After changing block parameters,
regenerate it in a configured Chaperone development environment with
`python docs/site/gen_block_docs.py`, then commit the updated Markdown.
