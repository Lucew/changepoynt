# Documentation Manual

Run all commands from the repository root.

## Set Up the Environment

Create a virtual environment and install the package with its documentation dependencies:

```bash
uv venv .venv
uv pip install -e ".[docs]" --python .venv/Scripts/python.exe
```

On Linux or macOS, use `.venv/bin/python` instead.

## Build the Documentation

```bash
uv run --no-sync mkdocs build --strict --config-file docs/mkdocs.yml
```

The generated static website is written to `site/`. A strict build fails on invalid documentation configuration or unresolved internal links.

## Serve It Locally

```bash
uv run --no-sync mkdocs serve --config-file docs/mkdocs.yml
```

Open [http://127.0.0.1:8000/changepoynt/](http://127.0.0.1:8000/changepoynt/). MkDocs watches the documentation files and rebuilds the local preview after changes.

Stop the server with `Ctrl+C`.

## Deploy with GitHub Actions

Pushes to `master` or `main` automatically deploy when files under `docs/`, `changepoynt/`, or the documentation workflow change.

To deploy manually:

1. Open the repository's **Actions** tab on GitHub.
2. Select the **Docs** workflow.
3. Choose **Run workflow**.
4. Select the branch and confirm the run.

The workflow builds the `site/` directory, uploads it as a GitHub Pages artifact, and deploys that artifact to the `github-pages` environment.
