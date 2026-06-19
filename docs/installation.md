# Installation

Install `changepoynt` from PyPI:

```bash
pip install changepoynt
```

If you use `uv`, install it with:

```bash
uv pip install changepoynt
```

## Install from Source

Install the current repository version directly from GitHub:

```bash
pip install git+https://github.com/Lucew/changepoynt.git
```

For local development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Lucew/changepoynt.git
cd changepoynt
pip install -e .
```

## Documentation Dependencies

To build this documentation locally, install the `docs` extra:

```bash
pip install -e ".[docs]"
```

Then serve the docs:

```bash
mkdocs serve --config-file docs/mkdocs.yml
```
