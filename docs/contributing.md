# Contributing

The main way to contribute is by opening an issue or pull request on GitHub.

Feedback, bug reports, documentation improvements, and algorithmic ideas are welcome. If you are an author of a paper in the field or have an implementation idea for another change point detection method, please open a pull request or start with an issue.

## Local Checks

Install the test dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

## Documentation

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Serve the documentation locally:

```bash
mkdocs serve --config-file docs/mkdocs.yml
```

Build the static site:

```bash
mkdocs build --strict --config-file docs/mkdocs.yml
```
