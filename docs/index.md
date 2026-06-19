# changepoynt

**changepoynt** is a Python package for change point detection in time series. It includes subspace-based methods such as SST, IKA-SST, ESST, and MESST, density-ratio methods such as uLSIF and RuLSIF, and matrix-profile segmentation methods such as FLUSS and FLOSS.

The package focuses on efficient implementations with readable code, so it can be useful both for research experiments and practical time-series workflows.

## Start Here

Install the package:

```bash
pip install changepoynt
```

Then run a detector on a one-dimensional time series:

```python
import numpy as np
from changepoynt.algorithms.esst import ESST

signal = np.concatenate([
    np.ones(200),
    np.exp(-np.linspace(0, 5, 200)),
    np.exp(-5) * np.ones(150),
    0.2 * np.sin(np.linspace(0, 30 * np.pi, 300)),
])

detector = ESST(window_length=40)
score = detector.transform(signal)
```

See the [quickstart](quickstart.md) for a complete plotted example.

## Documentation Map

- [Installation](installation.md) covers PyPI and source installs.
- [Quickstart](quickstart.md) shows a minimal end-to-end example.
- [Algorithms](algorithms.md) summarizes implemented methods and their status.
- [User Guides](guides/index.md) explains how to choose and configure related method families.
- [API Reference](api/index.md) is generated from package docstrings.
- [FAQ](FAQ.md) answers common parameter and performance questions.
