# FLUSS

`FLUSS` segments a completed one-dimensional signal using matrix-profile nearest-neighbor arcs. Regime boundaries tend to have fewer arcs crossing them, which produces peaks in the returned `1 - corrected_arc_curve` score.

## Parameters

`window_length` : `int`
: Subsequence length used by the matrix profile. Choose it near the shortest recurring pattern that should define a regime.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: Complete one-dimensional signal.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples - window_length + 1,)`
: Inverted corrected arc curve. Larger values indicate candidate regime boundaries.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.fluss import FLUSS

signal = np.r_[np.sin(np.linspace(0, 8 * np.pi, 200)),
               np.sin(np.linspace(0, 24 * np.pi, 200))]
score = FLUSS(window_length=30).transform(signal)
```

**Reference:** [Gharghabi et al., Matrix Profile VIII](https://doi.org/10.1109/ICDM.2017.21).
