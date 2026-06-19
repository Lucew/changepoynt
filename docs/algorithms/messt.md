# MESST

`MESST` applies the entangled ESST score to multivariate block-Hankel matrices. It is intended for structural changes involving several channels.

Use the [tuning guide](../guides/tuning-subspace-methods.md) for a parameter-by-parameter workflow and runtime advice.

## Parameters

`window_length` : `int`
: Subsequence length in samples.

`n_windows` : `int`, optional
: Number of subsequences per side. Defaults to `window_length // 2`.

`lag` : `int`, optional
: Separation between compared matrices. Defaults to `n_windows`.

`rank` : `int`, default `5`
: Number of joint singular directions used by the score.

`scale` : `bool`, default `True`
: Min-max scale every channel independently to `[1, 2]`.

`method` : `{"rsvd"}`, default `"rsvd"`
: Decomposition method. Only randomized SVD is currently available.

`random_rank` : `int`, optional
: Sampled randomized-SVD dimension. Defaults to at most `rank + 10`.

`scoring_step` : `int`, default `1`
: Distance between evaluated positions.

`use_fast_hankel` : `bool`, default `False`
: Use the implicit fast block-Hankel representation.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples, n_channels)`
: Multivariate signal with samples on axis 0.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Joint entangled subspace-change score.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.messt import MESST

t = np.linspace(0, 12 * np.pi, 400)
signal = np.column_stack((np.sin(t), np.cos(t)))
signal[200:, 1] = np.cos(2 * t[200:])
score = MESST(window_length=40).transform(signal)
```

**Reference:** [Boelter et al. (2025)](https://ntrs.nasa.gov/citations/20250002705).
