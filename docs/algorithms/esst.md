# ESST

`ESST` entangles past and future Hankel matrices before decomposition and scores how their contributions differ. It is a one-dimensional research method intended for changes in local signal structure.

Use the [tuning guide](../guides/tuning-subspace-methods.md) for a parameter-by-parameter workflow and runtime advice.

## Parameters

`window_length` : `int`
: Subsequence length in samples and main temporal scale.

`n_windows` : `int`, optional
: Number of subsequences per side. Defaults to `window_length // 2`.

`lag` : `int`, optional
: Separation between compared matrices. Defaults to `n_windows`.

`rank` : `int`, default `5`
: Number of singular directions used by the score.

`scale` : `bool`, default `True`
: Min-max scale the signal to `[1, 2]`.

`method` : `{"fbrsvd", "rsvd"}`, default `"fbrsvd"`
: Randomized decomposition implementation. Use `"rsvd"` with fast Hankel products.

`random_rank` : `int`, optional
: Sampled randomized-SVD dimension. Defaults to at most `rank + 10`.

`scoring_step` : `int`, default `1`
: Distance in samples between evaluated positions.

`use_fast_hankel` : `bool`, default `False`
: Use implicit FFT-based Hankel products. Requires `method="rsvd"`.

`mitigate_offset` : `bool`, default `False`
: Remove column-wise offsets. Cannot be combined with fast Hankel products.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: Signal to score.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Entangled subspace-change score with an initial zero-padded region.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.esst import ESST

signal = np.r_[np.sin(np.linspace(0, 8 * np.pi, 200)),
               np.sin(np.linspace(0, 20 * np.pi, 200))]
score = ESST(window_length=40).transform(signal)
```

**Reference:** [Boelter et al. (2025)](https://ntrs.nasa.gov/citations/20250002705).
