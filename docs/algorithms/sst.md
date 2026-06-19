# SST

`SST` applies Singular Spectrum Transformation to a one-dimensional signal. It compares low-rank representations of past and future Hankel matrices and returns a structural-change score.

Use the [tuning guide](../guides/tuning-subspace-methods.md) for a parameter-by-parameter workflow and runtime advice.

## Parameters

`window_length` : `int`
: Subsequence length in samples. This is the main temporal scale.

`n_windows` : `int`, optional
: Number of subsequences in each Hankel matrix. Defaults to `window_length`.

`lag` : `int`, optional
: Separation between compared matrices. Defaults to `max(n_windows // 3, 1)`.

`rank` : `int`, default `5`
: Number of dominant past-subspace directions.

`scale` : `bool`, default `True`
: Min-max scale the signal to `[1, 2]` before scoring.

`method` : `str`, default `"ika"`
: Scoring/decomposition method: `"ika"`, `"svd"`, `"rsvd"`, `"fbrsvd"`, `"naive"`, `"naive updated"`, `"weighted"`, or `"symmetric"`.

`lanczos_rank` : `int`, optional
: Krylov approximation size for `method="ika"`. Derived from `rank` when omitted.

`random_rank` : `int`, optional
: Sampled dimension for randomized methods. Defaults to at most `rank + 10`.

`feedback_noise_level` : `float`, default `1e-3`
: Noise added to the recycled future direction used by IKA-style scoring.

`scoring_step` : `int`, default `1`
: Distance in samples between evaluated positions.

`use_fast_hankel` : `bool`, default `False`
: Use implicit FFT-based Hankel products where the selected method supports them.

`mitigate_offset` : `bool`, default `False`
: Remove column-wise offsets. Cannot be combined with fast Hankel products.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: Signal to score.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Structural-change score. Initial positions without enough context are zero.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.sst import SST

signal = np.r_[np.sin(np.linspace(0, 8 * np.pi, 200)),
               np.sin(np.linspace(0, 20 * np.pi, 200))]
score = SST(window_length=40, method="rsvd").transform(signal)
```

**References:** [Ide and Inoue (2005)](https://epubs.siam.org/doi/10.1137/1.9781611972757.63), [Ide and Tsuda (2007)](https://epubs.siam.org/doi/10.1137/1.9781611972771.54).
