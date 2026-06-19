# MSST

`MSST` extends SST to multivariate signals by forming a block-Hankel matrix from all channels. Use it when changes are expressed by joint temporal behavior rather than one channel alone.

Use the [tuning guide](../guides/tuning-subspace-methods.md) for a parameter-by-parameter workflow and runtime advice.

## Parameters

`window_length` : `int`
: Subsequence length in samples.

`n_windows` : `int`, optional
: Number of subsequences in each block-Hankel matrix. Defaults to `window_length`.

`lag` : `int`, optional
: Separation between compared matrices. Defaults to `max(n_windows // 3, 1)`.

`rank` : `int`, default `5`
: Number of dominant joint directions.

`scale` : `bool`, default `True`
: Min-max scale every channel independently to `[1, 2]`.

`method` : `str`, default `"ika"`
: One of `"ika"`, `"rsvd"`, `"weighted"`, or `"symmetric"`.

`lanczos_rank` : `int`, optional
: Krylov approximation size for IKA.

`random_rank` : `int`, optional
: Sampled dimension for randomized methods.

`feedback_noise_level` : `float`, default `1e-3`
: Noise added to the recycled future direction.

`scoring_step` : `int`, default `1`
: Distance between evaluated positions.

`use_fast_hankel` : `bool`, default `False`
: Select fast block-Hankel products. The current implementation uses the fast representation internally for scoring regardless of this value.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples, n_channels)`
: Multivariate signal with samples on axis 0.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Joint structural-change score.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.msst import MSST

t = np.linspace(0, 12 * np.pi, 400)
signal = np.column_stack((np.sin(t), np.cos(t)))
signal[200:, 1] = np.cos(2 * t[200:])
score = MSST(window_length=40, method="rsvd").transform(signal)
```

**References:** [Ide and Inoue (2005)](https://epubs.siam.org/doi/10.1137/1.9781611972757.63), [Ide and Tsuda (2007)](https://epubs.siam.org/doi/10.1137/1.9781611972771.54).
