# ULSIF

`ULSIF` is the `alpha=0` specialization of the package's RuLSIF detector. It estimates an ordinary density ratio between neighboring groups of subsequences using unconstrained least-squares importance fitting.

## Parameters

`window_length` : `int`, default `10`
: Length of each subsequence used as one density-estimation sample.

`n_windows` : `int`, default `50`
: Number of subsequences in each compared group.

`lag` : `int`, optional
: Separation between comparison regions. The wrapped RuLSIF detector defaults to `n_windows`.

`estimation_lag` : `int`, optional
: Reserved for hyperparameter re-estimation; currently not used by the wrapped estimator.

`scoring_step` : `int`, default `1`
: Intended distance between evaluated positions. The current wrapper stores but does not forward this value.

`n_kernels` : `int`, default `100`
: Reserved kernel-count setting; currently not consumed by the wrapped estimator.

`symmetric` : `bool`, default `True`
: Sum forward and reverse density-ratio scores.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: One-dimensional signal.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Density-change score returned by RuLSIF with `alpha=0`.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.ulsif import ULSIF

rng = np.random.default_rng(4)
signal = np.r_[rng.normal(0, 1, 40), rng.normal(0, 2, 40)]
score = ULSIF(window_length=3, n_windows=5,
              symmetric=False).transform(signal)
```

**References:** [Kanamori, Hido, and Sugiyama (2009)](https://www.jmlr.org/papers/v10/kanamori09a.html), [Liu et al. (2013)](https://arxiv.org/abs/1203.0453).
