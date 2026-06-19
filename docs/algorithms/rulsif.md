# RuLSIF

`RuLSIF` compares the distributions of subsequences before and after a candidate position. It directly estimates a relative density ratio and converts the relative Pearson divergence into a change score.

## Parameters

`window_length` : `int`, default `10`
: Length of each subsequence used as one density-estimation sample.

`n_windows` : `int`, default `50`
: Number of subsequences in each compared group.

`lag` : `int`, optional
: Separation between comparison regions. Defaults to `n_windows`.

`estimation_lag` : `int`, optional
: Reserved for controlling hyperparameter re-estimation. It is currently stored but not used.

`scoring_step` : `int`, default `1`
: Distance between evaluated positions.

`n_kernels` : `int`, default `100`
: Reserved kernel-count setting. It is currently stored but not passed to the estimator.

`alpha` : `float`, default `0.01`
: Relative-density smoothing in `[0, 1)`. Values above zero bound the influence of very large ordinary density ratios.

`symmetric` : `bool`, default `True`
: Sum forward and reverse density-ratio scores.

`parallel` : `bool`, default `False`
: Compute the two symmetric directions in separate worker processes.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: One-dimensional signal.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Density-change score with zero padding where a full comparison is unavailable.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.rulsif import RuLSIF

rng = np.random.default_rng(4)
signal = np.r_[rng.normal(0, 1, 40), rng.normal(0, 2, 40)]
detector = RuLSIF(window_length=3, n_windows=5,
                  scoring_step=5, symmetric=False)
score = detector.transform(signal)
```

**Reference:** [Liu et al., Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation](https://arxiv.org/abs/1203.0453).
