# BOCPD

`BOCPD` performs Bayesian online change-point detection for a Gaussian signal with unknown mean and known observation variance. It tracks a posterior over run length, the number of samples since the latest change.

The current implementation stores a quadratic run-length matrix, so memory and runtime grow quickly with signal length.

## Parameters

`run_length` : `int`
: Prior expected segment length. The implementation uses a constant hazard of `1 / run_length`.

`prior_mean` : `float`, optional
: Gaussian prior mean. Estimated from sliding-window means during `fit()` when omitted.

`prior_var` : `float`, optional
: Variance of the prior over the unknown mean. Estimated during `fit()` when omitted.

`signal_var` : `float`, optional
: Observation variance. Estimated from sliding-window variances when omitted.

`change_length_threshold` : `int`, optional
: Maximum run length counted as a recent change. Defaults to `int(0.1 * run_length)`.

## Methods

### `fit(time_series)`

Estimate any omitted prior parameters from a one-dimensional signal. The signal must contain at least `run_length` samples.

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: Signal to process in chronological order. Calls `fit()` automatically when needed.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Posterior probability assigned to run lengths at or below `change_length_threshold`.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.bocpd import BOCPD

rng = np.random.default_rng(4)
signal = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
score = BOCPD(run_length=40).transform(signal)
```

**Reference:** [Adams and MacKay, Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742).
