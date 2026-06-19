# MovingWindow

`MovingWindow` is a deliberately simple baseline. It compares the mean, variance, or both between adjacent windows and is useful as a sanity check before applying a more complex detector.

## Parameters

`window_length` : `int`
: Number of samples in each adjacent comparison window.

`method` : `{"mean", "var", "meanvar"}`, default `"mean"`
: Statistic used in the absolute-difference score.

## Methods

### `fit(time_series)`

Validate that the input is one-dimensional and longer than twice `window_length`.

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: Signal to compare. `fit()` is called automatically when necessary.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Absolute mean and/or variance difference with zero-padded boundaries.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.baseline import MovingWindow

signal = np.r_[np.zeros(100), np.ones(100)]
score = MovingWindow(window_length=20, method="meanvar").transform(signal)
```

**Background:** [Wu and Keogh, Current Time Series Anomaly Detection Benchmarks are Flawed](https://doi.org/10.1109/TKDE.2021.3112126).
