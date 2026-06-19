# ZERO

`ZERO` returns an all-zero score and serves as a negative-control baseline. A metric or benchmark on which ZERO performs well deserves careful inspection.

## Parameters

This detector has no parameters.

## Methods

### `fit(time_series)`

No-op included for interface compatibility.

### `transform(time_series)`

`time_series` : `numpy.ndarray`
: Input whose shape and dtype are copied.

**Returns**

`score` : `numpy.ndarray`
: Zeros with the same shape and dtype as `time_series`.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.baseline import ZERO

signal = np.array([0.2, 0.4, 1.8, 2.0])
score = ZERO().transform(signal)
```

**Background:** [van den Burg and Williams, An Evaluation of Change Point Detection Algorithms](https://arxiv.org/abs/2003.06222).
