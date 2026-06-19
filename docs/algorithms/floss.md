# FLOSS

`FLOSS` is the streaming counterpart to FLUSS. It updates a matrix profile and corrected arc curve as new samples arrive.

!!! warning "Currently unavailable"
    `FLOSS.__init__()` raises `NotImplementedError` in this package because the wrapper is not yet considered functional.

## Parameters

`window_length` : `int`
: Matrix-profile subsequence length.

`initial_length` : `int`, optional
: Number of initial samples used to initialize the stream. The intended default is `10 * window_length` and it must be at least twice `window_length`.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: One-dimensional stream represented as an array.

**Returns**

`score` : `numpy.ndarray`
: Intended online segmentation score. This method is unreachable through the current constructor.

## Minimal Example

```python
from changepoynt.algorithms.floss import FLOSS

try:
    detector = FLOSS(window_length=30)
except NotImplementedError:
    detector = None  # FLOSS is not available in the current release
```

**Reference:** [Gharghabi et al., Matrix Profile VIII](https://doi.org/10.1109/ICDM.2017.21).
