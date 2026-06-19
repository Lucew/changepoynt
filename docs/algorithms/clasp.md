# ClaSP

ClaSP frames segmentation as a classification problem: subsequences from opposite sides of a candidate split should be distinguishable when the regimes differ.

!!! warning "Currently unavailable"
    `CLASP.__init__()` raises `NotImplementedError` because its former dependency combination is not supported by the package.

## Parameters

`n_segments` : `int` or `"learn"`, default `"learn"`
: Number of segments, or automatic selection.

`n_estimators` : `int`, default `10`
: Number of ensemble estimators.

`window_size` : `int` or `"suss"`, default `"suss"`
: Subsequence length or automatic SuSS selection.

`k_neighbours` : `int`, default `3`
: Number of nearest neighbors used by classification.

`distance`, `score`, `validation` : `str`
: Distance, classification score, and validation strategy forwarded to the intended ClaSP backend.

`early_stopping` : `bool`, default `True`
: Enable early stopping of recursive segmentation.

`threshold` : `float`, default `1e-15`
: Significance threshold.

`excl_radius` : `int`, default `5`
: Exclusion radius around selected split points.

`random_state` : `int`, default `2357`
: Random seed.

## Minimal Example

```python
from changepoynt.algorithms.clasp import CLASP

try:
    detector = CLASP()
except NotImplementedError:
    detector = None  # ClaSP is unavailable in the current release
```

**Reference:** [Schafer, Ermshaus, and Leser, ClaSP - Time Series Segmentation](https://doi.org/10.1145/3459637.3482240).
