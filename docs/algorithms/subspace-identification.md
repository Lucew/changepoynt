# Subspace Identification

`SubspaceIdentification` is intended to detect changes in an estimated dynamical-system subspace rather than directly comparing signal shapes.

!!! warning "Currently unavailable"
    The constructor raises `NotImplementedError`; the offline implementation is unfinished.

## Parameters

`window_length` : `int`
: Subsequence length in samples.

`n_windows` : `int`, optional
: Number of subsequences. The intended default is `window_length`.

`lag` : `int`, optional
: Separation between compared regions.

`rank` : `int`, default `5`
: Estimated system-subspace rank.

`scale` : `bool`, default `True`
: Intended min-max scaling before scoring.

`method` : `{"offline"}`, default `"offline"`
: Intended implementation variant.

`random_rank` : `int`, optional
: Randomized approximation dimension.

## Minimal Example

```python
from changepoynt.algorithms.si import SubspaceIdentification

try:
    detector = SubspaceIdentification(window_length=40)
except NotImplementedError:
    detector = None  # Subspace Identification is not yet available
```

**Reference:** [Kawahara, Yairi, and Machida (2007)](https://doi.org/10.1109/ICDM.2007.78).
