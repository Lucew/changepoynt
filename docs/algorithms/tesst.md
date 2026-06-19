# TESST

`TESST` is an experimental GPU implementation related to ESST. It batches Hankel matrices and uses `torch.svd_lowrank` on a CUDA device.

!!! warning "Experimental"
    TESST requires a CUDA-enabled PyTorch installation and is not part of the package's stable algorithm API.

## Parameters

`window_length` : `int`
: Subsequence length in samples.

`n_windows` : `int`, optional
: Number of subsequences per side. Defaults to `window_length // 2`.

`lag` : `int`, optional
: Separation between compared matrices. Defaults to `n_windows`.

`rank` : `int`, default `5`
: Low-rank decomposition size.

`scoring_step` : `int`, default `1`
: Distance between scored positions.

`scale` : `bool`, default `True`
: Min-max scale the signal before batching.

## Methods

### `transform(time_series)`

`time_series` : `numpy.ndarray`, shape `(n_samples,)`
: One-dimensional signal.

**Returns**

`score` : `numpy.ndarray`, shape `(n_samples,)`
: Experimental GPU-computed entangled score.

## Minimal Example

```python
import numpy as np
from changepoynt.algorithms.torch_esst import TESST

signal = np.sin(np.linspace(0, 20 * np.pi, 400))
detector = TESST(window_length=40)  # requires CUDA-enabled PyTorch
score = detector.transform(signal)
```

TESST does not currently have a separate publication; see the [ESST reference](https://ntrs.nasa.gov/citations/20250002705) for the related score family.
