# Experimental Buffered Stream Processing

Most subspace scores depend on a bounded neighborhood, not the complete time series. This makes an experimental streaming pattern possible: keep a rolling buffer, call the existing `transform()` method whenever a sample arrives, and extract the one newly available score.

!!! warning "Experimental, not a streaming API"
    This tutorial reuses the current batch implementation. It proves that bounded-buffer processing is possible, but it does not preserve decomposition state or avoid repeated allocation. Treat it as a prototype for a future streaming interface.

## What We Are Building

For every incoming sample:

1. Append the sample to a fixed-size buffer.
2. Wait until the buffer contains enough context.
3. Call `detector.transform(buffer)`.
4. Read the newly computed value from `detector.first_score_position`.
5. Attach that value to the correct, earlier sample in the output stream.

```text
new sample -> rolling buffer -> transform(buffer)
                                |
                                +-> one delayed score update
```

The delay is expected. A subspace score compares past and future behavior, so the algorithm cannot score a position until its future context has arrived.

## Buffer Size and Score Position

All four subspace algorithms inherit the information needed from `SingularSubspaceAlgorithm`:

```python
total_region, matrix_region = detector.covered_regions()
buffer_size = total_region + 1
score_position = detector.first_score_position
```

`total_region` is the context required by the two trajectory matrices. The current `transform()` implementations require one additional sample so that at least one scoring iteration can run; therefore the smallest callable buffer has `total_region + 1` samples.

The score is not written at the newest buffer position. Past and future context are aligned to an earlier sample:

```python
delay = buffer_size - 1 - score_position
global_score_index = current_input_index - delay
```

For `scoring_step=1`, the delay is:

- `n_windows // 2 + lag` for `SST` and `MSST`.
- `n_windows + lag` for `ESST` and `MESST`.

This is delayed bounded-memory processing, not a zero-latency score for the newest sample.

## Reusable Buffered Transform

The following function works with `SST`, `ESST`, `MSST`, and `MESST`. A one-dimensional input is required by SST/ESST; MSST/MESST expect shape `(n_samples, n_channels)`.

```python
from collections import deque

import numpy as np


def transform_buffered(detector, samples):
    """Replay an array sample-by-sample through a subspace detector."""
    if detector.scoring_step != 1:
        raise ValueError("Buffered replay requires scoring_step=1.")

    total_region, _ = detector.covered_regions()
    buffer_size = total_region + 1
    score_position = detector.first_score_position
    delay = buffer_size - 1 - score_position

    output = np.full(samples.shape[0], np.nan, dtype=float)
    buffer = deque(maxlen=buffer_size)

    for current_index, sample in enumerate(samples):
        buffer.append(sample)

        if len(buffer) < buffer_size:
            continue

        buffer_signal = np.asarray(buffer)
        buffer_score = detector.transform(buffer_signal)

        output_index = current_index - delay
        output[output_index] = buffer_score[score_position]

    return output
```

The `NaN` prefix marks samples for which the stream has not yet accumulated enough future context. It is preferable to zero padding when zero is itself a valid score.

## Univariate Example

```python
import numpy as np

from changepoynt.algorithms.esst import ESST

rng = np.random.default_rng(7)
signal = np.r_[rng.normal(0.0, 0.2, 200),
               rng.normal(1.0, 0.2, 200)]

detector = ESST(
    window_length=40,
    method="rsvd",
    scoring_step=1,
    scale=False,
)

stream_score = transform_buffered(detector, signal)
```

The wrapper is interchangeable. For example, replace the detector with:

```python
from changepoynt.algorithms.sst import SST

detector = SST(
    window_length=40,
    method="rsvd",
    scoring_step=1,
    scale=False,
)
```

The wrapper still obtains the correct buffer and output positions from the shared base-class interface.

## Multivariate Example

The same wrapper accepts samples containing multiple channels:

```python
from changepoynt.algorithms.messt import MESST

multivariate_signal = np.column_stack((temperature, pressure, vibration))

detector = MESST(
    window_length=40,
    scoring_step=1,
    scale=False,
)

stream_score = transform_buffered(detector, multivariate_signal)
```

Use `MSST` in the same way when the classic SST score is preferred.

## Runtime Estimation

The base class can estimate the cost before replaying the stream:

```python
estimated_seconds, estimated_std = detector.estimate_runtime(
    signal,
    steps=30,
)
```

`estimate_runtime()` warms up first-use compilation, repeatedly times the smallest processable slice, and scales the result to the signal length. That is close to this experimental minimal-buffer pattern, although real stream ingestion and buffer-conversion overhead may differ.

The first live call can still have noticeable initialization cost. Warm the configured detector before latency-sensitive use.

## Important Limitations

### Use `scoring_step=1`

Every buffered call starts a new batch and contains only one scoring iteration. Setting `scoring_step > 1` inside the detector therefore does not skip calls across the stream; it only changes where and how widely that one result is written in the temporary output.

To score only every fifth incoming sample, keep the detector at `scoring_step=1` and call the buffered transform conditionally in a dedicated streaming implementation. Decide separately whether downstream consumers should hold the previous score between updates.

### Define Scaling Outside the Buffer

With `scale=True`, every call min-max scales its current buffer independently. This differs from transforming one complete signal and can make the scale change as old samples leave the buffer.

For controlled streaming behavior, apply a fixed calibration or a deliberately designed online normalization before buffering, then construct the detector with `scale=False`.

### Decomposition State Is Reset

Each `transform()` call starts from scratch:

- IKA and related SST paths do not preserve their feedback vector across buffers.
- Randomized decompositions draw new projections.
- Repeated calls rebuild representations and allocate output arrays.

The resulting score can therefore differ from one full batch call, even when it uses the same samples and parameters. A production streaming implementation should preserve useful decomposition state between updates.

### Density-Ratio Methods Need Separate Alignment

RuLSIF and ULSIF also operate on bounded windows, but they do not inherit `covered_regions()` or `first_score_position`. Their symmetric mode additionally combines forward and reversed processing, which is not naturally causal.

The same buffering idea may be applied to a carefully defined one-direction density-ratio stream, but its buffer size, score alignment, and state should be exposed through a dedicated interface before treating it as interchangeable with the subspace example.

## Complete SST Comparison Script

This script computes SST in two ways:

1. One direct call with the complete signal.
2. Repeated calls with the smallest valid rolling buffer.

It uses `method="naive updated"` and `scale=False` deliberately. This decomposition is deterministic and does not carry a feedback vector between score positions, while external scaling avoids changing normalization from one buffer to the next. The comparable outputs should therefore match exactly apart from floating-point tolerance.

```python
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from changepoynt.algorithms.sst import SST


def transform_buffered(detector, samples):
    """Replay samples through the batch transform using a rolling buffer."""
    if detector.scoring_step != 1:
        raise ValueError("Buffered replay requires scoring_step=1.")

    total_region, _ = detector.covered_regions()
    buffer_size = total_region + 1
    score_position = detector.first_score_position
    delay = buffer_size - 1 - score_position

    output = np.full(samples.shape[0], np.nan, dtype=float)
    buffer = deque(maxlen=buffer_size)

    for current_index, sample in enumerate(samples):
        buffer.append(sample)

        if len(buffer) < buffer_size:
            continue

        buffer_score = detector.transform(np.asarray(buffer))
        output_index = current_index - delay
        output[output_index] = buffer_score[score_position]

    return output


# A frequency change with a little noise.
rng = np.random.default_rng(7)
t = np.linspace(0, 12 * np.pi, 300)
signal = np.sin(t) + rng.normal(0.0, 0.05, t.size)
signal[150:] = (
    np.sin(2.2 * t[150:])
    + rng.normal(0.0, 0.05, t.size - 150)
)

settings = dict(
    window_length=20,
    n_windows=20,
    lag=6,
    rank=3,
    method="naive updated",
    scoring_step=1,
    scale=False,
)

# Direct batch processing.
direct_detector = SST(**settings)
direct_score = direct_detector.transform(signal)

# Simulated streaming with the same configuration.
stream_detector = SST(**settings)
streamed_score = transform_buffered(stream_detector, signal)

# NaNs mark positions for which the stream lacks past or future context.
valid = np.isfinite(streamed_score)
np.testing.assert_allclose(
    streamed_score[valid],
    direct_score[valid],
    rtol=1e-10,
    atol=1e-10,
)

max_error = np.max(
    np.abs(streamed_score[valid] - direct_score[valid])
)
print(f"Compared samples: {valid.sum()}")
print(f"Maximum absolute difference: {max_error:.3e}")

figure, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
axes[0].plot(signal)
axes[0].set_title("Signal")
axes[1].plot(direct_score, color="tab:blue")
axes[1].set_title("Direct SST")
axes[2].plot(streamed_score, color="tab:orange")
axes[2].set_title("Buffered SST")
figure.tight_layout()
plt.show()
```

For this signal and configuration, the script compares 255 scored positions and reports a maximum absolute difference of `0.000e+00`.

If you replace `"naive updated"` with `"rsvd"` or `"ika"`, compare the score shape and peaks rather than requiring exact equality. Those methods restart randomized projections or feedback state on every buffered call in this prototype.
