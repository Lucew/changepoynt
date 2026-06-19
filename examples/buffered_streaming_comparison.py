"""Compare direct SST scoring with experimental rolling-buffer scoring."""

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


def main():
    """Run and visualize the direct-versus-buffered comparison."""
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

    direct_score = SST(**settings).transform(signal)
    streamed_score = transform_buffered(SST(**settings), signal)

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


if __name__ == "__main__":
    main()
