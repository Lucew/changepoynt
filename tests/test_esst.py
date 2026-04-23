import numpy as np
import pytest

from changepoynt.algorithms.esst import ESST


def _make_frequency_change_signal(
    n_per_segment: int = 320,
    period_before: int = 48,
    period_after: int = 14,
    noise: float = 0.02,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n_per_segment)
    left = np.sin(2 * np.pi * t / period_before)
    right = np.sin(2 * np.pi * t / period_after)
    signal = np.concatenate([left, right])
    signal += noise * rng.standard_normal(signal.shape[0])
    return signal, n_per_segment


def _outside_region(score: np.ndarray, center: int, half_width: int, valid_start: int) -> np.ndarray:
    left = score[valid_start:max(valid_start, center - half_width)]
    right = score[min(center + half_width, score.shape[0]):]
    if left.size and right.size:
        return np.concatenate([left, right])
    if left.size:
        return left
    if right.size:
        return right
    raise AssertionError("Need some samples outside the change region for comparison.")


def test_esst_unknown_method_raises_value_error():
    with pytest.raises(ValueError):
        ESST(window_length=40, method="does-not-exist")


def test_esst_rejects_fast_hankel_for_fbrsvd():
    with pytest.raises(ValueError):
        ESST(window_length=40, method="fbrsvd", use_fast_hankel=True)


def test_esst_rejects_fast_hankel_with_offset_mitigation():
    with pytest.raises(ValueError):
        ESST(window_length=40, method="rsvd", use_fast_hankel=True, mitigate_offset=True)


def test_esst_rejects_non_1d_input():
    signal, _ = _make_frequency_change_signal()
    detector = ESST(window_length=40, method="rsvd")
    with pytest.raises(AssertionError):
        detector.transform(np.vstack([signal, signal]))


def test_esst_rejects_too_short_signal():
    detector = ESST(window_length=40, n_windows=20, lag=20, method="rsvd")
    too_short = np.linspace(0.0, 1.0, 80)
    with pytest.raises(AssertionError):
        detector.transform(too_short)


def test_esst_score_is_zero_before_first_possible_output():
    signal, _ = _make_frequency_change_signal()
    detector = ESST(window_length=40, n_windows=20, lag=20, method="rsvd")
    np.random.seed(7)
    score = detector.transform(signal)

    expected_first_scored_idx = detector.window_length
    np.testing.assert_allclose(score[:expected_first_scored_idx], 0.0)


def test_esst_detects_frequency_change_near_boundary():
    signal, change_idx = _make_frequency_change_signal()
    detector = ESST(window_length=48, n_windows=24, lag=24, rank=2, method="rsvd")

    np.random.seed(11)
    score = detector.transform(signal)

    neighborhood = score[change_idx - 60: change_idx + 60]
    outside = _outside_region(score, center=change_idx, half_width=120, valid_start=detector.window_length)

    assert neighborhood.size > 0
    assert outside.size > 0
    assert np.isfinite(score).all()
    assert neighborhood.max() > np.percentile(outside, 97)


def test_esst_offset_mitigation_makes_scores_translation_invariant():
    signal, _ = _make_frequency_change_signal(noise=0.01)
    shifted_signal = signal + 250.0
    detector = ESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", scale=False, mitigate_offset=True)

    np.random.seed(21)
    reference = detector.transform(signal)
    np.random.seed(21)
    shifted = detector.transform(shifted_signal)

    np.testing.assert_allclose(reference, shifted, rtol=1e-6, atol=1e-7)


def test_esst_fast_hankel_tracks_reference_implementation():
    signal, _ = _make_frequency_change_signal()
    slow = ESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=False)
    fast = ESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=True)

    np.random.seed(31)
    slow_score = slow.transform(signal)
    np.random.seed(31)
    fast_score = fast.transform(signal)

    valid_start = slow.window_length
    corr = np.corrcoef(slow_score[valid_start:], fast_score[valid_start:])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95
