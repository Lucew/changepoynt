import numpy as np
import pytest

from changepoynt.algorithms.messt import MESST
import changepoynt.algorithms.esst as cpesst


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


def test_messt_unknown_method_raises_value_error():
    with pytest.raises(ValueError):
        MESST(window_length=40, method="does-not-exist")


def test_messt_rejects_fast_hankel_for_fbrsvd():
    with pytest.raises(ValueError):
        MESST(window_length=40, method="fbrsvd", use_fast_hankel=True)


def test_messt_rejects_1d_input():
    signal, _ = _make_frequency_change_signal()
    detector = MESST(window_length=40, method="rsvd")
    with pytest.raises(AssertionError):
        detector.transform(signal)


def test_messt_rejects_too_short_signal():
    detector = MESST(window_length=40, n_windows=20, lag=20, method="rsvd")
    too_short = np.linspace(0.0, 1.0, 80)
    with pytest.raises(AssertionError):
        detector.transform(too_short)


def test_messt_score_is_zero_before_first_possible_output():
    signal, _ = _make_frequency_change_signal()
    detector = MESST(window_length=40, n_windows=20, lag=20, method="rsvd")
    np.random.seed(7)
    score = detector.transform(signal[..., None])

    expected_first_scored_idx = detector.window_length
    np.testing.assert_allclose(score[:expected_first_scored_idx], 0.0)


def test_messt_detects_frequency_change_near_boundary():
    signal, change_idx = _make_frequency_change_signal()
    detector = MESST(window_length=48, n_windows=24, lag=24, rank=2, method="rsvd")

    np.random.seed(11)
    score = detector.transform(signal[..., None])

    neighborhood = score[change_idx - 60: change_idx + 60]
    outside = _outside_region(score, center=change_idx, half_width=120, valid_start=detector.window_length)

    assert neighborhood.size > 0
    assert outside.size > 0
    assert np.isfinite(score).all()
    assert neighborhood.max() > np.percentile(outside, 97)


def test_messt_fast_hankel_tracks_reference_implementation():
    signal, _ = _make_frequency_change_signal()
    slow = MESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=False)
    fast = MESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=True)

    np.random.seed(31)
    slow_score = slow.transform(signal[..., None])
    np.random.seed(31)
    fast_score = fast.transform(signal[..., None])

    valid_start = slow.window_length
    corr = np.corrcoef(slow_score[valid_start:], fast_score[valid_start:])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95


def test_messt_tracks_esst():
    signal, _ = _make_frequency_change_signal()
    slow = MESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=False)
    fast = MESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=True)
    orig_esst = cpesst.ESST(window_length=40, n_windows=20, lag=20, rank=2, method="rsvd", use_fast_hankel=False)

    np.random.seed(31)
    slow_score = slow.transform(signal[..., None])
    np.random.seed(31)
    fast_score = fast.transform(signal[..., None])
    np.random.seed(31)
    esst_score = orig_esst.transform(signal)


    valid_start = slow.window_length
    corr = np.corrcoef(slow_score[valid_start:], esst_score[valid_start:])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95
    corr = np.corrcoef(fast_score[valid_start:], esst_score[valid_start:])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95