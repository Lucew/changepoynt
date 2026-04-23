import pytest
import numpy as np
import changepoynt.algorithms.sst as ssts
import logging


class TestSST:
    def setup_method(self):
        # set a random seed
        np.random.seed(3455)

        # create a random steps signal of a certain length
        self.signal_length = 300
        x0 = 1 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1
        x1 = 3 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 2
        x2 = 5 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1.5
        x = np.hstack([x0, x1, x2])
        x += np.random.rand(x.size)
        self.signal = x

    def teardown_method(self):
        pass

    def test_all_methods(self):
        # initialize random default method
        sst = ssts.SST(30)

        # get the different method names
        methods = list(sst.methods.keys())

        # go through the methods and check execution
        for method in methods:
            ssts.SST(50, rank=2, method=method).transform(self.signal)

    def test_svd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="svd")
        sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="svd", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_rectangle_matrix(self):

        # initialize the scorer
        sst = ssts.SST(50, 20, method="ika")
        sst.transform(self.signal)

    def test_fbrsvd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="fbrsvd")
        sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="fbrsvd", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_rsvd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="rsvd")
        res = sst.transform(self.signal)
        sst = ssts.SST(50, method="rsvd", use_fast_hankel=True)
        res2 = sst.transform(self.signal)

    def test_ika_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="ika")
        res = sst.transform(self.signal)
        sst = ssts.SST(50, method="ika", use_fast_hankel=True)
        res2 = sst.transform(self.signal)

    def test_weighted_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="weighted")
        res = sst.transform(self.signal)
        sst = ssts.SST(50, method="weighted", use_fast_hankel=True)
        res2 = sst.transform(self.signal)

    def test_naive_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="naive")
        res = sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="naive", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_all_methods_mitigate_offset(self):
        # initialize random default method
        sst = ssts.SST(30)

        # get the different method names
        methods = list(sst.methods.keys())

        # go through the methods and check execution
        for method in methods:
            ssts.SST(50, rank=2, method=method, mitigate_offset=True).transform(self.signal)

    def test_default(self):
        ssts.SST(min(5, self.signal_length//2), rank=2).transform(self.signal)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            ssts.SST(10, method='asdafwegrhqh')


def _make_frequency_change_signal(
    n_per_segment: int = 320,
    period_before: int = 48,
    period_after: int = 14,
    noise: float = 0.02,
    seed: int = 5678,
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


def test_sst_rejects_fast_hankel_with_offset_mitigation():
    with pytest.raises(ValueError):
        ssts.SST(window_length=40, method="rsvd", use_fast_hankel=True, mitigate_offset=True)


def test_sst_rejects_non_1d_input():
    signal, _ = _make_frequency_change_signal()
    detector = ssts.SST(window_length=40, method="rsvd")
    with pytest.raises(AssertionError):
        detector.transform(np.vstack([signal, signal]))


def test_sst_rejects_too_short_signal():
    detector = ssts.SST(window_length=40, n_windows=40, lag=10, method="rsvd")
    too_short = np.linspace(0.0, 1.0, 80)
    with pytest.raises(AssertionError):
        detector.transform(too_short)


def test_sst_score_is_zero_before_first_possible_output():
    signal, _ = _make_frequency_change_signal()
    detector = ssts.SST(window_length=40, n_windows=40, lag=10, rank=2, method="rsvd")
    np.random.seed(7)
    score = detector.transform(signal)

    expected_first_scored_idx = detector.window_length + detector.n_windows // 2
    np.testing.assert_allclose(score[:expected_first_scored_idx], 0.0)


@pytest.mark.parametrize("method", ["rsvd", "ika", "weighted"])
def test_sst_detects_frequency_change_near_boundary(method: str):
    signal, change_idx = _make_frequency_change_signal()
    detector = ssts.SST(window_length=48, n_windows=48, lag=16, rank=2, method=method)

    np.random.seed(11)
    score = detector.transform(signal)

    neighborhood = score[change_idx - 70: change_idx + 70]
    outside = _outside_region(
        score,
        center=change_idx,
        half_width=140,
        valid_start=detector.window_length + detector.n_windows // 2,
    )

    assert neighborhood.size > 0
    assert outside.size > 0
    assert np.isfinite(score).all()
    assert neighborhood.max() > np.percentile(outside, 97)


def test_sst_offset_mitigation_makes_scores_translation_invariant():
    signal, _ = _make_frequency_change_signal(noise=0.01)
    shifted_signal = signal + 250.0
    detector = ssts.SST(window_length=40, n_windows=40, lag=10, rank=2, method="rsvd", scale=False, mitigate_offset=True)

    np.random.seed(21)
    reference = detector.transform(signal)
    np.random.seed(21)
    shifted = detector.transform(shifted_signal)

    np.testing.assert_allclose(reference, shifted, rtol=1e-6, atol=1e-7)


def test_sst_fast_hankel_tracks_reference_implementation():
    signal, _ = _make_frequency_change_signal()
    slow = ssts.SST(window_length=40, n_windows=40, lag=10, rank=2, method="rsvd", use_fast_hankel=False)
    fast = ssts.SST(window_length=40, n_windows=40, lag=10, rank=2, method="rsvd", use_fast_hankel=True)

    np.random.seed(31)
    slow_score = slow.transform(signal)
    np.random.seed(31)
    fast_score = fast.transform(signal)

    valid_start = slow.window_length + slow.n_windows // 2
    corr = np.corrcoef(slow_score[valid_start:], fast_score[valid_start:])[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95


if __name__ == "__main__":
    pytest.main()
