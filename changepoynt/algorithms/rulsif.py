import numpy as np
from changepoynt.utils import linalg as lg
from changepoynt.utils import densityratioestimation as dre
from changepoynt.algorithms.base_algorithm import Algorithm


class RuLSIF(Algorithm):
    """
    This class implements change point detection based on relative density ration estimation optimizing a least squares
    approach from:

    [1]
    Liu, Song, et al.
    "Change-point detection in time-series data by relative density-ratio estimation."
    Neural Networks 43 (2013): 72-83.

    We use an implemented package called densratio for the computations of the density ratio and the Pearson divergence
    calculations. We merely adopt some steps to account for the symmetric Pearson divergence used in [1]. We adopted
    the function RuLSIF(x, y, alpha, sigma_range, lambda_range, kernel_num=100, verbose=True) from the pacakge.

    The code has been created looking at:
    - https://github.com/hoxo-m/densratio_py
    - https://pypi.org/project/densratio/
    - http://www.makotoyamada-ml.com/RuLSIF.html (Matlab implementation original author)

    and was adopted from:
    - https://github.com/DeepOmics/rulsif_abrupt-change_detection
    and we can not thank the authors enough for writing such deep mathematical code!
    The reason for adoption were the one named above (symmetric pearson divergence)
    and more direct use of numba as a jit compiler, as well as we do not want and use many things computed in the
    package, so we can save time by not computing them.
    """

    def __init__(self, window_length: int = 10, n_windows: int = 50, lag: int = None, estimation_lag: int = None,
                 n_kernels: int = 100, alpha: float = 0.1) -> None:
        """
        This defines all necessary parameters for the RuLSIF to work.
        :param window_length: the length of the windows we want to compare the densities for (k in the paper)
        :param n_windows: the amount of windows (n in the paper)
        :param lag: the difference in time between the past and the future comparison window
        :param estimation_lag: how often we should re-estimate lambda and sigma for the gaussian kernels
        :param alpha: the smoothing parameter for the RELATIVE in RuLSIF
        :param n_kernels: the amount of gaussian kernels for the density ratio
        """
        self.window_length = window_length
        self.n_windows = n_windows
        self.lag = lag
        self.estimation_lag = estimation_lag
        self.n_kernels = n_kernels
        self.alpha = alpha

        # check that alpha does not exceed the bounds
        assert 0 <= self.alpha < 1, 'The alpha parameter should be in the interval [0,1).'

        # set some rule of thumb parameters
        if not self.estimation_lag:
            self.estimation_lag = window_length*2

        # set the lag if it is not given
        if not self.lag:
            self.lag = self.n_windows

        # create the estimator from the utils
        self.estimator = dre.RULSIF(alpha=self.alpha, n_kernels=self.n_kernels)

    def transform(self, time_series: np.ndarray):

        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # compute the starting point of the scoring (past and future hankel need to fit)
        starting_point = self.window_length + self.n_windows + self.lag
        assert starting_point < time_series.shape[0], "The time series is too short to score any points."

        # copy the time series to protect it from any modifications within the algorithm
        time_series = time_series.copy()

        # call the function to compute the values
        return _transform(time_series, starting_point, self.window_length, self.n_windows, self.lag,
                          self.estimation_lag, self.estimator)


def _transform(time_series: np.ndarray, starting_point: int, window_length: int, n_windows: int, lag: int,
               estimation_lag: int, estimator: dre.Estimator) -> np.ndarray:

    # compile the past hankel matrix (Y)
    hankel_past = lg.compile_hankel(time_series, starting_point - lag, window_length, n_windows)

    # compile the future hankel matrix (Y')
    hankel_future = lg.compile_hankel(time_series, starting_point, window_length, n_windows)

    # create the estimation for sigma and lambda via built in cross validation
    estimator.train(hankel_past, hankel_future)

    # create the empty score vector
    score = np.zeros_like(time_series)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(starting_point, time_series.shape[0]):

        # compile the past hankel matrix (Y)
        hankel_past = lg.compile_hankel(time_series, idx - lag, window_length, n_windows)

        # compile the future hankel matrix (Y')
        hankel_future = lg.compile_hankel(time_series, idx, window_length, n_windows)

        # check whether we need the cross validation for the kernel width and position
        if idx % estimation_lag == 0:
            estimator.train(hankel_past, hankel_future)

        # compute the score and save the returned feedback vector
        score[idx] = estimator.apply(hankel_past, hankel_future)

    return score


def short_test():
    from time import time
    # make synthetic step function
    np.random.seed(123)
    length = 300
    x = np.hstack([1 * np.ones(length) + np.random.rand(length) * 1,
                   3 * np.ones(length) + np.random.rand(length) * 2,
                   5 * np.ones(length) + np.random.rand(length) * 1.5])
    x += np.random.rand(x.size)

    # create the rulsif method
    ruli = RuLSIF()

    # make the scoring
    start = time()
    score = ruli.transform(x)
    print((time() - start) / (length * 3))
    print(score[600:610])


if __name__ == '__main__':
    short_test()
