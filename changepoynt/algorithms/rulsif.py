import numpy as np
from changepoynt.utils import linalg as lg
from changepoynt.utils import densityratioestimation as dre
from changepoynt.algorithms.base_algorithm import Algorithm
import multiprocessing as mp


class RuLSIF(Algorithm):
    """
    This class implements change point detection based on relative density ration estimation optimizing the least
    squares approach (RuLSIF) from:

    [1]
    Liu, Song, et al.
    "Change-point detection in time-series data by relative density-ratio estimation."
    Neural Networks 43 (2013): 72-83.

    The code has been created looking at:
    - https://github.com/hoxo-m/densratio_py
    - https://pypi.org/project/densratio/
    - http://www.makotoyamada-ml.com/RuLSIF.html (Matlab implementation original author)
    - https://github.com/anewgithubname/change_detection (Matlab implementation co-author)
    """

    def __init__(self, window_length: int = 10, n_windows: int = 50, lag: int = None, estimation_lag: int = None,
                 scoring_step: int = 1, n_kernels: int = 100, alpha: float = 0.01, symmetric=True) -> None:
        """
        This defines all necessary parameters for the RuLSIF to work.
        :param window_length: the length of the windows we want to compare the densities for (k in the paper)
        :param n_windows: the amount of windows (n in the paper)
        :param lag: the difference in time between the past and the future comparison window
        :param estimation_lag: how often we should re-estimate lambda and sigma for the gaussian kernels
        :param scoring_step: the number of samples between each change score value (e.g. 2 would half the computations)
        :param n_kernels: the number of kernels for the density ration estimation
        :param alpha: the smoothing parameter for the RELATIVE in RuLSIF
        :param symmetric: specifies whether to use two processes to compute a forward and backward pass simultaneously
        """
        self.window_length = window_length
        self.n_windows = n_windows
        self.lag = lag
        self.estimation_lag = estimation_lag
        self.n_kernels = n_kernels
        self.alpha = alpha
        self.scoring_step = scoring_step
        self.symmetric = symmetric

        # check that alpha does not exceed the bounds
        assert 0 <= self.alpha < 1, 'The alpha parameter should be in the interval [0,1).'

        # check for the estimation lag
        assert self.estimation_lag is None or 1 <= self.estimation_lag, \
            'The estimation lag needs to be bigger than zero samples.'

        # set the lag if it is not given
        if not self.lag:
            self.lag = self.n_windows

    def transform(self, time_series: np.ndarray):

        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # compute the starting point of the scoring (past and future hankel need to fit)
        starting_point = self.window_length + self.n_windows + self.lag
        assert starting_point < time_series.shape[0], "The time series is too short to score any points."

        # create the estimator
        estimator = dre.RuLSIF(self.alpha)

        # copy the time series to protect it from any modifications within the algorithm
        time_series = time_series.copy()

        # check whether we want to compute the symmetric score
        if self.symmetric:

            # compute the scores with two workers one going forward and the other one backward
            with mp.Pool(2) as workers:
                result = workers.starmap(_transform,
                                         ((time_series, starting_point, self.window_length, self.n_windows, self.lag,
                                           self.scoring_step, estimator),
                                          (time_series[::-1], starting_point, self.window_length, self.n_windows,
                                           self.lag, self.scoring_step, estimator)))

            return result[0] + result[1][::-1]

        else:
            # call the function to compute the values
            return _transform(time_series, starting_point, self.window_length, self.n_windows, self.lag,
                              self.scoring_step, estimator)


def _transform(time_series: np.ndarray, starting_point: int, window_length: int, n_windows: int, lag: int,
               scoring_step: int, estimator: dre.RuLSIF) -> np.ndarray:

    # create the empty score vector
    score = np.zeros_like(time_series)

    # compute the offset
    offset = n_windows

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(starting_point, time_series.shape[0], scoring_step):

        # compile the past hankel matrix (Y)
        hankel_matrix = lg.compile_hankel(time_series, idx, window_length, 2*n_windows)

        # compute the score
        score[idx-offset-scoring_step//2:idx-offset+(scoring_step+1)//2] = estimator(hankel_matrix[:, :n_windows],
                                                                                     hankel_matrix[:, n_windows:])

    return score


def short_test():
    from time import time
    import matplotlib.pyplot as plt
    import scipy

    # make synthetic step function
    np.random.seed(123)
    length = 200
    x = np.hstack([1 * np.ones(length) + np.random.rand(length) * 1,
                   3 * np.ones(length) + np.random.rand(length) * 2,
                   5 * np.ones(length) + np.random.rand(length) * 1.5])
    x = np.hstack([np.random.normal(scale=4, size=length),
                   np.random.normal(scale=1, size=length),
                   np.random.normal(scale=4, size=length)])
    x = scipy.io.loadmat('..\..\examples\logwell.mat')["y"][0, :]

    # plot and show the signal
    plt.plot(x)
    plt.show()

    # create the rulsif method
    ruli = RuLSIF(symmetric=True)

    # make the scoring
    start = time()
    score = ruli.transform(x)
    print((time() - start))
    plt.plot(score)
    plt.show()


if __name__ == '__main__':
    short_test()
