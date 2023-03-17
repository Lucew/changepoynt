import numpy as np
from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.algorithms.rulsif import RuLSIF


class ULSIF(Algorithm):
    """
    This class implements change point detection based on density ration estimation optimizing a least squares
    approach from:

    [1]
    "A least-squares approach to direct importance estimation"
    T. Kanamori, S. Hido, and M. Sugiyama.
    Journal of Machine Learning Research, 10:1391–1445, 2009.

    and can be implemented as a special version of with alpha = 0

    [2]
    Liu, Song, et al.
    "Change-point detection in time-series data by relative density-ratio estimation."
    Neural Networks 43 (2013): 72-83.

    which we are doing here.
    """

    def __init__(self, window_length: int = 10, n_windows: int = 50, lag: int = None, estimation_lag: int = None,
                 n_kernels: int = 100) -> None:
        """
        This defines all necessary parameters for the RuLSIF to work.
        :param window_length: the length of the windows we want to compare the densities for (k in the paper)
        :param n_windows: the amount of windows (n in the paper)
        :param lag: the difference in time between the past and the future comparison window
        :param estimation_lag: how often we should re-estimate lambda and sigma for the gaussian kernels
        :param n_kernels: the amount of gaussian kernels for the density ratio
        """
        self.window_length = window_length
        self.n_windows = n_windows
        self.lag = lag
        self.estimation_lag = estimation_lag
        self.n_kernels = n_kernels

        # create the specialized version of RuLSIF
        self.transformer = RuLSIF(window_length=self.window_length, n_windows=self.n_windows, lag=self.lag,
                                  estimation_lag=self.estimation_lag, n_kernels=self.n_kernels, alpha=0.0)

    def transform(self, time_series: np.ndarray):
        return self.transformer.transform(time_series)


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
    ruli = ULSIF()

    # make the scoring
    start = time()
    score = ruli.transform(x)
    print((time() - start) / (length * 3))
    print(score[600:610])


if __name__ == '__main__':
    short_test()
