# the methods in this file are not published algorithms but reference methods for sanity checking evaluation metrics
# and benchmark signals. If these methods reach high performance, either the benchmark data is relatively "easy" to
# solve or the metrics used for performance evaluation are not suitable.
#
# With this we want to follow an argument made for Time Series Anomaly Detection and sanity check whether a new method
# really is progress. See the following line from a very interesting paper:
# "However, as we will show, much of the results of this complex approach can be duplicated with a single line
# of code and a few minutes of effort."
# in "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress"
# Renjie Wu and Eamonn J. Keogh
# IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
# 2023
#
# The ZERO method is also used as a baseline method in:
# "An Evaluation of Change Point Detection Algorithms"
# Gerrit J. J. van den Burg and Christopher K. I. Williams
# arXiv preprint arXiv:2003.06222, 2020
import numpy as np

from changepoynt.algorithms.base_algorithm import Algorithm


class ZERO(Algorithm):
    """
    This class implements the ZERO baseline algorithm.

    "An Evaluation of Change Point Detection Algorithms"
    Gerrit J. J. van den Burg and Christopher K. I. Williams
    arXiv preprint arXiv:2003.06222, 2020
    """

    def __init__(self) -> None:
        super().__init__()

    def fit(self, time_series: np.ndarray) -> None:
        pass

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        return np.zeros_like(time_series)


class MovingWindow(Algorithm):
    """
    This class implements moving window algorithms that are as simple as possible for sanity checking.

    This class implements a similar ideas as in:
    Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress
    Renjie Wu and Eamonn J. Keogh
    IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING
    2023
    """

    def __init__(self, window_length: int, method: str = 'mean') -> None:
        super().__init__()

        # define the possible methods
        possible_methods = {"mean", "var", "meanvar"}

        # save the input parameters
        self.__fit = False
        assert window_length > 0, 'Window length must be greater than zero.'
        self.window_length = window_length
        assert method in possible_methods, f'Method must be one of the following: {possible_methods}.'
        self.method = method


    def fit(self, time_series: np.ndarray) -> None:
        """
        Check for the properties of the input.
        :param time_series:
        :return:
        """
        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # check that we have at least two windows
        assert time_series.shape[0] > 2*self.window_length, 'Time series needs to be longer than 2x window length.'

        self.__fit = True

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        if not self.__fit:
            self.fit(time_series)

        # use numpy tricks to create the sliding window
        sliding_window = np.lib.stride_tricks.sliding_window_view(time_series, self.window_length)
        sliding_window_var = np.var(sliding_window, axis=-1)
        sliding_window_mean = np.mean(sliding_window, axis=-1)

        # create a time series for the scores
        score = np.zeros_like(time_series)
        if self.method.startswith('mean'):
            score[self.window_length:-self.window_length+1] += np.abs(sliding_window_mean[:-self.window_length]
                                                                      - sliding_window_mean[self.window_length:])
        if self.method.endswith('var'):
            score[self.window_length:-self.window_length+1] += np.abs(sliding_window_var[:-self.window_length]
                                                                      - sliding_window_var[self.window_length:])
        return score