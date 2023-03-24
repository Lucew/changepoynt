import numpy as np
from changepoynt.algorithms.base_algorithm import Algorithm
import stumpy


class FLUSS(Algorithm):
    """
    This class uses the FLUSS algorithm described in:

    [1]
    Gharghabi, Shaghayegh, et al.
    "Matrix profile VIII: domain agnostic online semantic segmentation at superhuman performance levels."
    2017 IEEE international conference on data mining (ICDM). IEEE, 2017.

    This class essentially wraps the stumpy library which implements a fast approach to calculate the matrix profile.
    https://stumpy.readthedocs.io/en/latest/index.html

    TODO:
    Implement GPU support and streaming? (FLOSS)
    """

    def __init__(self, window_length: int) -> None:
        """
        We initialize the matrix profile and the subsequent floss using only the window length used for comarisons.

        :param window_length: the length of the window used for the distance comparisons in the matrix profile.
        """

        # save the specified parameters into instance variables
        self.window_length = window_length

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        This function calculate the anomaly score for each sample within the time series.

        It also does some assertion regarding time series length.

        :param time_series: 1D array containing the time series to be scored
        :return: anomaly score
        """

        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # check that we have at least two windows
        assert time_series.shape[0] > self.window_length, 'Time series needs to be longer than window length.'

        # compute the floss score which is the amount of index pointers over this sample
        # compared to an expected curve. It is capped by one.
        matrix_profile = stumpy.stump(time_series, m=self.window_length)
        corrected_arc_crossings, _ = stumpy.fluss(matrix_profile[:, 1], L=self.window_length, n_regimes=1)
        return 1-corrected_arc_crossings


def _main():
    """
    Internal quick testing function.

    :return:
    """
    from time import time
    # make synthetic step function
    np.random.seed(123)
    # synthetic (frequency change)
    x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 1000))
    x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, 1000))
    x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, 1000))
    x3 = np.sin(2 * np.pi * 4 * np.linspace(0, 10, 1000))
    x = np.hstack([x0, x1, x2, x3])
    x += np.random.rand(x.size)

    # create the method
    fluss_recognizer = FLUSS(50)

    # compute the score
    start = time()
    score = fluss_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
