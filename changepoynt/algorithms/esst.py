import numpy as np
from typing import Callable
from functools import partial
from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.utils import normalization
from changepoynt.utils import linalg as lg
import fbpca


class ESST(Algorithm):
    """
    This class implements an own idea for change point detection.

    TODO:
    Implement GPU support and streaming? (FLOSS)
    """

    def __init__(self, window_length: int, n_windows: int = None, lag: int = None, rank: int = 5, scoring_step: int = 1,
                 scale: bool = True) -> None:
        """
        We initialize the matrix profile and the subsequent floss using only the window length used for comarisons.

        :param window_length: the length of the window used for the distance comparisons in the matrix profile.
        """

        # save the specified parameters into instance variables
        self.window_length = window_length
        self.n_windows = n_windows
        self.rank = rank
        self.scale = scale
        self.lag = lag
        self.scoring_step = scoring_step

        # set some default values when they have not been specified
        if self.n_windows is None:
            self.n_windows = self.window_length//2
        if self.lag is None:
            # rule of thumb
            self.lag = self.n_windows

        # specify the methods and their corresponding functions as lambda functions expecting only the hanke matrix
        self.methods = {'rsvd': partial(left_entropy, rank=self.rank)}

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

        # compute the starting point of the scoring (past and future hankel need to fit)
        starting_point = self.window_length + self.n_windows + self.lag
        assert starting_point < time_series.shape[0], "The time series is too short to score any points."

        # scale the time series (or just copy it if already scaled)
        if self.scale:
            time_series = normalization.min_max_scaling(time_series, min_val=1.0, max_val=2.0, inplace=False)
        else:
            time_series = time_series.copy()

        return _transform(time_series, starting_point, self.window_length, self.n_windows, self.lag, self.scoring_step,
                          self.methods['rsvd'])


def _transform(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int, scoring_step: int,
               scoring_function: Callable) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: amount of columns in the hankel matrix
    """

    # initialize a scoring array with no values yet
    score = np.zeros_like(time_series)

    # compute the offset
    offset = (n_windows + lag)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0], scoring_step):

        # compile the past hankel matrix (H1)
        hankel_past = lg.compile_hankel(time_series, idx - lag, window_length, n_windows)

        # compile the future hankel matrix (H2)
        hankel_future = lg.compile_hankel(time_series, idx, window_length, n_windows)

        # compute the score and save the returned feedback vector
        score[idx-offset-scoring_step//2:idx-offset+(scoring_step+1)//2] = \
            scoring_function(np.concatenate((hankel_past, hankel_future), axis=1))

    return score


def skewness(pdf_matrix: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the skewness of a matrix of probability density functions given as a numpy array.

    Args:
        pdf_matrix (numpy array): Matrix of probability density functions, where each row is a probability density function
        x (numpy array): Array of values for which the pdfs are defined

    Returns:
        Skewness of each probability density function as a numpy array
    """
    mean_vector = np.sum(x * pdf_matrix, axis=1)
    variances_vector = np.sum((x - mean_vector[:, np.newaxis]) ** 2 * pdf_matrix, axis=1)
    std_vector = np.sqrt(variances_vector)
    skewness_vector = np.sum(((x - mean_vector[:, np.newaxis]) / std_vector[:, np.newaxis]) ** 3 * pdf_matrix, axis=1)
    return skewness_vector


def left_right_diff(left_eigenvectors: np.ndarray):
    n = left_eigenvectors.shape[1]//2
    return np.mean(left_eigenvectors[:, :n]-left_eigenvectors[:, n:], axis=1)


def left_entropy(hankel: np.ndarray, rank: int) -> float:
    """
    Entropy Singular Spectrum Transformation.

    :param hankel: the hankel matrix of the signal
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :return: the change point score, the input vector x0
    """
    # compute the left and right eigenvectors using the randomized svd for fast computation
    right_eigenvectors, eigenvalues, left_eigenvectors = fbpca.pca(hankel, rank, True)

    # shift the left eigenvectors up
    left_eigenvectors = left_eigenvectors - np.min(left_eigenvectors, axis=1)[:, None] + 1

    # make the eigenvectors into "probability distributions" so their sum of elements is one
    left_eigenvectors = left_eigenvectors/np.sum(left_eigenvectors, axis=1)[:, None]

    # compute the normalized entropy of the eigenvectors
    """entropy = (-1 *
               np.sum(np.log2(left_eigenvectors, out=np.zeros_like(left_eigenvectors), where=(left_eigenvectors != 0))
                      * left_eigenvectors, axis=1))/np.log2(hankel.shape[1])"""
    # skew = np.abs(skewness(left_eigenvectors, np.tile(np.linspace(0, hankel.shape[1]-1, hankel.shape[1]), (rank, 1))))
    skew = np.abs(left_right_diff(left_eigenvectors))

    # compute the weighted mean of the entropy
    # weighted_entropy = (eigenvalues @ ((1-entropy)*skew))/np.sum(eigenvalues)
    weighted_entropy = (eigenvalues @ skew) / np.sum(eigenvalues)

    return weighted_entropy


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
    esst_recognizer = ESST(50)

    # compute the score
    start = time()
    score = esst_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
