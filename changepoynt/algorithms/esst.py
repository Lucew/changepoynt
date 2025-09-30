import os

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from functools import partial
from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.utils import normalization
from changepoynt.utils import linalg as lg
import fbpca
import multiprocessing as mp
import numba as nb


class ESST(Algorithm):
    """
    This class implements an own idea for change point detection.

    It has been published together with Sarah Boelter (UMN)
    Boelter, Sarah*, Lucas Weber*, et al. "Fault Prediction in Planetary Drilling Using Subspace Analysis Techniques."
    International Conference on Intelligent Autonomous Systems (IAS-19). 2025.
    (* is equal contribution)
    https://ntrs.nasa.gov/citations/20250002705

    It is a research algorithm, please use with caution.
    """

    def __init__(self, window_length: int, n_windows: int = None, lag: int = None, rank: int = 5,
                 scale: bool = True, method: str = 'fbrsvd', random_rank: int = None, scoring_step: int = 1,
                 parallel: bool = False, use_fast_hankel: bool = False, threads: int = None,
                 mitigate_offset: bool = False) -> None:
        """
        Experimental change point detection method evaluation the prevalence of change points within a signal
        by comparing the difference in eigenvectors between to points in time.

        :param window_length: This specifies the length of the time series (in samples), which will be used to extract
        the representative "eigensignals" to be compared before and after the lag. The window length should be big
        enough to cover any wanted patterns (e.g., bigger than the periodicity of periodic signals).

        :param n_windows: This specifies the number of consecutive time windows used to extract the "eigensignals" from
        the given time series. It should be big enough to cover different parts of the target behavior. If one does not
        specify this value, we use the rule of thumb and take as many time windows as you specified the length of the
        window (rule of thumb).

        :param lag: This parameter specifies the distance of the comparison for behaviors. In easy terms it tells the
        algorithms how far it should look into the future to find change in behavior to the current signal. If you do
        not specify this parameter, we use a rule of thumb and look ahead half of the window length you specified to
        cover the behavior.

        :param rank: This parameter specifies the amount of "eigensignals" which will be used to measure the
        dissimilarity of the signal in the future behavior. As a rule of thumb, we take the five most dominant
        "eigensignals" if you do not specify otherwise.

        :param scale: Due to numeric stability, we REALLY RECOMMEND scaling the signal into a restricted value range. Per
        default, we use a min max scaling to ensure a restricted value range. In the presence of extreme outliers, this
        could cause problems, as the signal will be squished.

        :param random_rank: To use the randomized singular value decomposition, one needs to provide a
        randomized rank (size of second matrix dimension for the randomized matrix) as specified in [3]. The lower
        this value, the faster the computation but the higher the error (as the approximation gets worse).

        :param scoring_step: The distance between scoring steps in samples (e.g., 2 would half the computation).

        :param parallel: The execution for the different steps can be parallelized in different processes.

        :param use_fast_hankel: Whether to deploy the fast hankel matrix product.

        :param threads: The number of threads the fast hankel matrix product is allowed to use. Default is the half of
        the number of cpu cores your system has available.

        :param mitigate_offset: Use a sliding mean window to mitigate the constant offset of time series.
        """

        # save the specified parameters into instance variables
        self.window_length = window_length
        self.n_windows = n_windows
        self.rank = rank
        self.scale = scale
        self.random_rank = random_rank
        self.lag = lag
        self.scoring_step = scoring_step
        self.parallel = parallel
        self.use_fast_hankel = use_fast_hankel
        self.method = method
        self.threads = threads
        self.mitigate_offset = mitigate_offset

        # set some default values when they have not been specified
        if self.n_windows is None:
            self.n_windows = self.window_length//2
        if self.lag is None:
            # rule of thumb
            self.lag = self.n_windows
        if self.random_rank is None:
            # compute the rank as specified in [3] and
            # https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
            self.random_rank = min(self.rank + 10, self.window_length, self.n_windows)
        if self.threads is None:
            self.threads = os.cpu_count()//2

        # specify the methods and their corresponding functions as lambda functions expecting only the hankel matrix
        self.methods = {'rsvd': partial(left_entropy, rank=self.rank, random_rank=self.random_rank,
                                        method='rsvd', threads=self.threads),
                        'fbrsvd': partial(left_entropy, rank=self.rank, random_rank=self.random_rank,
                                          method='fbrsvd', threads=self.threads)}
        if self.method not in self.methods:
            raise ValueError(f'Method {self.method} not defined. Possible methods: {list(self.methods.keys())}.')

        # set up the methods we use for the construction of the hankel matrix (either it is the fft representation
        # of the other one)
        if use_fast_hankel and self.method == 'fbrsvd':
            raise ValueError(f'fbrsvd method is not defined with use_fast_hankel=True')
        self.hankel_construction = {
            False: lg.compile_hankel,
            True: lg.HankelFFTRepresentation
        }
        # TODO: Create tests for the ESST

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        This function calculates the anomaly score for each sample within the time series.

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

        # get the different methods
        scoring_function = self.methods[self.method]
        hankel_function = self.hankel_construction[self.use_fast_hankel]

        if self.parallel:
            return _transform_parallel(time_series, starting_point, self.window_length, self.n_windows, self.lag,
                                       self.scoring_step, scoring_function, hankel_function, self.mitigate_offset)
        else:
            return _transform(time_series, starting_point, self.window_length, self.n_windows, self.lag,
                              self.scoring_step, scoring_function, hankel_function, self.mitigate_offset)


def _transform(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int, scoring_step: int,
               scoring_function: Callable, hankel_construction_function: Callable,
               mitigate_offset: bool = False) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: column number of the hankel matrix
    """

    # initialize a scoring array with no values yet
    score = np.zeros_like(time_series)

    # compute the offset
    offset = (n_windows + lag)

    # compute the sliding mean value for the complete time series using a convolution
    # we get the length of the sliding window from the start index
    # the sliding window contains both hankel matrices
    current_min = None
    if mitigate_offset:
        current_min = np.median(np.lib.stride_tricks.sliding_window_view(time_series, window_shape=start_idx), axis=1)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0], scoring_step):

        # get the constant offset (is zero if the option is deactivated)
        if mitigate_offset:
            const_offset = current_min[idx - start_idx] - 1
        else:
            const_offset = None

        # compile the past hankel matrix (H1)
        hankel_past = hankel_construction_function(time_series, idx - lag, window_length, n_windows,
                                                   const_offset=const_offset)

        # compile the future hankel matrix (H2)
        hankel_future = hankel_construction_function(time_series, idx, window_length, n_windows,
                                                     const_offset=const_offset)

        # compute the score and save the returned feedback vector
        score[idx-offset-scoring_step//2:idx-offset+(scoring_step+1)//2] = \
            scoring_function(np.concatenate((hankel_past, hankel_future), axis=1))

    return score


def _transform_parallel(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int,
                        scoring_step: int, scoring_function: Callable,
                        hankel_construction_function: Callable, mitigate_offset: bool = False) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: column number of the hankel matrix
    """

    # initialize a scoring array with no values yet
    score = np.zeros_like(time_series)

    # compute the offset
    offset = (n_windows + lag)

    # make the generator for the hankel matrices
    gener = (np.concatenate(
                            (hankel_construction_function(time_series, idx-lag, window_length, n_windows),
                             hankel_construction_function(time_series, idx, window_length, n_windows)
                             ),
                            axis=1)
             for idx in range(start_idx, time_series.shape[0], scoring_step))

    # make a process pool with batches
    with mp.Pool(mp.cpu_count()) as pp:
        for idx, result in enumerate(pp.imap(scoring_function, gener, chunksize=100)):
            idx = start_idx + idx*scoring_step
            score[idx - offset - scoring_step // 2:idx - offset + (scoring_step + 1) // 2] = result

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


def left_entropy(hankel: np.ndarray, rank: int, random_rank: int, method: str, threads: int) -> float:
    """
    Entropy Singular Spectrum Transformation.

    :param hankel: the hankel matrix of the signal
    :param rank: the number of (approximated) eigenvectors as subspace of H1
    :param random_rank: the sampling parameter for the randomized svd
    :param method: which rsvd method to use
    :param threads: the numba of threads numba is allowed to use
    :return: the change point score, the input vector x0
    """
    # compute the left and right eigenvectors using the randomized svd for fast computation
    threads_before = nb.get_num_threads()
    nb.set_num_threads(threads)
    if method == 'fbrsvd':
        right_eigenvectors, eigenvalues, left_eigenvectors = fbpca.pca(hankel, rank, True)
    elif method == 'rsvd':
        right_eigenvectors, eigenvalues, left_eigenvectors = lg.randomized_hankel_svd(hankel, rank,
                                                                                      oversampling_p=random_rank - rank)
    else:
        raise NotImplementedError(f'Method {method} is not available.')
    nb.set_num_threads(threads_before)

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
    x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 3000))
    x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, 3000))
    x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, 3000))
    x3 = np.sin(2 * np.pi * 4 * np.linspace(0, 10, 3000))
    x = np.hstack([x0, x1, x2, x3])
    x += np.random.rand(x.size)

    # create the method
    esst_recognizer = ESST(300, method='rsvd', parallel=False, use_fast_hankel=True)

    # compute the score
    start = time()
    score1 = esst_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
