import os
from typing import Callable
from functools import partial

import numba as nb
import numpy as np

from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.utils import normalization
import changepoynt.utils.block_linalg as blg
import changepoynt.algorithms.esst as esst


class MESST(Algorithm):
    """
    This class implements the Multivariate Enhanced Singular Spectrum Transformation (MESST)

    The univariate algorithm ESST has been published together with Sarah Boelter (UMN)
    Boelter, Sarah*, Lucas Weber*, et al. "Fault Prediction in Planetary Drilling Using Subspace Analysis Techniques."
    International Conference on Intelligent Autonomous Systems (IAS-19). 2025.
    (* is equal contribution)
    https://ntrs.nasa.gov/citations/20250002705

    It is a research algorithm, please use with caution.
    """

    def __init__(self, window_length: int, n_windows: int = None, lag: int = None, rank: int = 5,
                 scale: bool = True, method: str = 'rsvd', random_rank: int = None, scoring_step: int = 1,
                 use_fast_hankel: bool = False, threads: int = None) -> None:
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

        :param use_fast_hankel: Whether to deploy the fast hankel matrix product.

        :param threads: The number of threads the fast hankel matrix product is allowed to use. Default is the half of
        the number of cpu cores your system has available.
        """

        # save the specified parameters into instance variables
        self.window_length = window_length
        self.n_windows = n_windows
        self.rank = rank
        self.scale = scale
        self.random_rank = random_rank
        self.lag = lag
        self.scoring_step = scoring_step
        self.use_fast_hankel = use_fast_hankel
        self.method = method
        self.threads = threads

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
        self.methods = {'rsvd': partial(esst.left_entropy, rank=self.rank, random_rank=self.random_rank,
                                        threads=self.threads, method=self.method)}
        if self.method not in self.methods:
            raise ValueError(f'Method {self.method} not defined. Possible methods: {list(self.methods.keys())}.')

        # set up the methods we use for the construction of the hankel matrix (either it is the fft representation
        # of the other one)
        if use_fast_hankel and self.method != 'rsvd':
            raise ValueError(f'method {self.method} is not defined with use_fast_hankel=True')

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        This function calculates the anomaly score for each sample within the time series.

        It also does some assertion regarding time series length.

        :param time_series: 1D array containing the time series to be scored
        :return: anomaly score
        """

        # check the dimensions of the input array
        assert time_series.ndim > 1, "Time series needs to be an N-D array. Currently it is 1-D."

        # compute the starting point of the scoring (past and future hankel need to fit)
        starting_point = self.window_length + self.n_windows + self.lag
        assert starting_point < time_series.shape[0], "The time series is too short to score any points."

        # scale the time series (or just copy it if already scaled)
        time_series = time_series.copy()
        if self.scale:
            for idx in range(time_series.shape[1]):
                time_series[:, idx] = normalization.min_max_scaling(time_series[:, idx], 1, 2, inplace=True)

        # get the different methods
        scoring_function = self.methods[self.method]
        return _transform(time_series, starting_point, self.window_length, self.n_windows, self.lag,
                          self.scoring_step, scoring_function, self.use_fast_hankel)


def _transform(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int, scoring_step: int,
               scoring_function: Callable, use_fast_hankel: bool) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: column number of the hankel matrix
    """

    # initialize a scoring array with no values yet
    score = np.zeros((time_series.shape[0],))

    # compute the offset
    offset = (n_windows + lag)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0], scoring_step):

        # compile the past hankel matrix (H1)
        hankel_past = blg.BlockHankelRepresentation(time_series, idx - lag, window_length, n_windows, use_fast_hankel)

        # compile the future hankel matrix (H2)
        hankel_future = blg.BlockHankelRepresentation(time_series, idx, window_length, n_windows, use_fast_hankel)

        # compute the score and save the returned feedback vector
        score[idx-offset-scoring_step//2:idx-offset+(scoring_step+1)//2] = \
            scoring_function(np.concatenate((hankel_past, hankel_future), axis=1))

    return score


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
    messt_recognizer = MESST(70, method='rsvd', use_fast_hankel=False)
    esst_recognizer = esst.ESST(70, method='rsvd', use_fast_hankel=False)

    # compute the score
    start = time()
    # score1 = messt_recognizer.transform(np.concatenate((x[..., None], x[..., None]), axis=1))
    score1 = messt_recognizer.transform(x[..., None])
    print(f'Computation for {len(x)} signal values took {time()-start} s.')

    # check for similarity with the esst
    score2 = esst_recognizer.transform(x)
    print(score1[1000:1020])
    print(score2[1000:1020])
    print(score1[1000:1020]-score2[1000:1020])
    print(np.allclose(score1, score2))
    print(np.corrcoef(score1, score2)[0, 1])


if __name__ == '__main__':
    _main()
