import numpy as np
import scipy as sp
from changepoynt.utils import linalg as lg
from changepoynt.utils import normalization
from typing import Callable
from functools import partial


class SubspaceIdentification:
    """
    This class implements all the utility and functionality necessary to compute the subspace change point detection
    algorithm as described in (offline method)

    [1]
    "Change-Point Detection in Time-Series Data based on Subspace Identification"
    Yoshinobu Kawahara, Takehisa Yairi, and Kazuo Machida
    Seventh IEEE International Conference on Data Mining, 2007

    TODO:
    Implement the online recursive algorithm.

    We decided to do type hinting but not type checking as it requires too much boilerplate code. We recommend
    to input the specified types as the program will break in unforeseen ways otherwise.
    """

    def __init__(self, window_length: int, n_windows: int = None, lag: int = None, rank: int = 5, scale: bool = True,
                 method: str = 'offline', random_rank: int = None) -> None:
        """
        Initializing the Singular Spectrum Transformation (SST) requires setting a lot of parameters. See the parameters
        explanation for some intuition into the right choices. Currently, there are two SST methods from [1] and [2]
        available for use.

        :param window_length: This specifies the length of the time series (in samples), which will be used to extract
        the representative "eigensignals" to be compared before and after the lag. The windows length should be big
        enough to cover any wanted patterns (e.g. bigger than the periodicity of periodic signals).

        :param n_windows: This specifies the amount of consecutive time windows used to extract the "eigensignals" from
        the given time series. It should be big enough to cover different parts of the target behavior. If one does not
        specify this value, we use the rule of thumb and take as many time windows as you specified the length of the
        window (rule of thumb).

        :param lag: This parameter specifies the distance of the comparison for behaviors. In easy terms it tells the
        algorithms how far it should look into the future to find change in behavior to the current signal. If you do
        not specify this parameter, we use a rule of thumb and look ahead half of the window length you specified to
        cover the behavior.

        :param rank: This parameter specifies the amount of "eigensignals" which will be used to measure the
        dissimilarity of the signal in the future behavior. As a rule of thumb we take the five most dominant
        "eigensignals" if you do no specify otherwise.

        :param scale: Due to numeric stability we REALLY RECOMMEND scaling the signal into a restricted value range. Per
        default, we use a min max scaling to ensure a restricted value range. In the presence of extreme outliers this
        could cause problems, as the signal will be squished.

        :param method: Currently only the offline method of [1] is available.

        :param random_rank: In order to use the randomized singular value decomposition, one needs to provide a
        randomized rank (size of second matrix dimension for the randomized matrix) as specified in [3]. The lower
        this value, the faster the computation but the higher the error (as the approximation gets worse).
        """
        raise NotImplementedError('SI is not yet available.')
        # save the specified parameters into instance variables
        self.window_length = window_length
        self.n_windows = n_windows
        self.lag = lag
        self.rank = rank
        self.scale = scale
        self.method = method
        self.random_rank = random_rank

        # set some default values when they have not been specified
        if self.n_windows is None:
            self.n_windows = self.window_length
        if self.lag is None:
            # rule of thumb
            self.lag = max(self.n_windows//3, 1)
        if self.random_rank is None:
            # compute the rank as specified in [3] and
            # https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
            self.random_rank = self.rank + 10

        # specify the methods and their corresponding functions as lambda functions expecting only the
        # 1) future hankel matrix,
        # 2) the current hankel matrix and
        # 3) the feedback vector (e.g. for dominant eigenvector feedback)
        # all other parameter should be specified as values in the partial lambda function
        self.methods = {'offline': partial(_subspace_identification,
                                           rank=self.rank,
                                           randomized_rank=self.random_rank)
                        }

        # check whether the method is correct
        assert self.method in self.methods, f'Specified method {self.method} is not available in {self.methods.keys()}.'

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        This function calculate the anomaly score for each sample within the time series starting from an initial sample
        being the first sample of fitting the past hankel matrix (window_length + n_windows samples) and the new future
        hankel matrix (+lag). Values before that will be zero

        It also does some assertions regarding the specified parameters and whether they fit the time series.

        This function builds the interface to the jit compiled static function used to iterate over the array.

        :param time_series: 1D array containing the time series to be scored
        :return: anomaly score
        """

        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # compute the starting point of the scoring (past and future hankel need to fit)
        starting_point = self.window_length + self.n_windows + self.lag
        assert starting_point < time_series.shape[0], "The time series is too short to score any points."

        # scale the time series (or just copy it if already scaled)
        if self.scale:
            time_series = normalization.min_max_scaling(time_series, min_val=1.0, max_val=2.0, inplace=False)
        else:
            time_series = time_series.copy()

        # get the changepoint scorer from the different methods
        scoring_function = self.methods[self.method]

        # start the scaling itself by calling the jit compiled staticmethod and return the result
        score = _transform(time_series=time_series, start_idx=starting_point, window_length=self.window_length,
                           n_windows=self.n_windows, lag=self.lag, scoring_function=scoring_function)
        return score


def _transform(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int,
               scoring_function: Callable) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: amount of columns in the hankel matrix
    :param lag: sample distance between future and past hankel matrix
    """

    # initialize a scoring array with no values yet
    score = np.zeros_like(time_series)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0]):

        # compile the past hankel matrix (H1)
        hankel_past = lg.compile_hankel(time_series, idx-lag, window_length, n_windows)

        # compile the future hankel matrix (H2)
        hankel_future = lg.compile_hankel(time_series, idx, window_length, n_windows)

        # compute the score and save the returned feedback vector
        score[idx] = scoring_function(hankel_past, hankel_future)

    return score


def _subspace_identification(hankel_past: np.ndarray, hankel_future: np.ndarray, hanke_test,
                             rank: int, randomized_rank: int) -> (float, np.ndarray):
    """
    This class implements all the utility and functionality necessary to compute the subspace change point detection
    algorithm as described in (offline method)

    [1]
    "Change-Point Detection in Time-Series Data based on Subspace Identification"
    Yoshinobu Kawahara, Takehisa Yairi, and Kazuo Machida
    Seventh IEEE International Conference on Data Mining, 2007

    param hankel_past: the hankel matrix H1 before the change point
    :param hankel_future: the hankel matrix H2 after the change point
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :param randomized_rank: the rank of the approximation used to construct the noise matrix
    :return: the change point score
    """

    # compute the covariance matrix for each time window and their cross covariance
    cov_pp = hankel_past.T @ hankel_past
    cov_ff = hankel_future.T @ hankel_future
    cov_fp = hankel_future.T @ hankel_past

    # compute the inverse square root of the matrices
    cov_pp = np.linalg.inv(sp.linalg.sqrtm(cov_pp))
    cov_ff = np.linalg.inv(sp.linalg.sqrtm(cov_ff))

    # compute the observability matrix of the system
    observability = cov_ff @ cov_fp @ cov_pp.T

    # compute the highest eigenvectors of the observability matrix
    _, eigenvectors = lg.randomized_singular_value_decomposition(observability, randomized_rank=randomized_rank)
    eigenvectors = eigenvectors[:, :rank] @ eigenvectors[:, :rank].T

    # compute the change point score
    return sum(vec.T @ vec - vec.T @ eigenvectors @ vec for vec in hankel_future.T)


if __name__ == '__main__':

    # make synthetic step function
    np.random.seed(123)
    length = 300
    x = np.hstack([1 * np.ones(length) + np.random.rand(length) * 1,
                   3 * np.ones(length) + np.random.rand(length) * 2,
                   5 * np.ones(length) + np.random.rand(length) * 1.5])
    x += np.random.rand(x.size)

    # create the sst method
    si = SubspaceIdentification(31)

    # make the scoring
    si.transform(x)