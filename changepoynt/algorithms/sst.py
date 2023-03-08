import numpy as np
from numba import jit
from changepoynt.utils import linalg as lg
from changepoynt.utils import normalization
from typing import Callable
from functools import partial


class SingularSpectrumTransformation:
    """
    This class implements all the utility and functionality necessary to compute the SST change point detection
    algorithm as described in:

    [1]
    Idé, Tsuyoshi, and Keisuke Inoue.
    "Knowledge discovery from heterogeneous dynamic systems using change-point correlations."
    Proceedings of the 2005 SIAM international conference on data mining.
    Society for Industrial and Applied Mathematics, 2005.

    [2]
    Idé, Tsuyoshi, and Koji Tsuda.
    "Change-point detection using krylov subspace learning."
    Proceedings of the 2007 SIAM International Conference on Data Mining.
    Society for Industrial and Applied Mathematics, 2007.

    It also uses a technique called randomized singular value decomposition which is surveyed and described in

    [3]
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.

    There will be a parameter specifying, whether to use the implicit krylov approximation from [2]. This
    significantly speeds up the computation but can reduce accuracy as the "eigensignals" are only approximated
    indirectly using a krylov subspace with the possible change point signal as seed. This method also deploys
    a feedback of the dominant "eigensignal" as seed into the power method (dominant eigenvector estimation)
    with additive gaussian noise.

    !NOTE!:
    Most computational heavy functions are implemented as standalone functions, even if they require instance variables
    as this enables us to use jit compiled code provided by the numba compiler for faster calculations.

    Also, we decided to do type hinting but not type checking as it requires too much boilerplate code. We recommend
    to input the specified types as the program will break in unforeseen ways otherwise.
    """

    def __init__(self, window_length: int, n_windows: int = None, lag: int = None, rank: int = 5, scale: bool = True,
                 method: str = 'ika', lanczos_rank: int = None, random_rank: int = 10,
                 feedback_noise_level: float = 1e-3) -> None:
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

        :param method: Currently "svd" [1] and "ika" [2] are available. ika corresponds to IKA-SST in [2] which
        will speed up the computation significantly.

        :param lanczos_rank: In order to use the implicit approximation of "eigensignals" by using "ika" [2] method
        one needs to decide the rank of the implicit approximation of each "eigensignal". As a rule of thumb we
        determine this value as beeing twice the amount of specified "eigensignals" to span the subspace
        (parameter rank). This is also recommend by the author of IKA-SST in [2].

        :param random_rank: In order to use the randomized singular value decomposition, one needs to provide a
        randomized rank (size of second matrix dimension for the randomized matrix) as specified in [3]. The lower
        this value, the faster the computation but the higher the error (as the approxiamtion gets worse).

        :param feedback_noise_level: This specifies the amplitude of additive white gaussian noise added to the dominant
        "eigensignal" of the future behavior when shifting forward. This idea is noted in [2] and initializes
        the seed of the power method for dominant eigenvector estimation with the precious dominant eigenvector
        plus the noise level specified here. The noise level should just be a small fraction of the value range
        of the signal.
        """

        # save the specified parameters into instance variables
        self.window_length = window_length
        self.n_windows = n_windows
        self.lag = lag
        self.rank = rank
        self.scale = scale
        self.method = method
        self.lanczos_rank = lanczos_rank
        self.random_rank = random_rank
        self.noise = feedback_noise_level

        # set some default values when they have not been specified
        if self.n_windows is None:
            self.n_windows = self.window_length
        if self.lag is None:
            self.lag = max(self.n_windows//2, 1)
        if self.lanczos_rank is None:
            # make rank even and multiply by two just as specified in [2]
            self.lanczos_rank = self.rank * 2 - (self.rank & 1)
        if self.random_rank is None:
            # compute the rank as specified in [3] and
            # https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
            self.random_rank = self.rank + 10

        # specify the methods and their corresponding functions as lambda functions expecting only the
        # 1) future hankel matrix,
        # 2) the current hankel matrix and
        # 3) the feedback vector (e.g. for dominant eigenvector feedback)
        # all other parameter should be specified as values in the partial lambda function
        self.methods = {'ika': partial(_implicit_krylov_approximation,
                                       rank=self.rank,
                                       lanczos_rank=self.lanczos_rank),
                        'svd': partial(_singular_value_decomposition,
                                       rank=self.rank),
                        'rsvd': partial(_randomized_singular_value_decomposition,
                                        rank=self.rank,
                                        randomized_rank=self.lanczos_rank
                                        )}

        # check whether the method is correct
        assert self.method in self.methods, f'Specified method {self.method} is not available in {self.methods}.'

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

    # create initial vector for ika method with feedback dominant eigenvector as proposed in [2]
    # with a norm of one
    x0 = np.random.rand(window_length)
    x0 /= np.linalg.norm(x0)

    # initialize a scoring array with no values yet
    score = np.zeros_like(time_series)

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0]):

        # compile the past hankel matrix (H1)
        hankel_past = _compile_hankel(time_series, idx-lag, window_length, n_windows)

        # compile the future hankel matrix (H2)
        hankel_future = _compile_hankel(time_series, idx, window_length, n_windows)

        # compute the score and save the returned feedback vector
        score[idx], x1 = scoring_function(hankel_past, hankel_future, x0)

        # add noise to the dominant eigenvector and normalize it again
        x0 = x1 + 1e-3 * np.random.rand(x0.size)
        x0 /= np.linalg.norm(x0)

    return score


@jit(nopython=True)
def _compile_hankel(time_series: np.ndarray, end_index: int, window_size: int, rank: int) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param rank: the amount of time series in the matrix
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, rank))

    # go through the time series and make the hankel matrix
    for cx in range(rank):
        hankel[:, cx] = time_series[(end_index-window_size-cx):(end_index-cx)]
    return hankel


def _implicit_krylov_approximation(hankel_past: np.ndarray, hankel_future: np.ndarray, x0: np.ndarray,
                                   rank: int, lanczos_rank: int) -> (float, np.ndarray):
    """
    This function computes the change point score based on the krylov subspace approximation of the SST as proposed in

    Idé, Tsuyoshi, and Koji Tsuda.
    "Change-point detection using krylov subspace learning."
    Proceedings of the 2007 SIAM International Conference on Data Mining.
    Society for Industrial and Applied Mathematics, 2007.

    Due to perfomance reason this function makes no sanity checks for matrix size and parameter validity.

    :param hankel_past: the hankel matrix H1 before the change point
    :param hankel_future: the hankel matrix H2 after the change point
    :param x0: the initialization value for the power method applied to H2 to find the dominant eigenvector
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :param lanczos_rank: the rank of the approximation of the "eigensignals"
    :param feedback_noise_level: the noise that will be added to the dominant eigenvector after we are finished
    :return: the change point score, the new dominant eigenvector of H2 for the feedback into the next H2
    """

    # compute the biggest eigenvector of the hankel matrix after the possible change point (h2)
    c_2 = hankel_future.T @ hankel_future
    _, eigvec_future = lg.power_method(c_2, x0, n_iterations=1)

    # compute the empirical covariance matrix before the possible change point (H1)
    c_1 = hankel_past.T @ hankel_past

    # compute the tridiagonal matrix from c1
    alphas, betas = lg.lanczos(c_1, eigvec_future, lanczos_rank)

    # compute the singular value decomposition of the tridiagonal matrix (only the biggest)
    _, eigvecs = lg.tridiagonal_eigenvalues(alphas, betas, rank)

    # compute the similarity score as defined in the ika sst paper and also return our u for the
    # feedback loop in figure 3 of the paper
    return 1 - (eigvecs[0, :] * eigvecs[0, :]).sum(), eigvec_future


def _singular_value_decomposition(hankel_past: np.ndarray, hankel_future: np.ndarray, x0: np.ndarray,
                                  rank: int) -> (float, np.ndarray):
    """
    This function implements change point detection using singular value decomposition as proposed in:

    Idé, Tsuyoshi, and Keisuke Inoue.
    "Knowledge discovery from heterogeneous dynamic systems using change-point correlations."
    Proceedings of the 2005 SIAM international conference on data mining.
    Society for Industrial and Applied Mathematics, 2005.

    :param hankel_past: the hankel matrix H1 before the change point
    :param hankel_future: the hankel matrix H2 after the change point
    :param x0: highest eigenvector of previous iteration (will be ignored in this function and is just added to complete
    the function signature, i.e. input and output size, to be compatible with other methods)
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :return: the change point score, the input vector x0
    """

    # compute the rank highest eigenvectors of the past hankel matrix
    _, eigvecs_past = lg.highest_k_eigenvectors(hankel_past, rank)

    # compute the dominant eigenvector of the future time series
    x0 = np.random.rand(hankel_future.shape[1])
    x0 /= np.linalg.norm(x0)
    c_2 = hankel_future.T @ hankel_future
    _, eigvec_future = lg.power_method(c_2, x0, n_iterations=20)

    # compute the projection distance
    alpha = eigvecs_past.T @ eigvec_future
    return 1 - alpha.T @ alpha, x0


def _randomized_singular_value_decomposition(hankel_past: np.ndarray, hankel_future: np.ndarray, x0: np.ndarray,
                                             rank: int, randomized_rank: int):
    """
    This function implements the idea of

    Idé, Tsuyoshi, and Keisuke Inoue.
    "Knowledge discovery from heterogeneous dynamic systems using change-point correlations."
    Proceedings of the 2005 SIAM international conference on data mining.
    Society for Industrial and Applied Mathematics, 2005.

    but uses a feedback vector for the future dominant eigenvector as proposed in

    Idé, Tsuyoshi, and Koji Tsuda.
    "Change-point detection using krylov subspace learning."
    Proceedings of the 2007 SIAM International Conference on Data Mining.
    Society for Industrial and Applied Mathematics, 2007.

    in contrast to the first two papers, the method uses randomized singular value decomposition.

    :param hankel_past: the hankel matrix H1 before the change point
    :param hankel_future: the hankel matrix H2 after the change point
    :param x0: the initialization value for the power method applied to H2 to find the dominant eigenvector
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :param randomized_rank: the rank of the approximation used to construct the noise matrix
    :return: the change point score, the new dominant eigenvector of H2 for the feedback into the next H2
    """

    # compute the biggest eigenvector of the hankel matrix after the possible change point (h2)
    c_2 = hankel_future.T @ hankel_future
    _, eigvec_future = lg.power_method(c_2, x0, n_iterations=1)

    # compute the eigenvectors of the past hankel matrix
    c_1 = hankel_past.T @ hankel_past
    _, eigenvecs_past = lg.randomized_singular_value_decomposition(c_1, randomized_rank=randomized_rank)

    # compute the change point score as defined in the papers
    alpha = eigenvecs_past[:, :rank].T @ eigvec_future
    return 1-alpha.T @ alpha, eigvec_future


if __name__ == '__main__':

    # make synthetic step function
    np.random.seed(123)
    length = 300
    x = np.hstack([1 * np.ones(length) + np.random.rand(length) * 1,
                   3 * np.ones(length) + np.random.rand(length) * 2,
                   5 * np.ones(length) + np.random.rand(length) * 1.5])
    x += np.random.rand(x.size)

    # create the sst method
    ika_sst = SingularSpectrumTransformation(31, method='ika')
    svd_sst = SingularSpectrumTransformation(31, method='svd')

    # make the scoring
    ika_sst.transform(x)
    svd_sst.transform(x)
