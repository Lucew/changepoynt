from typing import Callable
from functools import partial

import numpy as np

from changepoynt.utils import block_linalg as blg
from changepoynt.utils import normalization
import changepoynt.algorithms.sst as cpsst
from changepoynt.algorithms.base_algorithm import Algorithm


class MSST(Algorithm):
    """
    This class implements all the utility and functionality necessary to compute the Multivariate Singular Spectrum
    Transformation (MSST).

    This algorithm is an extension of the SST change point detection algorithm as described in:

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

    It is basically a merger of techniques from [1, 2] and:
    [3]
    Alanqary, Arwa, Abdullah Alomar, and Devavrat Shah.
    "Change point detection via multivariate singular spectrum analysis."
    Advances in Neural Information Processing Systems 34 (2021): 23218-23230.

    It also uses a technique called randomized singular value decomposition which is surveyed and described in

    [4]
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.

    For the option fast_hankel=True it uses and algorithm based on

    [5]
    L. Weber and R. Lenz.
    "Accelerating Singular Spectrum Transformation for Scalable Change Point Detection,"
    in IEEE Access, Volume 11, 2025.
    doi: 10.1109/ACCESS.2025.3640386.

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
                 method: str = 'ika', lanczos_rank: int = None, random_rank: int = None,
                 feedback_noise_level: float = 1e-3, scoring_step: int = 1, use_fast_hankel: bool = False) -> None:
        """
        Initializing the Singular Spectrum Transformation (SST) requires setting a lot of parameters. See the parameters
        explanation for some intuition into the right choices. Currently, there are two SST methods from [1] and [2]
        available for use.

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

        :param scale: Due to numeric stability, we REALLY RECOMMEND scaling the signal into a restricted value range.
        Per default, we use a min max scaling to ensure a restricted value range. In the presence of extreme outliers,
        this could cause problems, as the signal will be squished.

        :param method: Currently, "svd" [1], "ika" [2] and "rsvd" are available. ika corresponds to IKA-SST in [2] which
        will speed up the computation significantly.

        :param lanczos_rank: In order to use the implicit approximation of "eigensignals" by using "ika" [2] method,
        one needs to decide the rank of the implicit approximation of each "eigensignal". As a rule of thumb, we
        determine this value as being twice the amount of specified "eigensignals" to span the subspace
        (parameter rank). This is also recommended by the author of IKA-SST in [2].

        :param random_rank: In order to use the randomized singular value decomposition, one needs to provide a
        randomized rank (size of second matrix dimension for the randomized matrix) as specified in [3]. The lower
        this value, the faster the computation but the higher the error (as the approximation gets worse).

        :param feedback_noise_level: This specifies the amplitude of additive white gaussian noise added to the dominant
        "eigensignal" of the future behavior when shifting forward. This idea is noted in [2] and initializes
        the seed of the power method for dominant eigenvector estimation with the precious dominant eigenvector
        plus the noise level specified here. The noise level should just be a small fraction of the value range
        of the signal.

        :param scoring_step: the distance between scoring steps in samples (e.g., 2 would half the computation).

        :param use_fast_hankel: Use the O(N*logN) version for the decomposition. This is likely to have large
        speed improvements for window sizes > 150
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
        self.scoring_step = scoring_step
        self.use_fast_hankel = use_fast_hankel

        # set some default values when they have not been specified
        if self.n_windows is None:
            self.n_windows = self.window_length
        if self.lag is None:
            # rule of thumb
            self.lag = max(self.n_windows // 3, 1)
        if self.lanczos_rank is None:
            # make rank even and multiply by two just as specified in [2]
            self.lanczos_rank = self.rank * 2 - (self.rank & 1)
        if self.random_rank is None:
            # compute the rank as specified in [3] and
            # https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
            self.random_rank = min(self.rank + 10, self.window_length, self.n_windows)

        # specify the methods and their corresponding functions as lambda functions expecting only the
        # 1) future hankel matrix,
        # 2) the current hankel matrix and
        # 3) the feedback vector (e.g. for dominant eigenvector feedback)
        # all other parameter should be specified as values in the partial lambda function
        self.methods = {'ika': partial(cpsst._implicit_krylov_approximation,
                                       rank=self.rank,
                                       lanczos_rank=self.lanczos_rank),
                        'rsvd': partial(cpsst._random_singular_value_decomposition,
                                        rank=self.rank,
                                        randomized_rank=self.random_rank),
                        'weighted': partial(cpsst._weighted_random_singular_value_decomposition,
                                            rank=self.rank,
                                            randomized_rank=self.random_rank),
                        }
        if self.method not in self.methods:
            raise ValueError(f'Method {self.method} not defined. Possible methods: {list(self.methods.keys())}.')

        # set up the methods we use for the construction of the hankel matrix (either it is the fft representation
        # of the other one)
        if use_fast_hankel and self.method not in ["rsvd", "ika", "weighted"]:
            raise ValueError(f'{self.method} method is not defined with use_fast_hankel=True')

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        """
        This function calculates the anomaly score for each sample within the time series, starting from an initial
        sample being the first sample of fitting the past hankel matrix (window_length + n_windows samples) and the new
        future hankel matrix (+lag). Values before that will be zero

        It also does some assertions regarding the specified parameters and whether they fit the time series.

        This function builds the interface to the jit compiled static function used to iterate over the array.

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

        # get the changepoint scorer from the different methods
        scoring_function = self.methods[self.method]

        # start the scaling itself by calling the jit compiled staticmethod and return the result
        score = _transform(time_series=time_series, start_idx=starting_point, window_length=self.window_length,
                           n_windows=self.n_windows, lag=self.lag, scoring_step=self.scoring_step,
                           scoring_function=scoring_function, use_fast_hankel=True)
        return score


def _transform(time_series: np.ndarray, start_idx: int, window_length: int, n_windows: int, lag: int, scoring_step: int,
               scoring_function: Callable, use_fast_hankel: bool) -> np.ndarray:
    """
    Compute heavy and hopefully jit compilable score computation for the SST method. It does not do any parameter
    checking and can throw cryptic errors. It's only used for internal use as a private function.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: integer defining the start sample index for the score computation
    :param window_length: the size of the time series window for each column of the hankel matrix
    :param n_windows: number of columns in the hankel matrix
    :param lag: sample distance between future and past hankel matrix
    :param scoring_step: the distance between scoring steps in samples.
    :param scoring_function: the function that is called every step to assess a scalar change point score
    """

    # create initial vector for ika method with feedback dominant eigenvector as proposed in [2]
    # with a norm of one
    x0 = np.random.rand(window_length)[:, None]
    x0 /= np.linalg.norm(x0)

    # initialize a scoring array with no values yet
    score = np.zeros((time_series.shape[0],))

    # make an offset for the data construction
    offset = n_windows // 2 + lag

    # iterate over all the values in the signal starting at start_idx computing the change point score
    for idx in range(start_idx, time_series.shape[0], scoring_step):

        # compile the past hankel matrix (H1)
        hankel_past = blg.BlockHankelRepresentation(time_series, idx - lag, window_length, n_windows, use_fast_hankel)

        # compile the future hankel matrix (H2)
        hankel_future = blg.BlockHankelRepresentation(time_series, idx, window_length, n_windows, use_fast_hankel)

        # compute the score and save the returned feedback vector
        score[idx - offset - scoring_step // 2:idx - offset + (scoring_step + 1) // 2], x1 = \
            scoring_function(hankel_past, hankel_future, x0)

        # add noise to the dominant eigenvector and normalize it again
        x0 = x1 + 1e-3 * np.random.rand(x0.shape[0])[:, None]
        x0 /= np.linalg.norm(x0)

    return score


def main():
    """This function is not intended for users, but for quick testing during development."""
    from time import time

    # make synthetic step function
    np.random.seed(123)
    length = 300
    x = np.hstack([1 * np.ones(length) + np.random.rand(length) * 1,
                   3 * np.ones(length) + np.random.rand(length) * 2,
                   5 * np.ones(length) + np.random.rand(length) * 1.5])
    x += np.random.rand(x.size)
    x = x[..., None]

    # create the sst method
    ika_sst = MSST(31, method='ika', use_fast_hankel=True)
    rsvd_sst = MSST(31, method='rsvd', use_fast_hankel=True)
    weighted_sst = MSST(31, method='weighted', use_fast_hankel=True)

    # compare with original sst
    orig_ika = cpsst.SST(31, method='ika', use_fast_hankel=True).transform(x[:, 0])
    print('Comparison to original score', np.corrcoef(orig_ika, ika_sst.transform(x))[0, 1])

    # make the scoring
    start = time()
    ika_sst.transform(x)
    print(time() - start)
    rsvd_sst.transform(x)
    weighted_sst.transform(x)


if __name__ == '__main__':
    main()
