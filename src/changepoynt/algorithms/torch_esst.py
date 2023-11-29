import numpy as np
from typing import Callable
from functools import partial

import torch

from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.utils import normalization
from changepoynt.utils import linalg as lg


class TESST(Algorithm):
    """
    This class implements an own idea for change point detection.

    It is not yet published, please use with caution and do not copy or use in production (Lucas Weber, 2023).
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

        # check whether we have cuda
        assert torch.cuda.is_available(), 'A torch installation with cuda is necessary for this.'

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

    # compute the batched hankel matrix
    batched_hankel = np.zeros(((time_series.shape[0]-start_idx)//scoring_step, window_length, 2*n_windows))

    # iterate over all the values in the signal and construct the hankel matrix
    for idx in range(start_idx, time_series.shape[0], scoring_step):

        # compile the past hankel matrix (H1)
        hankel_past = lg.compile_hankel(time_series, idx - lag, window_length, n_windows)

        # compile the future hankel matrix (H2)
        hankel_future = lg.compile_hankel(time_series, idx, window_length, n_windows)

        # put the hankel matrix into the batched hankel matrix
        batched_hankel[idx-start_idx, :, :] = np.concatenate((hankel_past, hankel_future), axis=1)

    # transform into and torch tensor and put onto the gpu
    with torch.no_grad():
        batched_hankel = torch.from_numpy(batched_hankel).to('cuda:0')
        print(batched_hankel.shape)
        print(f'Tensor is on the GPU: {torch.cuda.get_device_name(0)}')

        # compute the score
        import time
        start = time.time()
        cuda_score = scoring_function(batched_hankel)
        print(f'Computation on GPU took {time.time() - start} s.')

        # repeat the numbers and get back to cpu
        cuda_score = cuda_score.repeat_interleave(scoring_step).cpu().numpy()
        score[start_idx-offset-scoring_step//2:-offset] = cuda_score

    return score


def left_right_diff(left_eigenvectors: torch.Tensor):
    n = left_eigenvectors.shape[1]//2
    return torch.mean(left_eigenvectors[:, :n, :]-left_eigenvectors[:, n:, :], dim=1)


def left_entropy(hankel: torch.Tensor, rank: int) -> float:
    """
    Entropy Singular Spectrum Transformation.

    :param hankel: the hankel matrix of the signal
    :param rank: the amount of (approximated) eigenvectors as subspace of H1
    :return: the change point score, the input vector x0
    """
    # compute the left and right eigenvectors using the randomized svd for fast computation
    left_eigenvectors, eigenvalues, right_eigenvectors = torch.svd_lowrank(hankel, rank)

    # compute the difference between left and right participation of eigenvectors
    difference = torch.abs(left_right_diff(right_eigenvectors))

    # compute the weighted mean of the difference
    weighted_difference = torch.sum(difference*eigenvalues, dim=1) / torch.sum(eigenvalues)

    return weighted_difference


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
    esst_recognizer = TESST(50)

    # compute the score
    start = time()
    score = esst_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
