"""
This file contains the interface for the base algorithm used in the package.
"""
from abc import ABC, abstractmethod, abstractproperty
import time

import numpy as np


class Algorithm(ABC):

    @abstractmethod
    def transform(self, time_series: np.ndarray):
        raise NotImplementedError


class SingularSubspaceAlgorithm(Algorithm):

    # some parameters defined in the subclasses
    window_length: int
    n_windows: int
    lag: int
    scoring_step: int

    def covered_regions(self) -> tuple[int, int]:
        """
        This function returns the size of the region of interest (ROI). The ROI is the total amount of samples covered
        for the score detection.

        The covered region depends on the window size, the number of windows, and the lag between the two Hankel
        matrices (past and future Hankel matrices).

        The function return two different sizes:

        1) total_region: The total region covered by both matrices with lag. This is also the minimum number of samples
        that a time series must have before the subspace method can run.

        2) matrix_region: The region covered by one Hankel matrix. Essentially the region that is sampled for
        subsequences that are used to extract the characteristics

        :return: total_region: int, matrix_region: int
        """

        # the number of samples one Hankel matrix covers
        matrix_region = self.window_length + self.n_windows - 1

        # we have two Hankel matrices that might overlap
        total_region = matrix_region + self.lag

        return total_region, matrix_region

    def estimate_runtime(self, signal: np.ndarray, steps: int = 30, verbose: bool = False) -> [float, float]:
        """
        This function estimates the runtime for a given signal by running the initial steps.

        Singular subspace algorithms grow linearly with the signal length of the time series, the overall runtime
        is number of steps times signal length.

        :param signal: The signal that you will run the algorithm on.

        :param steps: The number of estimation steps. Increasing this number increases the runtime
                      but also increases the accuracy of the estimation.

        :param verbose: If true, prints out the runtime of the algorithm.

        :return: Runtime estimation in seconds, standard deviation in seconds
        """

        # get the required signal length
        total_covered_region = self.covered_regions()[0]

        # get the number of steps necessary for the whole time series
        processing_steps = signal.shape[0] - total_covered_region

        # check whether the signal is long enough
        if total_covered_region > signal.shape[0]:
            raise ValueError(f'Test signal for runtime estimation is not long enough: {signal.shape=} < {total_covered_region}')

        # shorten the signal so we only run one step
        if signal.ndim == 2:
            shortened_signal = signal[:total_covered_region+1, :].copy()
        elif signal.ndim == 1:
            shortened_signal = signal[:total_covered_region + 1].copy()
        else:
            raise ValueError(f'Test signal for runtime estimation has weird shape {signal.shape=}.')

        # trigger the potential JIT compilers
        self.transform(shortened_signal)

        # compute the scoring while measuring the time
        times = np.zeros(steps)
        for idx in range(steps):

            # make the transformation and measure the time
            start = time.perf_counter()
            self.transform(shortened_signal)
            timer = time.perf_counter() - start

            # fill the time
            times[idx] = timer


        # compute the time per step
        timer = np.mean(times)
        std = np.std(times)

        # multiply with the amounts of samples that we will be processing
        timer = timer * processing_steps
        std = std * processing_steps

        # check whether we want to print the result
        if verbose:
            print(f"For {signal.shape=} and the current parameters, the runtime will be around {timer:.3f} seconds (+/- {std:.3f} seconds).")
        return timer, std

    @property
    def first_score_position(self):
        return self.covered_regions()[0] - self.compute_offset() - self.scoring_step//2

    @abstractmethod
    def compute_offset(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def transform(self, time_series: np.ndarray):
        raise NotImplementedError
