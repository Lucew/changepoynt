import numpy as np
from changepoynt.algorithms.base_algorithm import Algorithm
import stumpy


class FLOSS(Algorithm):
    """
    This class uses the FLOSS algorithm described in:

    [1]
    Gharghabi, Shaghayegh, et al.
    "Matrix profile VIII: domain agnostic online semantic segmentation at superhuman performance levels."
    2017 IEEE international conference on data mining (ICDM). IEEE, 2017.

    This class essentially wraps the stumpy library which implements a fast approach to calculate the matrix profile.
    https://stumpy.readthedocs.io/en/latest/index.html

    TODO:
    Implement GPU support and streaming? (FLOSS)
    """

    def __init__(self, window_length: int, initial_length: int = None) -> None:
        """
        We initialize the matrix profile and the subsequent floss using only the window length used for comarisons.

        :param window_length: the length of the window used for the distance comparisons in the matrix profile.
        :param initial_length: the length for the initial matrix profile length
        """
        raise NotImplementedError('FLOSS is not yet fully functional.')
        # save the specified parameters into instance variables
        self.window_length = window_length
        self.initial_length = initial_length

        # set default initial length if necessary
        if not initial_length:
            self.initial_length = 10*self.window_length

        # check for the size
        assert self.initial_length >= 2*self.window_length, 'The initial_length should be at least twice' \
                                                            'the window_length.'

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
        assert time_series.shape[0] > self.initial_length, \
            'Time series needs to be longer than specified initial length.'

        # feed it through the online process
        score = _transform(time_series, self.initial_length, self.window_length)
        return score


def save_animation(mp, windows, time_series):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import os

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})

    axs[0].set_xlim((0, mp.shape[0]))
    axs[0].set_ylim((min(time_series), max(time_series)))
    axs[1].set_xlim((0, mp.shape[0]))
    axs[1].set_ylim((-0.1, 1.1))

    lines = []
    for ax in axs:
        line, = ax.plot([], [], lw=2)
        lines.append(line)
    line, = axs[1].plot([], [], lw=2)
    lines.append(line)

    # mark the window lengths
    axs[1].plot([401, 401], (0, mp.shape[0]))

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(window):
        data_out, cac_out = window
        for line, data in zip(lines, [data_out, cac_out]):
            line.set_data(np.arange(data.shape[0]), data)
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=windows, interval=100,
                                   blit=True)
    writergif = animation.PillowWriter(fps=5)
    anim.save('semantic.gif', writer=writergif)


def _transform(time_series: np.ndarray, start_idx: int, window_length: int) -> np.ndarray:
    """
    Compute FLOSS from [1] as the inverse online 1D corrected arc crossing rate.

    :param time_series: 1D array containing the time series to be scored
    :param start_idx: the size for the initial matrix profile
    :param window_length: the size of the windows to be compared
    """

    # create the initial signal
    init_signal = time_series[:start_idx]

    # create the initial matrix profile
    matrix_profile = stumpy.stump(init_signal, m=window_length)

    # initialize the floss object
    stream = stumpy.floss(matrix_profile, init_signal, m=window_length, L=window_length, excl_factor=1)

    # make the score vector
    score = np.zeros_like(time_series)
    score[:start_idx] = 1

    # iterate over all the values in the signal starting at start_idx computing the change point score
    windows = []
    for idx in range(start_idx, time_series.shape[0]):

        # update the floss streaming module
        stream.update(time_series[idx])

        # get the latest score (1-cac)
        score[idx] = stream.cac_1d_[-window_length-2]

        if idx % 20 == 0: windows.append((stream.T_, stream.cac_1d_))
    save_animation(matrix_profile, windows, time_series)
    return 1-score


def _main():
    """
    Internal quick testing function.

    :return:
    """
    from time import time
    import matplotlib.pyplot as plt
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
    fluss_recognizer = FLOSS(50)

    # compute the score
    start = time()
    score = fluss_recognizer.transform(x)
    plt.plot(x)
    plt.plot(score)
    plt.show()
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
