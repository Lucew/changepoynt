import numpy as np
from changepoynt.algorithms.base_algorithm import Algorithm
# import claspy


class CLASP(Algorithm):
    """
    This class uses the ClaSP algorithm published in

    [1]
    Schäfer, Patrick, Arik Ermshaus, and Ulf Leser.
    "Clasp-time series segmentation."
    Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.

    This class essentially wraps the claspy library
    https://github.com/ermshaua/claspy

    TODO:
    Own implementation?
    """

    def __init__(self, window_length: int, initial_length: int = None) -> None:
        raise NotImplementedError('FLOSS is not yet fully functional.')

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        pass


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
    fluss_recognizer = CLASP(50)

    # compute the score
    start = time()
    score = fluss_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
