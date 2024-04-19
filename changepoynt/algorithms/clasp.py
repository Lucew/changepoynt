import numpy as np
from changepoynt.algorithms.base_algorithm import Algorithm
# from claspy.segmentation import BinaryClaSPSegmentation


class CLASP(Algorithm):
    """
    This class uses the ClaSP algorithm published in

    [1]
    Schäfer, Patrick, Arik Ermshaus, and Ulf Leser.
    "Clasp-time series segmentation."
    Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.

    This class essentially wraps the claspy library
    https://github.com/ermshaua/claspy
    https://sites.google.com/view/ts-clasp/

    In the current configuration it is recommended to set no parameters, as CLASP is able to extract it from the data.
    TODO:
    Own implementation?
    """

    def __init__(self, n_segments="learn", n_estimators=10, window_size="suss", k_neighbours=3,
                 distance="znormed_euclidean_distance", score="roc_auc",
                 early_stopping=True, validation="significance_test", threshold=1e-15, excl_radius=5,
                 random_state=2357) -> None:

        # say that we currently not support CLASP due to old packages in the requirements
        raise NotImplementedError('CLASP is currently not available, as it requires outdated package versions.')

        # save the specified parameters into instance variables
        self.n_segments = n_segments
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.score = score
        self.early_stopping = early_stopping
        self.validation = validation
        self.threshold = threshold
        self.excl_radius = excl_radius
        self.random_state = random_state

    def transform(self, time_series: np.ndarray) -> np.ndarray:
        # check the dimensions of the input array
        assert time_series.ndim == 1, "Time series needs to be an 1D array."

        # make the clasp segmentation object
        claspy_segmenter = BinaryClaSPSegmentation(n_segments=self.n_segments, n_estimators=self.n_estimators,
                                                   window_size=self.window_size, k_neighbours=self.k_neighbours,
                                                   score=self.score, early_stopping=self.early_stopping,
                                                   validation=self.validation, threshold=self.threshold,
                                                   excl_radius=self.excl_radius, random_state=self.random_state)
        claspy_segmenter.fit(time_series)
        return claspy_segmenter.profile


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
    clasp_recognizer = CLASP()

    # compute the score
    start = time()
    score = clasp_recognizer.transform(x)
    print(f'Computation for {len(x)} signal values took {time()-start} s.')


if __name__ == '__main__':
    _main()
