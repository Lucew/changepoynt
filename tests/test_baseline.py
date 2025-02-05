import pytest
import numpy as np
import changepoynt.algorithms.baseline as baseline


class TestSST:
    def setup_method(self):
        # set a random seed
        np.random.seed(3455)

        # create a random steps signal of a certain length
        self.signal_length = 300
        x0 = 1 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1
        x1 = 3 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 2
        x2 = 5 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1.5
        x = np.hstack([x0, x1, x2])
        x += np.random.rand(x.size)
        self.signal = x

    def teardown_method(self):
        pass

    def test_all_methods(self):

        # go through the methods and check execution
        for method in ['mean', 'var', 'meanvar']:
            baseline.MovingWindow(50, method=method).transform(self.signal)

    def test_moving_mean(self):

        # initialize the scorer
        transf = baseline.MovingWindow(50, method="mean")
        transf.transform(self.signal)

    def test_moving_var(self):

        # initialize the scorer
        transf = baseline.MovingWindow(50, method="var")
        transf.transform(self.signal)

    def test_moving_meanvar(self):

        # initialize the scorer
        transf = baseline.MovingWindow(50, method="meanvar")
        transf.transform(self.signal)

    def test_zero(self):
        # initialize the scorer
        transf = baseline.ZERO()
        transf.transform(self.signal)


if __name__ == "__main__":
    pytest.main()
