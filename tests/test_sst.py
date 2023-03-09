import pytest
import numpy as np
import changepoynt.algorithms.sst as ssts
import logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


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
        # initialize random default method
        sst = ssts.SingularSpectrumTransformation(30)

        # get the different method names
        methods = list(sst.methods.keys())

        # go through the methods and check execution
        for method in methods:
            LOGGER.info(f'Starting SST for method {method}')
            ssts.SingularSpectrumTransformation(min(5, self.signal_length//2), rank=2,
                                                method=method).transform(self.signal)

    def test_unknown_method(self):
        with pytest.raises(AssertionError):
            ssts.SingularSpectrumTransformation(10, method='asdafwegrhqh')


if __name__ == "__main__":
    pytest.main()
