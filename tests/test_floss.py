import pytest
import numpy as np
import changepoynt.algorithms.floss as floss
import logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)


class TestSST:
    def setup_method(self):
        # set a random seed
        np.random.seed(3455)

        # create a random steps signal of a certain length
        self.signal_length = 200
        x0 = 1 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1
        x1 = 3 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 2
        x2 = 5 * np.ones(self.signal_length) + np.random.rand(self.signal_length) * 1.5
        x = np.hstack([x0, x1, x2])
        x += np.random.rand(x.size)
        self.signal = x

    def teardown_method(self):
        pass

    def test_default(self):
        with pytest.raises(NotImplementedError):
            floss.FLOSS(20).transform(self.signal)


if __name__ == "__main__":
    pytest.main()
