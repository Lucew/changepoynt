import pytest
import numpy as np
import changepoynt.algorithms.bocpd as bocpd


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

    def test_general_method_call(self):

        # create the transformer
        transf = bocpd.BOCPD(200)
        score = transf.transform(self.signal)

    def test_specified_values(self):

        # create the transformer
        transf = bocpd.BOCPD(200, 0, 1, 1, 10)
        score = transf.transform(self.signal)


if __name__ == "__main__":
    pytest.main()
