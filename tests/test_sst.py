import pytest
import numpy as np
import changepoynt.algorithms.sst as ssts
import logging


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
        sst = ssts.SST(30)

        # get the different method names
        methods = list(sst.methods.keys())

        # go through the methods and check execution
        for method in methods:
            ssts.SST(50, rank=2, method=method).transform(self.signal)

    def test_svd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="svd")
        sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="svd", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_rectangle_matrix(self):

        # initialize the scorer
        sst = ssts.SST(50, 20, method="ika")
        sst.transform(self.signal)

    def test_fbrsvd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="fbrsvd")
        sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="fbrsvd", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_rsvd_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="rsvd")
        res = sst.transform(self.signal)
        sst = ssts.SST(50, method="rsvd", use_fast_hankel=True)
        res2 = sst.transform(self.signal)

    def test_ika_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="ika")
        res = sst.transform(self.signal)
        sst = ssts.SST(50, method="ika", use_fast_hankel=True)
        res2 = sst.transform(self.signal)

    def test_naive_method(self):

        # initialize the scorer
        sst = ssts.SST(50, method="naive")
        res = sst.transform(self.signal)
        with pytest.raises(ValueError):
            sst = ssts.SST(50, method="naive", use_fast_hankel=True)
            res2 = sst.transform(self.signal)

    def test_default(self):
        ssts.SST(min(5, self.signal_length//2), rank=2).transform(self.signal)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            ssts.SST(10, method='asdafwegrhqh')


if __name__ == "__main__":
    pytest.main()
