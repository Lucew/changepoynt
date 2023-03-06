# this file contains the tests for the linalg.py file in the utils of this package
import pytest
import numpy as np
from changepoynt.utils.linalg import power_method, lanczos, tridiagonal_eigenvalues


class TestLinearAlgebra:
    def setup_method(self):
        # set a random seed
        np.random.seed(3455)

        # create a random square matrix
        self.A = np.random.rand(50, 50)

        # create random starting vector for the power method
        # and normalize it
        self.x0 = np.random.rand(50)
        self.x0 /= np.linalg.norm(self.x0)

        # create a tridiagonal matrix and the diagonal elements
        self.d = 5 * np.random.rand(100)
        self.e = -1 * np.random.rand(99)
        self.T = np.diag(self.d) + np.diag(self.e, k=1) + np.diag(self.e, k=-1)

    def teardown_method(self):
        pass

    def test_power_method(self):
        eigval, eigvec = power_method(self.A, self.x0, n_iterations=100)
        eigvecs, eigvals, _ = np.linalg.svd(self.A)
        np.testing.assert_almost_equal(np.abs(eigval), np.abs(eigvals[0]))
        np.testing.assert_almost_equal(np.abs(eigvec), np.abs(eigvecs[:, 0]))

    def test_eig_tridiag(self):
        # take a look at the highest half of the eigenvalues as lower one might be unstable
        amount = self.d.shape[0]//2
        tri_eigvals, tri_eigvecs = tridiagonal_eigenvalues(self.d, self.e, amount)
        eigvecs, eigvals, _ = np.linalg.svd(self.T)
        np.testing.assert_almost_equal(tri_eigvals, eigvals[:amount])
        np.testing.assert_almost_equal(np.abs(tri_eigvecs), np.abs(eigvecs[:, :amount]))


if __name__ == "__main__":
    pytest.main()