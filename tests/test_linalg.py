# this file contains the tests for the linalg.py file in the utils of this package
import pytest
import numpy as np
import changepoynt.utils.linalg as lg


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
        eigval, eigvec = lg.power_method(self.A, self.x0, n_iterations=100)
        eigvecs, eigvals, _ = np.linalg.svd(self.A)
        np.testing.assert_almost_equal(eigval, eigvals[0])
        np.testing.assert_almost_equal(np.abs(eigvec), np.abs(eigvecs[:, 0]))

    def test_eig_tridiag(self):
        # take a look at the highest half of the eigenvalues as lower one might be unstable
        # https://stackoverflow.com/questions/46345217/diagonalization-of-a-tridiagonal-symmetric-sparse-matrix-with-python
        amount = self.d.shape[0]//2
        tri_eigvals, tri_eigvecs = lg.tridiagonal_eigenvalues(self.d, self.e, amount)
        eigvecs, eigvals, _ = np.linalg.svd(self.T)
        np.testing.assert_almost_equal(tri_eigvals, eigvals[:amount])
        np.testing.assert_almost_equal(np.abs(tri_eigvecs), np.abs(eigvecs[:, :amount]))

    def test_k_highest_eigenvectors(self):
        k = 20
        k_eigvals, k_eigvecs = lg.highest_k_eigenvectors(self.A, k)
        eigvecs, eigvals, _ = np.linalg.svd(self.A)

        # order is not guaranteed by the method, so we need to sort in decreasing order
        idx = np.argsort(k_eigvals)[::-1]
        k_eigvals = k_eigvals[idx]
        k_eigvecs = k_eigvecs[:, idx]

        # compare the values
        np.testing.assert_almost_equal(k_eigvals, eigvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs), np.abs(eigvecs[:, :k]))

    def test_randomized_svd(self):
        k = 5
        k_eigvals, k_eigvecs = lg.randomized_singular_value_decomposition(self.A, 50)
        eigvecs, eigvals, _ = np.linalg.svd(self.A)

        # compare the values
        np.testing.assert_almost_equal(k_eigvals[:k], eigvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs[:, :k]), np.abs(eigvecs[:, :k]))


if __name__ == "__main__":
    pytest.main()
