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

        # compute the svd of the matrix A as we will need it for comparisons
        self.eigvecs, self.eigvals, _ = np.linalg.svd(self.A)

    def teardown_method(self):
        pass

    def test_power_method(self):
        eigval, eigvec = lg.power_method(self.A, self.x0, n_iterations=100)
        np.testing.assert_almost_equal(eigval, self.eigvals[0])
        np.testing.assert_almost_equal(np.abs(eigvec), np.abs(self.eigvecs[:, 0]))

    def test_eig_tridiag(self):
        # take a look at the highest half of the eigenvalues as lower one might be unstable
        # https://stackoverflow.com/questions/46345217/diagonalization-of-a-tridiagonal-symmetric-sparse-matrix-with-python
        amount = self.d.shape[0]//2
        tri_eigvals, tri_eigvecs = lg.tridiagonal_eigenvalues(self.d, self.e, amount)
        eigvecs, eigvals, _ = np.linalg.svd(self.T)
        np.testing.assert_almost_equal(tri_eigvals, eigvals[:amount])
        np.testing.assert_almost_equal(np.abs(tri_eigvecs), np.abs(eigvecs[:, :amount]))

    def test_rayleigh_ritz_svd(self):
        k = 20
        k_eigvals, k_eigvecs = lg.rayleigh_ritz_singular_value_decomposition(self.A, k)

        # order is not guaranteed by the method, so we need to sort in decreasing order
        idx = np.argsort(k_eigvals)[::-1]
        k_eigvals = k_eigvals[idx]
        k_eigvecs = k_eigvecs[:, idx]

        # compare the values
        np.testing.assert_almost_equal(k_eigvals, self.eigvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs), np.abs(self.eigvecs[:, :k]))

    def test_randomized_svd(self):

        # set a threshold for the amount of eigenvectors we want to have and compute them
        k = 3
        k_eigvals, k_eigvecs = lg.facebook_randomized_svd(self.A, 50)

        # compare the values
        np.testing.assert_almost_equal(k_eigvals[:k], self.eigvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs[:, :k]), np.abs(self.eigvecs[:, :k]))

    def test_restarted_implicit_lanczos_svd(self):
        """
        !NOTE!
        This function currently only works for large eigenvalues. So we can mostly only compare the highest two.

        Is that a problem? We will see.
        :return:
        """

        # create a symmetric random matrix (as needed for this method)
        # as proposed in https://stackoverflow.com/questions/10806790/generating-symmetric-matrices-in-numpy
        a = np.random.rand(100, 100)
        a = np.tril(a) + np.tril(a, -1).T

        # test that the matrix is hermitian (symmetric for real valued matrices)
        np.testing.assert_almost_equal(a, a.T)

        # compute the svd of the matrix A as we will need it for comparisons
        eigvecs, eigvals, _ = np.linalg.svd(a)

        # get the results from the lanczos method
        k = 2
        k_eigvals, k_eigvecs = lg.implicit_restarted_lanczos_bidiagonalization(a, k, 2*k)

        # compare the values
        np.testing.assert_almost_equal(k_eigvals[::-1], eigvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs)[:, ::-1], np.abs(eigvecs[:, :k]))


if __name__ == "__main__":
    pytest.main()
