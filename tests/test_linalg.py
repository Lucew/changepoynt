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
        self.singvecs, self.singvals, _ = np.linalg.svd(self.A)

        # create a random signal
        self.signal = np.random.rand(100) * 1000

        # create test matrices to multiply with
        self.other_matrix = np.random.rand(20, 65)
        self.other_matrix2 = np.random.rand(50, 15)

        # create the two different hankel representations
        self.hankel_matrix_fft = lg.HankelFFTRepresentation(self.signal, end_index=100, window_length=50,
                                                            window_number=20, lag=1)
        self.hankel_matrix = lg.compile_hankel(self.signal, end_index=100, window_size=50, rank=20, lag=1)

    def teardown_method(self):
        pass

    def test_power_method(self):
        # make the correlation matrix
        a_corr = self.A @ self.A.T
        eigval, eigvec = lg.power_method(a_corr, self.x0, n_iterations=100)

        # get the eigenvector
        eigvals, eigvecs = np.linalg.eig(a_corr)
        idces = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idces]
        eigvecs = eigvecs[:, idces]

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

    def test_rayleigh_ritz_svd(self):
        k = 20
        k_eigvals, k_eigvecs = lg.rayleigh_ritz_singular_value_decomposition(self.A, k)

        # order is not guaranteed by the method, so we need to sort in decreasing order
        idx = np.argsort(k_eigvals)[::-1]
        k_eigvals = k_eigvals[idx]
        k_eigvecs = k_eigvecs[:, idx]

        # compare the values
        np.testing.assert_almost_equal(k_eigvals, self.singvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs), np.abs(self.singvecs[:, :k]))

    def test_randomized_svd(self):

        # set a threshold for the amount of eigenvectors we want to have and compute them
        k = 3
        k_eigvals, k_eigvecs = lg.facebook_randomized_svd(self.A, 50)

        # compare the values
        np.testing.assert_almost_equal(k_eigvals[:k], self.singvals[:k])
        np.testing.assert_almost_equal(np.abs(k_eigvecs[:, :k]), np.abs(self.singvecs[:, :k]))

    def test_right_hankel_product(self):
        # make the multiplications
        res = self.hankel_matrix_fft @ self.other_matrix
        res2 = self.hankel_matrix @ self.other_matrix
        np.testing.assert_almost_equal(res, res2)

    def test_left_hankel_product(self):
        # make the multiplications
        res = self.other_matrix2.T @ self.hankel_matrix_fft
        res2 = self.other_matrix2.T @ self.hankel_matrix
        np.testing.assert_almost_equal(res, res2)

    def test_transposed_right_hankel_product(self):
        # make the multiplications
        res = self.hankel_matrix_fft.T @ self.other_matrix2
        res2 = self.hankel_matrix.T @ self.other_matrix2
        np.testing.assert_almost_equal(res, res2)

    def test_transposed_left_hankel_product(self):
        # make the multiplications
        res = self.other_matrix.T @ self.hankel_matrix_fft.T
        res2 = self.other_matrix.T @ self.hankel_matrix.T
        np.testing.assert_almost_equal(res, res2)

    def test_hankel_correlation_product(self):
        # check the correlation matrix multiplication
        hankel_corr = self.hankel_matrix @ self.hankel_matrix.T
        hankel_fft_corr = self.hankel_matrix_fft @ self.hankel_matrix_fft.T
        res = hankel_fft_corr @ self.other_matrix2
        res2 = hankel_corr @ self.other_matrix2
        np.testing.assert_almost_equal(res, res2)


if __name__ == "__main__":
    pytest.main()
