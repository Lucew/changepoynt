# this file contains utility functions for handling and computing matrices needed for some of the algorithms in this
# package
import numpy as np
from numba import jit
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import svds, eigsh
import fbpca
import warnings


@jit(nopython=True)
def power_method(a_matrix: np.ndarray, x_vector: np.ndarray, n_iterations: int) -> (float, np.ndarray):
    """
    This function searches the largest (dominant) eigenvalue and corresponding eigenvector by repeated multiplication
    of the matrix A with an initial vector. It assumes a dominant eigenvalue bigger than the second one, otherwise
    it won't converge.

    For proof and explanation look at:
    https://pythonnumericalmethods.berkeley.edu/notebooks/chapter15.02-The-Power-Method.html
    :param a_matrix: 2D-Matrix of size NxN filled with floats
    :param x_vector: Vector of size Nx1 filled with floats
    :param n_iterations: the amount of iterations for the approximation
    :return: the dominant eigenvalue and corresponding eigenvector
    """

    # go through the iterations and continue to scale the returned vector, so we do not reach extreme values
    # during the iteration we scale the vector by its maximum as we can than easily extract the eigenvalue
    a_square = a_matrix.T @ a_matrix
    for _ in range(n_iterations):

        # multiplication with a_matrix.T @ a_matrix as can be seen in explanation of
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        x_vector = a_square @ x_vector

        # scale the vector so we keep the values in bound
        x_vector = x_vector / np.max(x_vector)

    # get the normed eigenvector
    x_vector = x_vector / np.linalg.norm(x_vector)

    # get the corresponding eigenvalue
    eigenvalue = np.linalg.norm(a_matrix @ x_vector)
    return eigenvalue, (a_matrix @ x_vector)/eigenvalue


# @jit(nopython=True)
def lanczos(a_matrix: np.ndarray, r_0: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    This function computes the tri-diagonalization matrix from the square matrix C which is the result of the lanczos
    algorithm.

    The algorithm has been described and proven in:
    Idé, Tsuyoshi, and Koji Tsuda.
    "Change-point detection using krylov subspace learning."
    Proceedings of the 2007 SIAM International Conference on Data Mining.
    Society for Industrial and Applied Mathematics, 2007.

    :param a_matrix: 2D-Matrix of size NxN filled with floats where we want to find the krylov subspace approx. for
    :param r_0: intial starting vector for the subspace approximation
    :param k: size of the approximation
    :return: Returns the alpha and beta values for the tridiagonal, symmetric matrix T. alphas are the values from the
    main diagonal and beta from the off diagonal as described in
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh_tridiagonal.html (alpha = d, beta = e)
    """

    # save the initial vector
    r_i = r_0
    q_i = np.zeros_like(r_i)

    # initialization of the diagonal elements
    alphas = np.zeros(shape=(k + 1,), dtype=np.float64)
    betas = np.ones(shape=(k + 1,), dtype=np.float64)

    # Subroutine 1 of the paper
    for j in range(k):
        # compute r_(j+1)
        new_q = r_i / betas[j]

        # compute the new alpha
        alphas[j + 1] = new_q.T @ a_matrix @ new_q

        # compute the new r
        r_i = a_matrix @ new_q - alphas[j + 1] * new_q - betas[j] * q_i

        # compute the next beta
        betas[j + 1] = np.linalg.norm(r_i)

        # update the previous q
        q_i = new_q

    return alphas[1:], betas[1:-1]


def tridiagonal_eigenvalues(alphas: np.ndarray, betas: np.ndarray, amount=-1):
    """
    This function uses a fast approach for symmetric tridiagonal matrices to calculate the [amount] highest eigenvalues
    and corresponding eigenvectors.

    :param alphas: main diagonal elements
    :param betas: off diagonal elements
    :param amount: The amount of eigenvalues you want to compute (from the highest)
    :return: eigenvalues and corresponding eigenvectors
    """

    # check whether we need to use default parameters
    if amount < 0:
        amount = alphas.shape[0]

    # assertions about shape and dimensions as well as amount of eigenvectors
    assert 0 < amount <= alphas.shape[0], 'We can only calculate one to size of matrix eigenvalues.'
    assert alphas.ndim == 1, 'The alphas need to be vectors.'
    assert betas.ndim == 1, 'The betas need to be vectors.'
    assert alphas.shape[0] - 1 == betas.shape[0], 'Alpha size needs to be exactly one bigger than beta size.'

    # compute the decomposition
    eigenvalues, eigenvectors = eigh_tridiagonal(d=alphas, e=betas,
                                                 select='i', select_range=(alphas.shape[0]-amount, alphas.shape[0]-1))

    # return them to be in sinking order
    return eigenvalues[::-1], eigenvectors[:, ::-1]


def rayleigh_ritz_singular_value_decomposition(a_matrix: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    This function uses the Rayleigh-Ritz method implemented in ARPACK to compute the k highest eigenvalues and
    corresponding eigenvectors. It should be faster than a complete svd.

    !NOTE!:
    The order of the k highest eigenvalues is not guaranteed by this method!

    :param a_matrix: 2D-Matrix filled with floats for which we want to find the left eigenvectors
    :param k: the amount of highest eigenvectors we want to find
    :return: returns the eigenvalues and eigenvectors as numpy arrays
    """
    eigenvectors, eigenvalues, _ = svds(a_matrix, k=k)
    return eigenvalues, eigenvectors


def facebook_randomized_svd(a_matrix: np.ndarray, randomized_rank: int) -> (np.ndarray, np.ndarray):
    """
    This function implements randomized singular vector decomposition of a matrix as surveyed and described in

    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp.
    "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions."
    SIAM review 53.2 (2011): 217-288.

    on page 4 chapter 1.3 and further.

    :param a_matrix: 2D-Matrix filled with floats for which we want to find the left eigenvectors
    :param randomized_rank: the rank of the noise matrix used for randomized svd. the higher the rank, the better the
    approximation but the lower the precision of the eigenvectors
    :return:
    """
    eigenvectors, eigenvalues, _ = fbpca.pca(a_matrix, randomized_rank, True)
    return eigenvalues, eigenvectors


def implicit_restarted_lanczos_bidiagonalization(a_matrix: np.ndarray, rank: int,
                                                 lanczos_rank: int) -> (np.ndarray, np.ndarray):
    """
    This function uses the implicitly Restarted Lanczos method implemented in ARPACK to compute the k highest
    eigenvalues and corresponding eigenvectors. It should be faster as a complete svd.

    !NOTE!
    This method only works for symmetric (hermitian) matrices!!

    :param a_matrix: 2D-Matrix filled with floats for which we want to find the left eigenvectors
    :param rank: the amount of highest eigenvectors we want to find
    :param lanczos_rank: the size of the lanczos subspace approximation
    :return: returns the eigenvalues and eigenvectors as numpy arrays
    """
    eigenvalues, eigenvectors = eigsh(a_matrix, k=rank, which='LM', ncv=lanczos_rank)
    return eigenvalues, eigenvectors


def examples():
    """
    This function implements some usage examples for quick internal testing. It is not aimed for beeing used.
    :return: None
    """

    import time

    # set a random seed
    np.random.seed(1234)

    # create the random starting vector
    x = np.random.rand(100)
    x /= np.linalg.norm(x)
    A = np.random.rand(100, 100)

    # test the power method
    eigval, eigvec = power_method(A, x, n_iterations=100)
    eigvecs, eigvals, _ = np.linalg.svd(A)
    print(eigval, eigvals[0])

    # test the tridiagonalization method
    size = 1000
    d = 3 * np.random.rand(size)
    e = -1 * np.random.rand(size - 1)
    start = time.time()
    tri_eigvals, tri_eigvecs = tridiagonal_eigenvalues(d, e, size // 2)
    print(f'Specialized tridiagonal SVD took: {time.time() - start} s.')
    T = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    start = time.time()
    eigvecs, eigvals , _ = np.linalg.svd(T)
    print(f'Normal SVD took: {time.time() - start} s.')

    # test randomized svd
    eigval, eigvec = randomized_singular_value_decomposition(A, 20)
    eigvecs, eigvals, _ = np.linalg.svd(A)
    print(eigval, eigvals[0])

    # test irlbd
    eigval, eigvec, = implicit_restarted_lanczos_bidiagonalization(A, 20)
    eigvecs, eigvals, _ = np.linalg.svd(A)
    print(eigval, eigvals[0])


@jit(nopython=True)
def compile_hankel(time_series: np.ndarray, end_index: int, window_size: int, rank: int, lag: int = 1) -> np.ndarray:
    """
    This function constructs a hankel matrix from a 1D time series. Please make sure constructing the matrix with
    the given parameters (end index, window size, etc.) is possible, as this function does no checks due to
    performance reasons.

    :param time_series: 1D array with float values as the time series
    :param end_index: the index (point in time) where the time series starts
    :param window_size: the size of the windows cut from the time series
    :param rank: the amount of time series in the matrix
    :param lag: the lag between the time series of the different columns
    :return: The hankel matrix with lag one
    """

    # make an empty matrix to place the values
    #
    # almost no faster way:
    # https://stackoverflow.com/questions/71410927/vectorized-way-to-construct-a-block-hankel-matrix-in-numpy-or-scipy
    hankel = np.empty((window_size, rank))

    # go through the time series and make the hankel matrix
    for cx in range(rank):
        hankel[:, -cx-1] = time_series[(end_index-window_size-cx*lag):(end_index-cx*lag)]
    return hankel


if __name__ == '__main__':
    examples()
