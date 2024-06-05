# this file contains utility functions for handling and computing matrices needed for some of the algorithms in this
# package
import numpy as np
import numba as nb

import scipy as sp
import scipy.linalg as splg
import scipy.sparse.linalg
import fbpca


# @nb.jit(nopython=True)
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
    # TODO: This only works for our symmetric correlation matrices but not for any arbitrary matrices
    for _ in range(n_iterations):
        # multiplication with a_matrix.T @ a_matrix as can be seen in explanation of
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        x_vector = a_matrix @ x_vector

        # scale the vector so we keep the values in bound
        x_vector = x_vector / np.max(x_vector)

    # get the normed eigenvector
    x_vector = x_vector / np.linalg.norm(x_vector)

    # get the corresponding eigenvalue
    eigenvalue = np.linalg.norm(a_matrix @ x_vector)
    return eigenvalue, (a_matrix @ x_vector) / eigenvalue


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
        alphas[j + 1] = new_q.T @ (a_matrix @ new_q)

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
    :param amount: The number of eigenvalues you want to compute (from the highest)
    :return: eigenvalues and corresponding eigenvectors
    """

    # check whether we need to use default parameters
    if amount < 0:
        amount = alphas.shape[0]

    # assertions about shape and dimensions as well as number of eigenvectors
    assert 0 < amount <= alphas.shape[0], 'We can only calculate one to size of matrix eigenvalues.'
    assert alphas.ndim == 1, 'The alphas need to be vectors.'
    assert betas.ndim == 1, 'The betas need to be vectors.'
    assert alphas.shape[0] - 1 == betas.shape[0], 'Alpha size needs to be exactly one bigger than beta size.'

    # compute the decomposition
    eigenvalues, eigenvectors = splg.eigh_tridiagonal(d=alphas, e=betas, select='i',
                                                      select_range=(alphas.shape[0] - amount, alphas.shape[0] - 1))

    # return them to be in sinking order
    return eigenvalues[::-1], eigenvectors[:, ::-1]


def rayleigh_ritz_singular_value_decomposition(a_matrix: np.ndarray, k: int) -> (np.ndarray, np.ndarray):
    """
    This function uses the Rayleigh-Ritz method implemented in ARPACK to compute the k highest eigenvalues and
    corresponding eigenvectors. It should be faster than a complete svd.

    !NOTE!:
    The order of the k highest eigenvalues is not guaranteed by this method!

    :param a_matrix: 2D-Matrix filled with floats for which we want to find the left eigenvectors
    :param k: the number of highest eigenvectors we want to find
    :return: returns the eigenvalues and eigenvectors as numpy arrays
    """
    singular_vectors, singular_values, _ = scipy.sparse.linalg.svds(a_matrix, k=k)
    return singular_values, singular_vectors


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
    singular_vectors, singular_values, _ = fbpca.pca(a_matrix, randomized_rank, True)
    return singular_values, singular_vectors


def randomized_hankel_svd(hankel_matrix: np.ndarray, k: int, subspace_iteration_q: int = 2, oversampling_p: int = 2):
    """
    Function for the randomized singular vector decomposition using [1].
    Implementation modified from: https://pypi.org/project/fbpca/
    """

    # get the parameter l from the paper
    sample_length_l = k+oversampling_p
    assert 1.25*sample_length_l < min(hankel_matrix.shape)

    # Apply A to a random matrix, obtaining Q.
    random_matrix_omega = np.random.uniform(low=-1, high=1, size=(hankel_matrix.shape[0], sample_length_l))
    projection_matrix_q = hankel_matrix@random_matrix_omega

    # Form a matrix Q whose columns constitute a well-conditioned basis for the columns of the earlier Q.
    if subspace_iteration_q == 0:
        (projection_matrix_q, _) = sp.linalg.qr(projection_matrix_q, mode='economic')
    if subspace_iteration_q > 0:
        (projection_matrix_q, _) = sp.linalg.lu(projection_matrix_q, permute_l=True)

    # Conduct normalized power iterations.
    for it in range(subspace_iteration_q):

        # QA
        projection_matrix_q = (projection_matrix_q.T @ hankel_matrix).T

        (projection_matrix_q, _) = splg.lu(projection_matrix_q, permute_l=True)

        # AAQ
        projection_matrix_q = hankel_matrix @ projection_matrix_q

        if it + 1 < subspace_iteration_q:
            (projection_matrix_q, _) = splg.lu(projection_matrix_q, permute_l=True)
        else:
            (projection_matrix_q, _) = splg.qr(projection_matrix_q, mode='economic')

    # SVD Q'*A to obtain approximations to the singular values and right singular vectors of A; adjust the left singular
    # vectors of Q'*A to approximate the left singular vectors of A.
    lower_space_hankel = projection_matrix_q.T @ hankel_matrix
    (R, s, Va) = splg.svd(lower_space_hankel, full_matrices=False)
    U = projection_matrix_q.dot(R)

    # Retain only the leftmost k columns of U, the uppermost k rows of Va, and the first k entries of s.
    return U[:, :k], s[:k], Va[:k, :]


@nb.jit(nopython=True)
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
        hankel[:, -cx - 1] = time_series[(end_index - window_size - cx * lag):(end_index - cx * lag)]
    return hankel


@nb.njit(parallel=True)
def fast_numba_hankel_matmul(hankel_fft: np.ndarray, l_windows: int, fft_shape: int, other_matrix: np.ndarray,
                             lag: int):
    # get the shape of the other matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros) to make up for the lag
    if lag > 1:
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
    else:
        out = other_matrix

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((l_windows, n))

    # flip the other matrix
    out = np.flipud(out)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in nb.prange(n):
        # compute the fft of the vector
        fft_x = sp.fft.rfft(out[:, index], n=fft_shape)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = sp.fft.irfft(hankel_fft * fft_x, n=fft_shape)[(m - 1) * lag:(m - 1) * lag + l_windows]
    return result_buffer


@nb.njit(parallel=True)
def fast_numba_hankel_left_matmul(hankel_fft: np.ndarray, n_windows: int, fft_shape: int, other_matrix: np.ndarray,
                                  lag: int):
    # transpose the other matrix
    other_matrix = other_matrix.T

    # get the shape of the other matrix
    m, n = other_matrix.shape

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((n_windows, n))

    # flip the other matrix
    other_matrix = np.flipud(other_matrix)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in nb.prange(n):
        # compute the fft of the vector
        fft_x = sp.fft.rfft(other_matrix[:, index], n=fft_shape)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = sp.fft.irfft(hankel_fft * fft_x, n=fft_shape)[(m - 1):(m - 1) + n_windows * lag:lag]
    return result_buffer.T


@nb.njit(parallel=True)
def fast_numba_hankel_correlation_matmul(hankel_fft: np.ndarray, fft_shape: int, window_number: int,
                                         other_matrix: np.ndarray, lag: int):

    # get the shape of the other matrix
    m, n = other_matrix.shape
    wn = window_number

    # make a buffer if we need it for lagged representation
    if lag > 1:
        buffer = np.zeros((lag * wn - lag + 1, n), dtype=other_matrix.dtype)
    else:
        # we can just use the other matrix, as we won't ever touch the buffer for lag < 1
        # so we don't modify it
        buffer = other_matrix

    # create the new empty matrix that we will fill with values
    result_buffer = np.empty((m, n))

    # flip the other matrix
    out = np.flipud(other_matrix)

    # make a numba parallel loop over the vector of the other matrix (columns)
    for index in nb.prange(n):

        # compute the fft of the vector
        fft_x = sp.fft.rfft(out[:, index], n=fft_shape)

        # make the convolution and transform it back
        first = sp.fft.irfft(hankel_fft * fft_x, n=fft_shape)[(m - 1):(m - 1) + lag * wn:lag]
        # assert np.allclose(first, result[:, index]), f'{first}\n{result[:, index]}'

        # check whether we have lag
        if lag > 1:
            buffer[::lag, index] = first
            first = buffer[:, index]
        first = np.flip(first)

        # compute the fft of the vector
        fft_x = sp.fft.rfft(first, n=fft_shape)

        # multiply the ffts with each other to do the convolution in frequency domain and convert it back
        # and save it into the output buffer
        result_buffer[:, index] = sp.fft.irfft(hankel_fft*fft_x, n=fft_shape)[(wn-1)*lag:(wn-1)*lag+m]
    return result_buffer


def get_fast_hankel_representation(time_series, end_index, length_windows, number_windows,
                                   lag=1) -> (np.ndarray, int, np.ndarray):
    # get the last column of the hankel matrix. The reason for that is that we will use an algorithm for Toeplitz
    # matrices to speed up the multiplication and Hankel[:, ::-1] == Toeplitz.
    #
    # The algorithm requires the first Toeplitz column and therefore the last Hankel Column
    #
    # It also requires the inverse of the row columns ignoring the last element. For reference see:
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # additionally we only need the combined vector from both for our multiplication. In order to employ the
    # fastest DFT possible, the vector needs have a length with a clean power of two, which we create using
    # zero padding
    #
    # We can also use the signal itself, as multiplying a vector with the hankel matrix is effectively a convolution
    # over the signal. Therefore, we can use the fft convolution with way lower complexity or any other built in
    # convolution library!

    # get column and row, or we can also do it by using the signal itself!
    # last_column = time_series[end_index-length_windows:end_index]
    # row_without_last_element = time_series[end_index-lag*(number_windows-1)-length_windows:end_index-length_windows]
    signal = time_series[end_index - lag * (number_windows - 1) - length_windows:end_index]

    # get the length of the matrices
    # col_length = last_column.shape[0]
    # row_length = row_without_last_element.shape[0]
    # combined_length = col_length + row_length

    # compute the padded vector length for an optimal fft length. Here we can use the built-in scipy function that
    # computes the perfect fft length optimized for their implementation, so it is even faster than with powers
    # of two!
    # fft_len = 1 << int(np.ceil(np.log2(combined_length)))
    # fft_len = sp.fft.next_fast_len(combined_length, True)
    fft_len = sp.fft.next_fast_len(signal.shape[0] + number_windows, True)

    # compute the fft over the padded hankel matrix
    # if we would pad in the middle like this: we would introduce no linear phase to the fft
    #
    # padded_toeplitz = np.concatenate((last_column, np.zeros((fft_len - combined_length)), row_without_last_element))
    #
    # but we want to use the built-in padding functionality of the fft in scipy, so we pad at the end like can be seen
    # here.
    # This introduces a linear phase and a shift to the fft, which we need to account for in the reverse
    # functions.
    #
    # More details see:
    # https://dsp.stackexchange.com/questions/82273/why-to-pad-zeros-at-the-middle-of-sequence-instead-at-the-end-of-the-sequence
    # https://dsp.stackexchange.com/questions/83461/phase-of-an-fft-after-zeropadding
    #
    # Workers are not necessary as we expect a 1D time series
    hankel_rfft = sp.fft.rfft(signal, n=fft_len, axis=0).reshape(-1, 1)
    return hankel_rfft[:, 0], fft_len, signal


class HankelFFTRepresentation:

    def __init__(self, time_series: np.ndarray, end_index: int, window_length: int, window_number: int, lag: int = 1,
                 _copy_representation: 'HankelFFTRepresentation' = None):

        if _copy_representation is None:

            # save the parameters that are necessary for later computations
            self.window_length = window_length
            self.window_number = window_number
            self.lag = lag
            self.shape = (window_length, window_number)

            # create the representation and save it into the class
            hankel_rfft, fft_len, _ = get_fast_hankel_representation(time_series, end_index,
                                                                     window_length, window_number, lag=lag)
            self.hankel_rfft = hankel_rfft
            self.fft_length = fft_len
        else:

            # check that nothing else is set
            assert time_series is None and end_index is None and window_length is None and lag is None \
                , 'Using the constructor for copying requires all other options to be set to None.'

            # copy the parameters from the other model
            self.window_length = _copy_representation.window_length
            self.window_number = _copy_representation.window_number
            self.lag = 1
            self.hankel_rfft = _copy_representation.hankel_rfft
            self.fft_length = _copy_representation.fft_length

    def multiply_other_from_right(self, other_matrix: np.ndarray) -> np.ndarray:
        # check the dimensions
        if other_matrix.shape[0] != self.window_number:
            raise ValueError(f'matmul: Right matrix has a mismatch in its core dimension 0 '
                             f'(size {other_matrix.shape[0]} is different from {self.window_number})')

        # make the product
        return fast_numba_hankel_matmul(self.hankel_rfft, self.window_length, self.fft_length, other_matrix,
                                        self.lag)

    def multiply_other_from_left(self, other_matrix: np.ndarray) -> np.ndarray:

        # check the dimensions
        if other_matrix.shape[1] != self.window_length:
            raise ValueError(f'matmul: Left matrix has a mismatch in its core dimension 0 '
                             f'(size {other_matrix.shape[1]} is different from {self.window_length})')

        # make the product
        return fast_numba_hankel_left_matmul(self.hankel_rfft, self.window_number, self.fft_length, other_matrix,
                                             self.lag)

    @property
    def T(self) -> 'HankelFFTRepresentation':

        # create a shallow copy
        new_matrix = self.shallow_copy()

        # check that we have no lag in column direction, as we have not implemented lag in row direction
        # for the multiplications yet
        if new_matrix.lag != 1:
            raise ValueError('As of now we can not calculate with transposed lagged Hankel matrices.')

        # switch the window sizes
        new_matrix.window_number, new_matrix.window_length = new_matrix.window_length, new_matrix.window_number
        return new_matrix

    def __matmul__(self, other):

        # right side matmul
        if isinstance(other, np.ndarray):
            return self.multiply_other_from_right(other)

        # right side matmul with transposed
        elif isinstance(other, self.__class__):
            # check that the fft representation is the same and only transposed
            if self.equal_hankel_transposed(other):
                return HankelCorrelationFFTRepresentation(self)
            else:
                raise ValueError('At this point matmul is only supported for itself with its transposed from left.')

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != '__call__' or len(kwargs) > 0 or len(args) != 2:
            raise NotImplemented
        if ufunc == np.matmul:
            # left side matmul
            if isinstance(args[0], np.ndarray) and args[1] is self:
                return self.multiply_other_from_left(args[0])
            else:
                raise NotImplemented
        else:
            raise NotImplemented

    def shallow_copy(self) -> 'HankelFFTRepresentation':

        # type hints don't like this, but I don't want to add None to the hint, as normal users should not use
        # directly use the constructor for copying.
        return HankelFFTRepresentation(time_series=None, end_index=None, window_length=None, window_number=None,
                                       lag=None, _copy_representation=self)

    def equal_hankel(self, other: 'HankelFFTRepresentation') -> bool:
        return (self.hankel_rfft is other.hankel_rfft
                and self.window_number == other.window_number and self.window_length == other.window_length)

    def equal_hankel_transposed(self, other: 'HankelFFTRepresentation') -> bool:
        return (self.hankel_rfft is other.hankel_rfft
                and self.window_number == other.window_length and self.window_length == other.window_number)


class HankelCorrelationFFTRepresentation:

    def __init__(self, hankel_matrix: HankelFFTRepresentation):

        # save the parameters that are necessary for later computations
        self.hankel_rfft = hankel_matrix.hankel_rfft
        self.fft_length = hankel_matrix.fft_length
        self.window_length = hankel_matrix.window_length
        self.window_number = hankel_matrix.window_number
        self.lag = hankel_matrix.lag
        self.hankel_matrix = hankel_matrix
        self.shape = (self.window_length, self.window_length)

    def __matmul__(self, other_matrix: np.ndarray) -> np.ndarray:
        if isinstance(other_matrix, np.ndarray):
            return fast_numba_hankel_correlation_matmul(self.hankel_rfft, self.fft_length, self.window_number, other_matrix, self.lag)
        else:
            raise NotImplemented


def examples():
    """
    This function implements some usage examples for quick internal testing. It is not aimed at being used.
    :return: None
    """

    import time

    # create a random signal
    signal = np.random.rand(100)*1000

    # create the two different hankel representations
    hankel_matrix_fft = HankelFFTRepresentation(signal, 100, 50, 20, lag=1)
    hankel_matrix = compile_hankel(signal, end_index=100, window_size=50, rank=20, lag=1)

    # create test matrices to multiply with
    other_matrix = np.random.rand(20, 65)
    other_matrix2 = np.random.rand(50, 15)

    # test the right product
    res = hankel_matrix_fft @ other_matrix
    res2 = hankel_matrix @ other_matrix
    print('Test right:', np.allclose(res, res2))

    # test the left product
    res = other_matrix2.T @ hankel_matrix_fft
    res2 = other_matrix2.T @ hankel_matrix
    print('Test left:', np.allclose(res, res2))

    # test the transposed right product
    res = hankel_matrix_fft.T @ other_matrix2
    res2 = hankel_matrix.T @ other_matrix2
    print('Test transposed right:', np.allclose(res, res2))

    # test the transposed left product
    res = other_matrix.T @ hankel_matrix_fft.T
    res2 = other_matrix.T @ hankel_matrix.T
    print('Test transposed left:', np.allclose(res, res2))

    # check the correlation matrix multiplication
    hankel_corr = hankel_matrix @ hankel_matrix.T
    hankel_fft_corr = hankel_matrix_fft @ hankel_matrix_fft.T
    res = hankel_fft_corr @ other_matrix2
    res2 = hankel_corr @ other_matrix2
    print('Test correlation multiplication:', np.allclose(res, res2))

    # set a random seed
    np.random.seed(1234)

    # create the random starting vector
    x = np.random.rand(100)
    x /= np.linalg.norm(x)
    A = np.random.rand(100, 100)

    # test the power method
    singval, singvec = power_method(A, x, n_iterations=100)
    singvecs, singvals, _ = np.linalg.svd(A)
    print(singval, singvals[0])

    # test the tridiagonalization method for speed
    size = 1000
    d = 3 * np.random.rand(size)
    e = -1 * np.random.rand(size - 1)
    start = time.time()
    tri_eigvals, tri_eigvecs = tridiagonal_eigenvalues(d, e, size // 2)
    print(f'Specialized tridiagonal SVD took: {time.time() - start} s.')
    T = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    start = time.time()
    singvecs, singvals, _ = np.linalg.svd(T)
    print(f'Normal SVD took: {time.time() - start} s.')

    # test randomized svd
    singval, singvec = facebook_randomized_svd(A, 20)
    singvecs, singvals, _ = np.linalg.svd(A)
    print(singval, singvals[0])


if __name__ == '__main__':
    examples()
