import typing
import abc

import numpy as np
import scipy.fft as spfft


def block_hankel_left_matmat_fft(hankel_fft: np.ndarray, other_matrix: np.ndarray, fft_length: int,
                                 window_length: int, window_number: int):
    """
    Compute A @ H using FFT, where H is a block Hankel matrix.

    H has block structure:

        H[i, j] = B[i + j]


    Definitions
    ----------
    p, q : the shape of the Hankel matrix (p rows, q columns)
    m, n : the shape of the blocks in the Hankel matrix (each block has m rows, n columns)

    Parameters
    ----------

    hankel_fft : ndarray, shape (p + q - 1, m, n)
        The fft transformed (along axis 0) of the Block sequence. B[k] is an m x n matrix block.

    other_matrix : ndarray
        The matrix that we multiply from the right (other_matrix @ hankel_fft).
        Either:
        - dense form: shape (s, p*m)
        - block form: shape (p, s, m)

    fft_length : int
        The length of the fft (with padding). We recommend using scipy.fft.next_fast_length

    window_length : int
        The number of blocks per row of the Hankel matrix

    window_number : int
        The number of blocks per column of the Hankel matrix

    Returns
    -------
    result_matrix : ndarray
        Outcome of other_matrix @ hankel_fft
        If A was dense, returns dense shape (s, q*n).
        If A was block-form, returns block shape (q, s, n).
    """

    # get the dimensions of the hankel matrix
    _, m, n = hankel_fft.shape
    p = window_length
    q = window_number

    return_dense = False

    if other_matrix.ndim == 2:
        # other_matrix is dense: shape (s, p*m)
        s, cols = other_matrix.shape

        if cols != p*m:
            raise ValueError("A has incompatible number of columns.")

        other_matrix_blocks = other_matrix.reshape(s, p, m).transpose(1, 0, 2)
        return_dense = True

    elif other_matrix.ndim == 3:
        # other_matrix is already block-form: shape (p, s, m)
        p_x, s, m_x = other_matrix.shape

        if m_x != m:
            raise ValueError("Block dimensions do not match.")

        other_matrix_blocks = other_matrix

    else:
        raise ValueError("A must have shape (s, p*m) or (p, s, m).")

    if q <= 0:
        raise ValueError("Need len(B) >= p.")

    # For left multiplication:
    #
    # result_matrix_j = sum_i fft_hankel_i other_matrix_{i+j}
    #
    # This is obtained from convolution of reversed A with B.
    other_matrix_rev = other_matrix_blocks[::-1]

    other_matrix_fft = spfft.rfft(other_matrix_rev, n=fft_length, axis=0)  # shape (fft_len, s, m)

    # Frequency-wise matrix-matrix multiplication:
    #
    # result_matrix_fft[ell] = other_matrix_fft[ell] @ fft_hankel[ell]
    #
    # Shapes:
    #   fft_hankel[ell] : (s, m)
    #   other_matrix_fft[ell] : (m, n)
    #   result_matrix_fft[ell] : (s, n)
    result_matrix_fft = other_matrix_fft @ hankel_fft                                # shape (fft_len, s, n)

    conv = spfft.irfft(result_matrix_fft, n=fft_length, axis=0)

    # Extract the desired Hankel result
    result_matrix = conv[p - 1 : p - 1 + q]          # shape (q, s, n)

    result_matrix = np.real_if_close(result_matrix)

    if return_dense:
        return result_matrix.transpose(1, 0, 2).reshape(s, q * n)

    return result_matrix


def block_hankel_right_matmat_direct(hankel: np.ndarray, other_matrix: np.ndarray):
    """
    Compute H @ X without FFT, using a vectorized einsum.

    H has block structure:

        H[i, j] = B[i + j]

    Definitions
    ----------
    p, q : the shape of the Hankel matrix (p rows, q columns)
    m, n : the shape of the blocks in the Hankel matrix (each block has m rows, n columns)

    Parameters
    ----------
    hankel : ndarray, shape (p + q - 1, m, n)
        Block sequence. hankel[k] is an m x n matrix block.

    other_matrix : ndarray
        Either:
        - dense form: shape (q*n, r)
        - block form: shape (q, n, r)

    Returns
    -------
    Y : ndarray
        If X was dense, returns dense shape (p*m, r).
        If X was block-form, returns block shape (p, m, r).
    """

    num_blocks, m, n = hankel.shape
    return_dense = False

    if other_matrix.ndim == 2:
        rows, r = other_matrix.shape

        if rows % n != 0:
            raise ValueError("other_matrix_blocks has incompatible number of rows.")

        q = rows // n
        other_matrix_blocks = other_matrix.reshape(q, n, r)
        return_dense = True

    elif other_matrix.ndim == 3:
        (q, n_other_matrix, r) = other_matrix.shape

        if n_other_matrix != n:
            raise ValueError("Block dimensions do not match.")

        other_matrix_blocks = other_matrix

    else:
        raise ValueError("other_matrix_blocks must have shape (q*n, r) or (q, n, r).")

    p = num_blocks - q + 1

    if p <= 0:
        raise ValueError("Need len(hankel) >= q.")

    # sliding_window_view gives shape (p, m, n, q)
    hankel_windows = np.lib.stride_tricks.sliding_window_view(hankel, window_shape=q, axis=0)

    # Move the window axis so that:
    # b_moved[i, j] = hankel[i + j]
    # b_moved has shape (p, q, m, n)
    hankel_windows_moved = np.moveaxis(hankel_windows, -1, 1)

    # result_matrix[i, a, c] = sum_{j,b} b_moved[i,j,a,b] * other_matrix_blocks_blocks[j,b,c]
    result_matrix = np.einsum("ijmn,jnr->imr", hankel_windows_moved, other_matrix_blocks, optimize=True)

    if return_dense:
        return result_matrix.reshape(p * m, r)

    return result_matrix


def block_hankel_right_matmat_fft(hankel_fft: np.ndarray, other_matrix: np.ndarray, fft_length: int,
                                  window_length: int, window_number: int):
    """
    Compute H @ X using FFT, where H is a block Hankel matrix.

    H has block structure:

        H[i, j] = B[i + j]


    Definitions
    ----------
    p, q : the shape of the Hankel matrix (p rows, q columns)
    m, n : the shape of the blocks in the Hankel matrix (each block has m rows, n columns)

    Parameters
    ----------
    hankel_fft : ndarray, shape (p + q - 1, m, n)
        The fft transformed (along axis 0) of the Block sequence. B[k] is an m x n matrix block.

    other_matrix : ndarray
        Either:
        - dense form: shape (q*n, r)
        - block form: shape (q, n, r)

    fft_length : int
        The length of the fft (with padding). We recommend using scipy.fft.next_fast_length

    window_length : int
        The number of blocks per row of the Hankel matrix

    window_number : int
        The number of blocks per column of the Hankel matrix

    Returns
    -------
    Y : ndarray
        If X was dense, returns dense shape (p*m, r).
        If X was block-form, returns block shape (p, m, r).
    """
    _, m, n = hankel_fft.shape
    p = window_length
    q = window_number

    return_dense = False

    if other_matrix.ndim == 2:
        # X is dense: shape (q*n, r)
        rows, r = other_matrix.shape

        if rows != q*n:
            raise ValueError("X has incompatible number of rows.")

        other_matrix_blocks = other_matrix.reshape(q, n, r)
        return_dense = True

    elif other_matrix.ndim == 3:
        # X is already block-form: shape (q, n, r)
        q_x, n_x, r = other_matrix.shape

        if q_x != q or n_x != n:
            raise ValueError("Block dimensions do not match.")

        other_matrix_blocks = other_matrix

    else:
        raise ValueError("X must have shape (q*n, r) or (q, n, r).")

    if p <= 0:
        raise ValueError("Need len(B) >= q.")

    # Hankel multiplication:
    #
    # Y_i = sum_j B_{i+j} X_j
    #
    # This becomes convolution of B with reversed X.
    other_matrix_rev = other_matrix_blocks[::-1]
    other_matrix_fft = spfft.rfft(other_matrix_rev, n=fft_length, axis=0)  # shape (fft_len, n, r)

    # Frequency-wise matrix-matrix multiplication:
    #
    # FY[ell] = FB[ell] @ FX[ell]
    #
    # Shapes:
    #   FB[ell] : (m, n)
    #   FX[ell] : (n, r)
    #   FY[ell] : (m, r)
    result_matrix_fft = hankel_fft @ other_matrix_fft                                # shape (fft_len, m, r)

    conv = spfft.irfft(result_matrix_fft, n=fft_length, axis=0)

    # Extract the desired Hankel result
    result_matrix_blocks = conv[q - 1 : q - 1 + p]          # shape (p, m, r)

    result_matrix_blocks = np.real_if_close(result_matrix_blocks)

    if return_dense:
        return result_matrix_blocks.reshape(p * m, r)

    return result_matrix_blocks


def block_hankel_left_matmat_direct(hankel: np.ndarray, other_matrix: np.ndarray):
    """
    Compute A @ H without FFT, using a vectorized einsum.

    H has block structure:

        H[i, j] = B[i + j]

    Definitions
    ----------
    p, q : the shape of the Hankel matrix (p rows, q columns)
    m, n : the shape of the blocks in the Hankel matrix (each block has m rows, n columns)

    Parameters
    ----------
    hankel : ndarray, shape (p + q - 1, m, n)
        Block sequence. hankel[k] is an m x n matrix block.

    other_matrix : ndarray
        Matrix to multiply by
        Either:
        - dense form: shape (s, p*m)
        - block form: shape (p, s, m)

    Returns
    -------
    Z : ndarray
        If A was dense, returns dense shape (s, q*n).
        If A was block-form, returns block shape (q, s, n).
    """

    num_blocks, m, n = hankel.shape
    return_dense = False

    if other_matrix.ndim == 2:
        s, cols = other_matrix.shape

        if cols % m != 0:
            raise ValueError("A has incompatible number of columns.")

        p = cols // m
        other_matrix_blocks = other_matrix.reshape(s, p, m).transpose(1, 0, 2)
        return_dense = True

    elif other_matrix.ndim == 3:
        p, s, m_A = other_matrix.shape

        if m_A != m:
            raise ValueError("Block dimensions do not match.")

        other_matrix_blocks = other_matrix

    else:
        raise ValueError("A must have shape (s, p*m) or (p, s, m).")

    q = num_blocks - p + 1

    if q <= 0:
        raise ValueError("Need len(B) >= p.")

    # sliding_window_view gives shape (q, m, n, p)
    hankel_windows = np.lib.stride_tricks.sliding_window_view(hankel, window_shape=p, axis=0)

    # Move the window axis so that:
    # BH[j, i] = B[j + i]
    # BH has shape (q, p, m, n)
    hankel_windows_moved = np.moveaxis(hankel_windows, -1, 1)

    # Z[j, s, n] = sum_{i,m} A_blocks[i,s,m] * BH[j,i,m,n]
    result_matrix_blocks = np.einsum("ism,jimn->jsn", other_matrix_blocks, hankel_windows_moved, optimize=True)

    if return_dense:
        return result_matrix_blocks.transpose(1, 0, 2).reshape(s, q * n)

    return result_matrix_blocks


def get_block_hankel_representation(time_series: np.ndarray, end_index: int, window_length: int, window_number: int) -> [np.ndarray, int]:

    # make some checks for the signal
    if time_series.ndim != 2:
        raise ValueError(f'Input time series has to have two dimensions. Currently: {time_series.ndim=} != 2.')

    # extract the block hankel matrix representation
    # we expect it to have shape (window_number+window_length-1, m, n) where m is the number of time series
    # in the multivariate time series and n is always one in our case
    size = window_number+window_length-1
    return time_series[end_index-(window_number+window_length-1):end_index, :][..., None], size


class BlockHankel:
    """
    This class is the base class for BlockHankel matrices.
    It mainly serves as the interface to capture numpy matrix multiplications (@) using the functions
    __matmul__ (if a BlockHankel matrix is on the left side of the @ operator)
    and __array_ufunc__ (if a BlockHankel is on the right side of the @ operator)

    If these functions are called, the input is given to the abstract methods
    multiply_other_from_left
    and
    multiply_other_from_right
    which we expect the subclasses to implement in detail
    """

    # denote that we will have a shape
    shape: tuple[int, ...]

    @abc.abstractmethod
    def materialize(self) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def multiply_other_from_right(self, other_maxtrix: np.ndarray) -> np.ndarray:
        """Subclasses should implement this method."""
        raise NotImplementedError

    @abc.abstractmethod
    def multiply_other_from_left(self, other_maxtrix: np.ndarray) -> np.ndarray:
        """Subclasses should implement this method."""
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self, deep: bool = False) -> "BlockHankel":
        """Subclasses should implement this method."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def T(self,) -> "BlockHankel":
        """Subclasses should implement this method. The method should transpose the matrix."""
        raise NotImplementedError

    def __matmul__(self, other) -> typing.Union[np.ndarray, "BlockHankel"]:
        """
        This function is called by numpy if we use an instance of the current class on the left side of
        a matrix multiplication operator (@)

        :param other: the other matrix
        :return:
        """
        # right side matmul
        if isinstance(other, np.ndarray):
            return self.multiply_other_from_right(other)
        if isinstance(other, BlockHankel):
            return BlockHankelProductRepresentation(self, other)
        else:
            raise ValueError('At this point matmul is only with other matrices.')

    def __array_ufunc__(self, ufunc, method, *args, **kwargs) -> typing.Union[np.ndarray, "BlockHankel"]:
        """
        This function is called by numpy, if an instance of the current class on the left side of
        an operator where the left side is any numpy object.

        Basically, numpy first checks if the left side of an operator has defined an operation for
        both objects on the right and left. If this fails, it checks the object on the right.

        :param ufunc: the function that is called on the objects
        :param method: what type of invocation it was
        :param args: the arguments of the function
        :param kwargs: the keyword arguments of the function
        :return:
        """
        if method != '__call__' or len(kwargs) > 0 or len(args) != 2:
            return NotImplemented
        if ufunc == np.matmul:
            # left side matmul
            if isinstance(args[0], np.ndarray) and args[1] is self:
                return self.multiply_other_from_left(args[0])
            else:
                return NotImplemented
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs) -> "MultilevelBlockHankelRepresentation":
        """
        We currently only support concatenation.
        https://numpy.org/neps/nep-0018-array-function-protocol.html
        """
        if func != np.concatenate:
            return NotImplemented
        if not all(issubclass(t, BlockHankel) for t in types) or not issubclass(self.__class__, BlockHankel):
            return NotImplemented

        # check the keyword arguments
        axis = kwargs.get('axis', 0)
        if axis not in [0, 1]:
            raise ValueError('Concatenation only in the first two dimensions.')

        # check that we only received one arg
        if len(args) > 1:
            raise ValueError('We only accept one non keyword argument.')

        # create the list of matrices
        return MultilevelBlockHankelRepresentation(args[0], axis)


class BlockHankelRepresentation(BlockHankel):
    """
    This matrix represents a Block Hankel matrix
    for large matrices and fast_hankel=True it makes matrix multiplication faster:

    See our paper for more details:
    Efficient Hankel Matrix Decomposition for Changepoint Detection
    Lucas Weber, Richard Lenz 2024
    """

    def __init__(self, time_series: np.ndarray, end_index: int, window_length: int, window_number: int,
                 fast_hankel: bool):

        # save the parameters that are necessary for later computations
        self.window_length = window_length
        self.window_number = window_number
        self.fast_hankel = fast_hankel

        # get the hankel matrix block representation
        hankel, size = get_block_hankel_representation(time_series, end_index, window_length, window_number)
        self.size = size

        # create the representation and save it into the class
        if self.fast_hankel:

            # get the next fast fft length for efficient fft
            self.fft_length = spfft.next_fast_len(hankel.shape[0], real=True)

            # make the fft along the individual time series axis
            hankel = spfft.rfft(hankel, n=self.fft_length, axis=0)

        # save the hankel matrix and the shape
        self.hankel = hankel
        self.shape = (self.window_length*time_series.shape[1], self.window_number)

    def copy(self, deep: bool = False) -> "BlockHankelRepresentation":
        new = self.__class__.__new__(self.__class__)

        new.window_length = self.window_length
        new.window_number = self.window_number
        new.fast_hankel = self.fast_hankel
        new.shape = self.shape
        new.size = self.size

        if hasattr(self, "fft_length"):
            new.fft_length = self.fft_length

        if deep:
            new.hankel = self.hankel.copy()
        else:
            new.hankel = self.hankel

        return new

    def multiply_other_from_right(self, other_matrix: np.ndarray) -> np.ndarray:
        # check the dimensions
        if other_matrix.shape[0] != self.shape[1]:
            raise ValueError(f'matmul: Right matrix has a mismatch in its core dimension 0 '
                             f'(size {other_matrix.shape[0]=} is different from {self.shape[1]=})')

        # make the product based on whether we use the fast Hankel option
        if self.fast_hankel:
            return block_hankel_right_matmat_fft(self.hankel, other_matrix, self.fft_length, self.window_length, self.window_number)
        else:
            return block_hankel_right_matmat_direct(self.hankel, other_matrix)

    def multiply_other_from_left(self, other_matrix: np.ndarray) -> np.ndarray:

        # check the dimensions
        if other_matrix.shape[1] != self.shape[0]:
            raise ValueError(f'matmul: Left matrix has a mismatch in its core dimension 0 '
                             f'(size {other_matrix.shape[1]=} is different from {self.shape[0]=})')

        # make the product based on whether we use the fast Hankel option
        if self.fast_hankel:
            return block_hankel_left_matmat_fft(self.hankel, other_matrix, self.fft_length, self.window_length, self.window_number)
        else:
            return block_hankel_left_matmat_direct(self.hankel, other_matrix)

    @property
    def T(self) -> 'BlockHankelRepresentation':

        # make a copy of the object
        new = self.copy(deep=False)

        # switch the shape
        new.shape = tuple(reversed(new.shape))

        # switch the number of windows and window length
        new.window_length, new.window_number = new.window_number, new.window_length

        # switch the last dimensions of the hankel matrix
        new.hankel = new.hankel.transpose((0, 2, 1))
        return new

    def materialize(self) -> np.ndarray:
        """
        Explicitly build the full block Hankel matrix H for test multiplications.

        Definitions
        ----------
        p, q : the shape of the Hankel matrix (p rows, q columns)
        m, n : the shape of the blocks in the Hankel matrix (each block has m rows, n columns)

        B has shape (p + q - 1, m, n).
        H has shape (p*m, q*n).
        """
        hankel = self.hankel
        if self.fast_hankel:
            hankel = np.real_if_close(spfft.irfft(self.hankel, n=self.fft_length, axis=0))[:self.size,...]
        _, m, n = hankel.shape

        hankel_matrix_full = np.zeros(self.shape)
        for i in range(self.window_length):
            for j in range(self.window_number):
                hankel_matrix_full[i * m:(i + 1) * m, j * n:(j + 1) * n] = hankel[i + j]

        return hankel_matrix_full


class MultilevelBlockHankelRepresentation(BlockHankel):
    """
    This class implements multiplication with a multilevel Block Hankel matrix, i.e., a matrix which can be divided into
    multiple parts that each have Hankel structure.

    We expect to receive a list of:
    - Hankel matrices as BlockHankelRepresentation objects
    - The concatenation axis (currently only 0 and 1 are supported)
    """

    def __init__(self, matrix_tuples: list[BlockHankelRepresentation], axis: int):

        # check the input
        self.shape, self.individual_shapes = self.check_input(matrix_tuples, axis)

        # extract the matrices
        self.matrices = matrix_tuples
        self.axis = axis

    @staticmethod
    def check_input(input_list: list[BlockHankelRepresentation], axis: int) -> (int, int):

        # check whether the list is empty
        if len(input_list) < 1:
            raise ValueError('You did not input any Block Hankel matrices.')

        # check that the axis is either 0 or 1
        if axis not in (0, 1):
            raise ValueError(f'Concatenation only in the first two dimensions. Currently specified {axis=}.')

        # save the shapes
        shapes = [(0, 0)]*len(input_list)

        # check that we received only BlockHankelRepresentation matrices and the shapes match
        for idx, ele in enumerate(input_list):

            # check for the type of the class
            if not isinstance(ele, BlockHankelRepresentation):
                raise TypeError(f'We can only concatenate BlockHankelRepresentations in this class. Got {type(ele)}.')

            # save the dimension that we have to check
            shapes[idx] = ele.shape

        # decide which dimension we have to check for uniqueness
        # e.g., if we concatenate along axis 0, all axis 1 sizes have to match
        checkdim = 0 if axis == 1 else 1
        if len(set(ele[checkdim] for ele in shapes)) != 1:
            raise ValueError(f'We cannot concatenate BlockHankelRepresentations in this class due to different shapes in dimension {checkdim}: {shapes}.')

        if axis == 1:
            shape = (shapes[0][0], sum(ele[1] for ele in shapes))
        elif axis == 0:
            shape = (sum(ele[0] for ele in shapes), shapes[0][1])
        else:
            raise ValueError(f'{axis=} is invalid.')
        return shape, shapes

    def copy(self, deep: bool = False) -> "MultilevelBlockHankelRepresentation":
        new = self.__class__.__new__(self.__class__)

        if deep:
            new.matrices = [ele.copy(deep=deep) for ele in self.matrices]
        else:
            new.matrices = self.matrices
        new.axis = self.axis
        new.shape = self.shape

        if deep:
            new.individual_shapes = [ele.copy() for ele in self.individual_shapes]
        else:
            new.individual_shapes = self.individual_shapes
        return new

    @property
    def T(self) -> 'MultilevelBlockHankelRepresentation':
        """
        Compute the transposed.
        :return:
        """
        new = self.copy()
        # switch the shape
        new.shape = tuple(reversed(new.shape))

        # go through all of our matrices and transpose them
        new.matrices = [matrix.T for matrix in new.matrices]

        # update the shapes
        new.individual_shapes = [ele.shape for ele in new.matrices]

        # switch the axis
        new.axis = 1 if new.axis == 0 else 0

        return new

    def multiply_other_from_right(self, other: np.ndarray):

        # we need to create a target matrix
        result = np.zeros((self.shape[0], other.shape[1]))

        # check the matrix shapes
        if self.shape[1] != other.shape[0]:
            raise ValueError(f'Shape missmatch: {self.shape=} multiplied with {other.shape=} is not possible.')

        # choose depending on the axis of concatenation
        starting_point = 0
        if self.axis == 0:  # matrices are stacked ontop of each other
            for idx, matrix in enumerate(self.matrices):
                end_point = starting_point+self.individual_shapes[idx][0]
                result[starting_point:end_point, :] = matrix @ other
                starting_point = end_point
        if self.axis == 1:  # matrices are stacked next to each other
            for idx, matrix in enumerate(self.matrices):
                end_point = starting_point + starting_point+self.individual_shapes[idx][1]
                result += matrix @ other[starting_point:end_point, :]
                starting_point = end_point

        return result

    def multiply_other_from_left(self, other: np.ndarray):

        # we need to create a target matrix
        result = np.zeros((other.shape[0], self.shape[1]))

        # check the matrix shapes
        if self.shape[0] != other.shape[1]:
            raise ValueError(f'Shape missmatch: {other.shape=} multiplied with {self.shape=} is not possible.')

        # choose depending on the axis of concatenation
        starting_point = 0
        if self.axis == 0:  # matrices are stacked ontop of each other
            for idx, matrix in enumerate(self.matrices):
                end_point = starting_point + starting_point + self.individual_shapes[idx][0]
                result += other[:, starting_point:end_point] @ matrix
                starting_point = end_point
        if self.axis == 1:  # matrices are stacked next to each other
            for idx, matrix in enumerate(self.matrices):
                end_point = starting_point + self.individual_shapes[idx][1]
                result[:, starting_point:end_point] = other @ matrix
                starting_point = end_point
        return result

    def materialize(self) -> np.ndarray:
        """
        Explicitly build the full multilevel block Hankel matrix H for test multiplications.
        """

        return np.concatenate(tuple(mat.materialize()for mat in self.matrices), axis=self.axis)


class BlockHankelProductRepresentation(BlockHankel):
    """
    This class represents a matrix-matrix product of two hankel matrices.
    H @ H
    Instead of computing the product and materializing it, we keep both matrices so we have hankel property for
    products.

    See Section III-G in:
    L. Weber and R. Lenz,
    "Accelerating Singular Spectrum Transformation for Scalable Change Point Detection,"
    in IEEE Access, vol. 13, pp. 213556-213577, 2025, doi: 10.1109/ACCESS.2025.3640386.
    for details.
    """

    def __init__(self, first_hankel: BlockHankel, second_hankel: BlockHankel):

        # save the matrices for keeping
        self.first_hankel = first_hankel
        self.second_hankel = second_hankel

        # check that the dimensions match
        if self.first_hankel.shape[1] != self.second_hankel.shape[0]:
            raise ValueError(f'matmul: Operant shape missmatch: {self.first_hankel.shape} @ {self.second_hankel.shape} missmatch in inner dimensions.')

        # save the shape
        self.shape = (self.first_hankel.shape[0], self.second_hankel.shape[1])

    def copy(self, deep: bool = False) -> "BlockHankelProductRepresentation":
        new = self.__class__.__new__(self.__class__)

        # copy the matrices
        new.first_hankel = self.first_hankel.copy(deep=deep)
        new.second_hankel = self.second_hankel.copy(deep=deep)

        # copy the shape
        new.shape = self.shape
        return new

    @property
    def T(self):

        # make a copy of myself
        new = self.copy()

        # switch matrix order and transpose the matrices
        new.first_hankel, new.second_hankel = new.second_hankel.T, new.first_hankel.T

        # update the shape
        new.shape = tuple(reversed(new.shape))
        return new

    def multiply_other_from_right(self, other: np.ndarray) -> np.ndarray:
        return self.first_hankel @ (self.second_hankel @ other)

    def multiply_other_from_left(self, other: np.ndarray) -> np.ndarray:
        return (other @ self.first_hankel) @ self.second_hankel

    def materialize(self) -> np.ndarray:
        # we need to materialize the second otherwise, we will again only create a
        # BlockHankelProductRepresentation (the current class)
        #
        # materializing the second makes this a product of a BlockHankel with a np.ndarray
        # instead of BlockHankel with BlockHankel
        return self.first_hankel @ self.second_hankel.materialize()


def check_matrix_operations(hankel_typed_matrix: BlockHankel, materialized_hankel: np.ndarray, other_size: int = 10):

    # print the checks
    print()
    print(f"## Starting checks for matrix of type {hankel_typed_matrix.__class__.__name__}. ##")
    print('################################################')

    # check the shape
    assert hankel_typed_matrix.shape == materialized_hankel.shape, f'Shapes are off: {hankel_typed_matrix.shape=} != {materialized_hankel.shape=}.'
    assert materialized_hankel.ndim == 2, 'Materialized matrix is strange.'
    assert np.allclose(hankel_typed_matrix.materialize(), materialized_hankel), 'Hankel matrix is off.'

    # create the right matrix
    right_matrix = np.arange(0, materialized_hankel.shape[1]*other_size)
    right_matrix = right_matrix.reshape(materialized_hankel.shape[1], other_size)

    # check for right multiplication
    naive_matmul = materialized_hankel @ right_matrix
    hankel_typed_matmul = hankel_typed_matrix @ right_matrix
    is_close = np.allclose(naive_matmul, hankel_typed_matmul)
    print('Right multiplication valid?', is_close)
    assert is_close, 'Right multiplication failed.'

    # create the left matrix
    left_matrix = np.arange(0, materialized_hankel.shape[0] * other_size)
    left_matrix = left_matrix.reshape(other_size, materialized_hankel.shape[0])

    # check for left multiplication
    naive_matmul = left_matrix @ materialized_hankel
    hankel_typed_matmul = left_matrix @ hankel_typed_matrix
    is_close = np.allclose(naive_matmul, hankel_typed_matmul)
    print('Left multiplication valid?', is_close)
    assert is_close, 'Left multiplication failed.'

    # check for transpose
    assert np.allclose(hankel_typed_matrix.T.materialize(), materialized_hankel.T), 'Transpose does not work.'

    # check for right transposed multiplication
    naive_matmul = materialized_hankel.T @ left_matrix.T
    hankel_typed_matmul = hankel_typed_matrix.T @ left_matrix.T
    is_close = np.allclose(naive_matmul, hankel_typed_matmul)
    print('Transposed right multiplication valid?', is_close)
    assert is_close, 'Transposed right multiplication failed.'

    # check for left transposed multiplication
    naive_matmul = right_matrix.T @ materialized_hankel.T
    hankel_typed_matmul = right_matrix.T @ hankel_typed_matrix.T
    is_close = np.allclose(naive_matmul, hankel_typed_matmul)
    print('Transposed left multiplication valid?', is_close)
    assert is_close, 'Transposed left multiplication failed.'
    print('################################################')
    print()


def main():
    import timeit
    # -------------------------
    # Example
    # -------------------------

    p = 70  # block rows of H
    q = 60  # block columns of H
    m = 2  # rows per block
    s = 15  # rows of left multiplier A

    # make a signal of the correct shape
    signal = np.arange(0, (p+q-1)*m)
    signal = signal.reshape(p+q-1, m)

    # make the Hankel block matrix representation
    hankel_direct = BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False)
    hankel_fft = BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=True)
    hankel_naive = hankel_direct.materialize()

    # check whether the materialized hankel matrices are equal
    print('Materialized Hankel matrices equal?', np.allclose(hankel_direct.materialize(), hankel_fft.materialize()))

    # make some checks using the functions
    check_matrix_operations(hankel_fft, hankel_naive, s)
    check_matrix_operations(hankel_direct, hankel_naive, s)

    # create right matrix
    right_matrix = np.arange(0, q * s)
    right_matrix = right_matrix.reshape(q, s)

    # make a print to start multiplication from the right
    print()
    print('Starting multiplication timing from the RIGHT-------')

    run_num = 100
    fft_time = timeit.timeit(lambda: BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=True) @ right_matrix, number=run_num) / run_num * 1000
    print('FFT right multiplication took:', fft_time)
    direct_time = timeit.timeit(lambda: BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False) @ right_matrix, number=run_num) / run_num * 1000
    print('Direct right multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False).materialize() @ right_matrix, number=run_num) / run_num * 1000
    print('Naive right multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: hankel_naive @ right_matrix, number=run_num) / run_num * 1000
    print('Naive (already constructed, unfair) right multiplication took:', direct_time)

    # make a print to start multiplication from the left
    print()
    print('Starting multiplication timing from the LEFT-------')

    # create left matrix
    left_matrix = np.arange(0, p * m * s)
    left_matrix = left_matrix.reshape(s, p*m)

    run_num = 100
    fft_time = timeit.timeit(lambda: left_matrix @ BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=True), number=run_num) / run_num * 1000
    print('FFT left multiplication took:', fft_time)
    direct_time = timeit.timeit(lambda: left_matrix @ BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False), number=run_num) / run_num * 1000
    print('Direct left multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: left_matrix @ BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False).materialize(), number=run_num) / run_num * 1000
    print('Naive left multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: left_matrix @ hankel_naive,number=run_num) / run_num * 1000
    print('Naive (already constructed, unfair) left multiplication took:', direct_time)

    # make a print to start working on concatenated hankel matrices
    print()
    print('Starting working on concatenated matrices-------')

    naive_multilevel = np.concatenate((hankel_naive, hankel_naive), axis=1)
    direct_multilevel = np.concatenate((hankel_direct, hankel_direct), axis=1)
    fft_multilevel = np.concatenate((hankel_fft, hankel_fft), axis=1)

    # make some checks using the functions
    check_matrix_operations(fft_multilevel, naive_multilevel, s)
    check_matrix_operations(direct_multilevel, naive_multilevel, s)

    # create the right matrix
    right_matrix = np.arange(0, naive_multilevel.shape[1] * s)
    right_matrix = right_matrix.reshape(naive_multilevel.shape[1], s)

    # create the right matrix
    left_matrix = np.arange(0, naive_multilevel.shape[0] * s)
    left_matrix = left_matrix.reshape(s, naive_multilevel.shape[0])

    run_num = 100
    fft_time = timeit.timeit(
        lambda: np.concatenate((BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=True),
                                BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=True)), axis=1) @ right_matrix,
        number=run_num) / run_num * 1000
    print('FFT right multiplication took:', fft_time)
    direct_time = timeit.timeit(
        lambda: np.concatenate((BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False),
                                BlockHankelRepresentation(signal, p+q-1, p, q, fast_hankel=False)), axis=1) @ right_matrix,
        number=run_num) / run_num * 1000
    print('Direct right multiplication took:', direct_time)
    direct_time = timeit.timeit(
        lambda: np.concatenate((hankel_direct.materialize(), hankel_direct.materialize()), axis=1) @ right_matrix,
        number=run_num) / run_num * 1000
    print('Naive right multiplication took:', direct_time)
    direct_time = timeit.timeit(
        lambda: np.concatenate((hankel_naive, hankel_naive), axis=1) @ right_matrix,
        number=run_num) / run_num * 1000
    print('Naive right (already constructed, unfair) multiplication took:', direct_time)

    run_num = 100
    fft_time = timeit.timeit(
        lambda: left_matrix @ np.concatenate((BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=True),
                                              BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=True)),
                                             axis=1),
        number=run_num) / run_num * 1000
    print('FFT left multiplication took:', fft_time)
    direct_time = timeit.timeit(
        lambda: left_matrix @ np.concatenate((BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False),
                                              BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False)),
                                             axis=1),
        number=run_num) / run_num * 1000
    print('Direct left multiplication took:', direct_time)
    direct_time = timeit.timeit(
        lambda: left_matrix @ np.concatenate((hankel_direct.materialize(), hankel_direct.materialize()), axis=1),
        number=run_num) / run_num * 1000
    print('Naive left multiplication took:', direct_time)
    direct_time = timeit.timeit(
        lambda: left_matrix @ np.concatenate((hankel_naive, hankel_naive), axis=1),
        number=run_num) / run_num * 1000
    print('Naive left (already constructed, unfair) multiplication took:', direct_time)

    # make a print to start working on product hankel matrices
    print()
    print('Starting working on product Hankel matrices-------')

    hankel_product_direct = hankel_direct @ hankel_direct.T
    hankel_product_fft = hankel_fft @ hankel_fft.T
    hankel_product_naive = hankel_naive @ hankel_naive.T

    check_matrix_operations(hankel_product_direct, hankel_product_naive)
    check_matrix_operations(hankel_product_fft, hankel_product_naive)

    # create the right matrix
    right_matrix = np.arange(0, hankel_product_naive.shape[1] * s)
    right_matrix = right_matrix.reshape(hankel_product_naive.shape[1], s)

    # create the left matrix
    left_matrix = np.arange(0, hankel_product_naive.shape[0] * s)
    left_matrix = left_matrix.reshape(s, hankel_product_naive.shape[0])

    run_num = 100
    fft_time = timeit.timeit(lambda: left_matrix @ (hankel_fft @ hankel_fft.T), number=run_num) / run_num * 1000
    print('FFT left multiplication took:', fft_time)
    direct_time = timeit.timeit(lambda: left_matrix @ (hankel_direct @ hankel_direct.T),
                                number=run_num) / run_num * 1000
    print('Direct left multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: left_matrix @ (hankel_direct.materialize() @ hankel_direct.materialize().T),
                                number=run_num) / run_num * 1000
    print('Naive left multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda:left_matrix @ (hankel_naive @ hankel_naive.T), number=run_num) / run_num * 1000
    print('Naive left (already constructed, unfair) multiplication took:', direct_time)

    run_num = 100
    fft_time = timeit.timeit(lambda: (hankel_fft @ hankel_fft.T) @ right_matrix, number=run_num) / run_num * 1000
    print('FFT right multiplication took:', fft_time)
    direct_time = timeit.timeit(lambda: (hankel_direct @ hankel_direct.T) @ right_matrix,
                                number=run_num) / run_num * 1000
    print('Direct right multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: (hankel_direct.materialize() @ hankel_direct.materialize().T)  @ right_matrix,
                                number=run_num) / run_num * 1000
    print('Naive right multiplication took:', direct_time)
    direct_time = timeit.timeit(lambda: (hankel_naive @ hankel_naive.T)  @ right_matrix, number=run_num) / run_num * 1000
    print('Naive right (already constructed, unfair) multiplication took:', direct_time)



if __name__ == '__main__':
    main()
