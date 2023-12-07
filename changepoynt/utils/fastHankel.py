import os
import time

import numpy as np
import pandas as pd
import scipy as sp
import timeit
import changepoynt.utils.linalg as lg
from threadpoolctl import threadpool_limits
import torch
import stumpy


def get_fast_hankel_representation(time_series, end_index, length_windows, number_windows, lag=1):

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
    signal = time_series[end_index-lag*(number_windows-1)-length_windows:end_index]

    # get the length of the matrices
    # col_length = last_column.shape[0]
    # row_length = row_without_last_element.shape[0]
    # combined_length = col_length + row_length

    # compute the padded vector length for an optimal fft length. Here we can use the built-in scipy function that
    # computes the perfect fft length optimized for their implementation, so it is even faster than with powers
    # of two!
    # fft_len = 1 << int(np.ceil(np.log2(combined_length)))
    # fft_len = sp.fft.next_fast_len(combined_length, True)
    fft_len = sp.fft.next_fast_len(signal.shape[0], True)

    # compute the fft over the padded hankel matrix
    # if we would pad in the middle like this: we would introduce no linear phase to the fft
    #
    # padded_toeplitz = np.concatenate((last_column, np.zeros((fft_len - combined_length)), row_without_last_element))
    #
    # but we want to use the built-in padding functionality of the fft in scipy, so we pad at the end like can be seen
    # here. This introduces and linear phase and shift to the fft, which we need to account for in the reverse
    # functions.
    #
    # More details see:
    # https://dsp.stackexchange.com/questions/82273/why-to-pad-zeros-at-the-middle-of-sequence-instead-at-the-end-of-the-sequence
    # https://dsp.stackexchange.com/questions/83461/phase-of-an-fft-after-zeropadding
    hankel_rfft = sp.fft.rfft(signal, n=fft_len, axis=0).reshape(-1, 1)
    return hankel_rfft, fft_len, signal


def fast_hankel_matmul(hankel_fft: np.ndarray, l_windows, fft_shape: int, other_matrix: np.ndarray, lag, workers=None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = 1

    # save the shape of the matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros)
    if lag > 1:
        out = np.zeros((lag*m-lag+1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    fft_x = sp.fft.rfft(np.flipud(other_matrix), n=fft_shape, workers=workers, axis=0)

    # compute the inverse fft and take into account the offset due to circular convolution and zero padding as explained
    # in
    # https://dsp.stackexchange.com/questions/82273/why-to-pad-zeros-at-the-middle-of-sequence-instead-at-the-end-of-the-sequence
    # and
    # https://dsp.stackexchange.com/questions/83461/phase-of-an-fft-after-zeropadding
    mat_times_x = sp.fft.irfft(hankel_fft*fft_x, axis=0, n=fft_shape, workers=workers)[(m-1)*lag:(m-1)*lag+l_windows, :]

    return_shape = (l_windows,) if len(other_matrix.shape) == 1 else (l_windows, n)
    return mat_times_x.reshape(*return_shape)


def fast_hankel_left_matmul(hankel_fft: np.ndarray, n_windows, fft_shape: int, other_matrix: np.ndarray, lag, workers=None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = 1

    # transpose the other matrix
    other_matrix = other_matrix.T

    # save the shape of the matrix
    m, n = other_matrix.shape

    # make fft of x (while padding x with zeros)
    fft_x = sp.fft.rfft(np.flipud(other_matrix), n=fft_shape, workers=workers, axis=0)

    # compute the inverse fft
    mat_times_x = sp.fft.irfft(hankel_fft * fft_x, axis=0, n=fft_shape, workers=workers)[(m-1):(m-1)+n_windows*lag:lag]

    return_shape = (n_windows,) if len(other_matrix.shape) == 1 else (n_windows, n)
    return mat_times_x.reshape(*return_shape).T


def fast_fftconv_hankel_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag, workers=None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = 1

    if lag > 1:
        m, n = other_matrix.shape
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    result = np.stack([sp.signal.fftconvolve(hankel_signal, other_matrix[::-1, col], mode="valid") for col in range(other_matrix.shape[1])]).T
    return result


def fast_fftconv_hankel_left_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag, workers=None):
    # This code has been inspired by:
    #
    # Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    # Pyroomacoustics: https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/adaptive/util.py
    #
    # Fast Vector product is only a convolution of vector over signal!
    # Fastmat: https://fastmat.readthedocs.io/en/latest/classes/Toeplitz.html#fastmat.Toeplitz

    # check the workers
    if not workers:
        workers = 1
    result = np.stack([sp.signal.fftconvolve(hankel_signal, other_matrix[row, ::-1], mode="valid")[::lag] for row in range(other_matrix.shape[0])])
    return result


def fast_convolve_hankel_matmul(hankel_signal: np.ndarray, other_matrix: np.ndarray, lag):
    if lag > 1:
        m, n = other_matrix.shape
        out = np.zeros((lag * m - lag + 1, n), dtype=other_matrix.dtype)
        out[::lag, :] = other_matrix
        other_matrix = out
    result = np.stack([np.convolve(hankel_signal, other_matrix[::-1, col], mode="valid") for col in range(other_matrix.shape[1])]).T
    return result


def fast_torch_hankel_matmul(hankel_repr: torch.Tensor, other_matrix: torch.Tensor, lag):
    with torch.no_grad():
        if lag > 1:
            m, b, n = other_matrix.shape
            out = torch.zeros((m, b, lag * n - lag + 1), dtype=other_matrix.dtype)
            out[:, :, ::lag] = other_matrix
            other_matrix = out
        result = torch.nn.functional.conv1d(hankel_repr, other_matrix)
        result = result[0, :, :].transpose(0, 1)
        return result.detach().cpu().numpy()


def normal_hankel_matmul(hankel, other):
    return hankel @ other


def normal_hankel_left_matmul(hankel, other):
    return other @ hankel


def normal_hankel_inner(hankel):
    return hankel.T @ hankel


def evaluate_closeness(m1: np.ndarray, m2: np.ndarray, comment: str):
    absdiff = np.abs(m1-m2)
    assert m1.dtype == m2.dtype, f"Matrices need to have similar datatypes m1 has: {m1.dtype}, m2 has {m2.dtype}"
    return {"Comment": comment,
            "Is Close": np.allclose(m1, m2),
            "Max. Diff.": np.max(absdiff),
            "Median Diff.": np.median(absdiff),
            "Std. Diff.": np.std(absdiff),
            "Machine Precision": np.finfo(m1.dtype).eps}


def print_table(my_tuples: list[dict]):
    """
    Pretty print a list of dictionaries (my_dict) as a dynamically sized table.
    """

    # get the column list and check whether it is available in all dicts
    col_keys = [key for key in my_tuples[0].keys()]
    cols = {key: [f"{key}"] for key in col_keys}
    for idx, tupled in enumerate(my_tuples):

        # check whether they are available
        assert all(ele in tupled for ele in col_keys), (f"Tuple {idx} does not have the expected cols: {col_keys},"
                                                        f" it has: {list(tupled.keys())}")

        # insert the tuples into the columns
        for key, val in cols.items():
            val.append(f"{tupled[key]}")

    # check the column width for each column
    col_width = []
    for key in col_keys:
        col_width.append((key, max(map(len, cols[key]))))

    # print the stuff
    # formatting = "|" + "|".join(f"{{{idx}: <{width}" for idx, (_, width) in enumerate(col_width))
    header = "|".join(key.center(width+5) for key, width in col_width)
    print(header)
    print("-"*len(header))
    for tupled in my_tuples:
        text = "|".join(f"{tupled[key]}".ljust(width+5) for key, width in col_width)
        print(text)


def probe_fast_hankel_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fast_hankel_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag)
    way_two = normal_hankel_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_hankel_left_matmul(hankel_repr, l_windows, fft_len, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fast_hankel_left_matmul(hankel_repr, l_windows, fft_len, other_matrix, lag)
    way_two = normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_convolve_hankel_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fast_convolve_hankel_matmul(hankel_repr, other_matrix, lag)
    way_two = normal_hankel_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_fftconvolve_hankel_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fast_fftconv_hankel_matmul(hankel_repr, other_matrix, lag)
    way_two = normal_hankel_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_fftconvolve_hankel_left_matmul(hankel_repr, hankel_matrix, other_matrix, lag, comment: str):
    way_one = fast_fftconv_hankel_left_matmul(hankel_repr, other_matrix, lag)
    way_two = normal_hankel_left_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_torch_hankel_matmul(hankel_fft, hankel_matrix, other_matrix, other_matrix_torch, lag, comment: str):
    way_one = fast_torch_hankel_matmul(hankel_fft, other_matrix_torch, lag)
    way_two = normal_hankel_matmul(hankel_matrix, other_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def probe_fast_hankel_inner_product(hankel_fft, n_windows, fft_len, hankel_matrix, lag, comment: str):
    way_one = fast_hankel_left_matmul(hankel_fft, n_windows, fft_len, hankel_matrix.T, lag)
    way_two = normal_hankel_inner(hankel_matrix)
    return evaluate_closeness(way_one, way_two, comment)


def run_measurements(thread_counts: list[int],
                     window_lengths: list[int],
                     window_numbers: list[int],
                     signal_scaling: list[int],
                     other_matrix_dimensions: list[int],
                     other_matrix_scaling: list[int],
                     lags: list[int],
                     runs: int = 1000):

    # get the performance counter
    perc = time.perf_counter_ns

    # try to import tqdm
    from tqdm import tqdm

    # make a dict to save the values
    values = {"Naive Execution Time (over all Runs) Left Product Other@Hankel": [],
              "Naive Execution Time (over all Runs) Right Product Hankel@Other": [],
              "FFT Execution Time (over all Runs) Left Product Other@Hankel": [],
              "FFT Execution Time (over all Runs) Right Product Hankel@Other": [],
              "Window Length": [],
              "Window Number": [],
              "Signal Scaling": [],
              "Lag": [],
              "Other Matrix Dim.": [],
              "Other Matrix Scaling": [],
              "Thread Count": [],
              "Runs": [],
              "Median Diff. Left Product Other@Hankel": [],
              "Median Diff. Right Product Hankel@Other": [],
              "Max. Diff. Left Product Other@Hankel": [],
              "Max. Diff. Right Product Hankel@Other": [],
              "Std. Diff. Left Product Other@Hankel": [],
              "Std. Diff. Right Product Hankel@Other": [],
              "Machine Precision": []}

    # create a generator to make the tuples
    tuple_generator = [(wl, wn, sc, om, omsc, lag, tc)
                       for wl in window_lengths
                       for wn in window_numbers
                       for lag in lags
                       for sc in signal_scaling
                       for om in other_matrix_dimensions
                       for omsc in other_matrix_scaling
                       for tc in thread_counts]

    for (wl, wn, sc, om, omsc, lag, tc) in tqdm(tuple_generator, "Compute all the tuples"):
        # this limits the threads for numpy (at least for our version)
        with threadpool_limits(limits=tc):

            # save the parameters
            values["Window Length"].append(wl)
            values["Window Number"].append(wn)
            values["Signal Scaling"].append(sc)
            values["Lag"].append(lag)
            values["Other Matrix Dim."].append(om)
            values["Other Matrix Scaling"].append(omsc)
            values["Thread Count"].append(tc)
            values["Runs"].append(runs)

            # compute the necessary length for the signal
            end_idx = wl + lag * (wn - 1)

            # make the random signal with the given scale
            signal = np.random.uniform(size=(end_idx+10,))*sc

            # create the matrix representation
            hankel_rfft, fft_len, signal = get_fast_hankel_representation(signal, end_idx, wl, wn, lag)
            hankel = lg.compile_hankel(signal, end_idx, wl, wn, lag)

            # measure the time to get the matrix representation
            repr_time = timeit.timeit(lambda: get_fast_hankel_representation(signal, end_idx, wl, wn, lag),
                                      number=runs,
                                      timer=perc)

            # create the other two matrices we want to multiply with
            other_right = np.random.uniform(size=(wn, om))*omsc
            other_left = np.random.uniform(size=(om, wl))*omsc

            # measure the multiplication time for naive implementation
            naive_time_right = timeit.timeit(lambda: normal_hankel_matmul(hankel, other_right),
                                             number=runs,
                                             timer=perc)
            values["Naive Execution Time (over all Runs) Right Product Hankel@Other"].append(naive_time_right)
            naive_time_left = timeit.timeit(lambda: normal_hankel_left_matmul(hankel, other_left),
                                            number=runs,
                                            timer=perc)
            values["Naive Execution Time (over all Runs) Left Product Other@Hankel"].append(naive_time_left)

            # measure the multiplication time for fft implementation
            fft_time_right = timeit.timeit(lambda: fast_hankel_matmul(hankel_rfft, wl, fft_len,
                                                                      other_right, lag, workers=tc),
                                           number=runs,
                                           timer=perc)
            values["FFT Execution Time (over all Runs) Right Product Hankel@Other"].append(fft_time_right+repr_time)
            fft_time_left = timeit.timeit(lambda: fast_hankel_left_matmul(hankel_rfft, wn, fft_len,
                                                                          other_left, lag, workers=tc),
                                          number=runs,
                                          timer=perc)
            values["FFT Execution Time (over all Runs) Left Product Other@Hankel"].append(fft_time_left+repr_time)

            # compute the products for error estimation
            naive_right_product = normal_hankel_matmul(hankel, other_right)
            naive_left_product = normal_hankel_left_matmul(hankel, other_left)
            fft_right_product = fast_hankel_matmul(hankel_rfft, wl, fft_len, other_right, lag, workers=tc)
            fft_left_product = fast_hankel_left_matmul(hankel_rfft, wn, fft_len, other_left, lag, workers=tc)

            # compute the errors for right product
            right_error = evaluate_closeness(naive_right_product, fft_right_product, "")
            values["Median Diff. Right Product Hankel@Other"].append(right_error["Median Diff."])
            values["Max. Diff. Right Product Hankel@Other"].append(right_error["Max. Diff."])
            values["Std. Diff. Right Product Hankel@Other"].append(right_error["Std. Diff."])
            values["Machine Precision"].append(right_error["Machine Precision"])

            # compute the errors for left product
            left_error = evaluate_closeness(naive_left_product, fft_left_product, "")
            values["Median Diff. Left Product Other@Hankel"].append(left_error["Median Diff."])
            values["Max. Diff. Left Product Other@Hankel"].append(left_error["Max. Diff."])
            values["Std. Diff. Left Product Other@Hankel"].append(left_error["Std. Diff."])
            assert right_error["Machine Precision"] == left_error["Machine Precision"], "Something odd with eps."

    # create a dataframe and save the results
    df = pd.DataFrame(values)
    df.to_csv("Results_HankelMult.csv")


def main():
    # define some window length
    limit_threads = 12
    l_windows = 1200
    n_windows = 1200
    lag = 1
    run_num = 1000

    # create a time series of a certain length
    n = 30000
    ts = np.random.uniform(size=(n,))*1000
    # ts = np.linspace(0, n, n+1)

    # create a matrix to multiply by
    multi = np.random.uniform(size=(n_windows, 10))
    multi2 = np.random.uniform(size=(10, l_windows))

    # get the final index of the time series
    end_idx = l_windows+lag*(n_windows-1)

    # get both hankel representations
    hankel_rfft, fft_len, signal = get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, lag)
    hankel = lg.compile_hankel(ts, end_idx, l_windows, n_windows, lag)

    # use the torch function
    torch_sig = torch.from_numpy(signal)
    torch_sig = torch_sig[None, None, :]
    torch_mat = torch.from_numpy(multi)
    torch_mat = torch_mat[:, None, :]
    torch_mat = torch_mat.transpose(0, 2)

    # test the faster multiplication
    results = []
    results.append(probe_fast_hankel_matmul(hankel_rfft, l_windows, fft_len, hankel, multi, lag, 'Matmul working?'))
    results.append(probe_fast_hankel_left_matmul(hankel_rfft, n_windows, fft_len, hankel, multi2, lag, 'Left Matmul working?'))
    results.append(probe_fast_torch_hankel_matmul(torch_sig, hankel, multi, torch_mat, lag, 'Matmul torch working?'))
    results.append(probe_fast_convolve_hankel_matmul(signal, hankel, multi, lag, 'Matmul convolve working?'))
    results.append(probe_fast_fftconvolve_hankel_matmul(signal, hankel, multi, lag, 'Matmul fftconvolve working?'))
    results.append(probe_fast_fftconvolve_hankel_left_matmul(signal, hankel, multi2, lag, 'Left Matmul fftconvolve working?'))
    # results.append(probe_fast_hankel_inner_product(hankel_rfft, n_windows, fft_len, hankel, lag, 'Inner product working?'))
    print_table(results)

    # check for execution time of both approaches
    print()
    header = f"Measure some times for {run_num} repetitions using {limit_threads} threads and hankel of size {l_windows}*{n_windows}"
    print(header)
    print("-"*len(header))
    with threadpool_limits(limits=limit_threads):

        print("Times for rfft signal:")
        rfft_time = timeit.timeit(lambda: get_fast_hankel_representation(ts, end_idx, l_windows, n_windows, lag), number=run_num) / run_num * 1000
        print(rfft_time)

        print("Times for Own:")
        print(timeit.timeit(lambda: fast_hankel_matmul(hankel_rfft, l_windows, fft_len, multi, lag, workers=limit_threads), number=run_num)/run_num*1000+rfft_time)
        print(timeit.timeit(lambda: fast_hankel_left_matmul(hankel_rfft, n_windows, fft_len, multi2, lag, workers=limit_threads), number=run_num) / run_num * 1000+rfft_time)

        print("Times for FFTconv:")
        print(timeit.timeit(lambda: fast_fftconv_hankel_matmul(signal, multi, lag, workers=limit_threads), number=run_num) / run_num * 1000)
        print(timeit.timeit(lambda: fast_fftconv_hankel_left_matmul(signal, multi2, lag, workers=limit_threads), number=run_num) / run_num * 1000)
        # print(timeit.timeit(lambda: fast_torch_hankel_matmul(torch_sig, torch_mat, lag), number=run_num) / run_num * 1000)
        # print(timeit.timeit(lambda: fast_convolve_hankel_matmul(signal, multi, lag), number=run_num) / run_num * 1000)

        print("Times for Naive:")
        print(timeit.timeit(lambda: normal_hankel_matmul(hankel, multi), number=run_num)/run_num*1000)
        print(timeit.timeit(lambda: normal_hankel_left_matmul(hankel, multi2), number=run_num) / run_num * 1000)

        print("Times for Naive inner Product:")
        # print(timeit.timeit(lambda: normal_hankel_inner(hankel), number=run_num) / run_num * 1000)
        # print(timeit.timeit(lambda: fast_hankel_left_matmul(hankel_rfft, n_windows, fft_len, hankel.T, lag), number=run_num) / run_num * 1000)


if __name__ == "__main__":
    # main()
    run_measurements(thread_counts=[1, 6, 12],
                     window_lengths=list(np.geomspace(10, 20_000, num=100, dtype=int)),
                     window_numbers=list(np.geomspace(10, 20_000, num=100, dtype=int)),
                     signal_scaling=[1, 10],
                     other_matrix_dimensions=[5, 10, 20, 30, 40, 50],
                     other_matrix_scaling=[1, 10],
                     lags=[1, 7],
                     runs=100)
