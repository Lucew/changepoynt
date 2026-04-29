import numpy as np
import pytest

import changepoynt.utils.block_linalg as blg


def _setup_signal():
    p = 70  # block rows of H
    q = 60  # block columns of H
    m = 2  # rows per block
    s = 15  # rows of left multiplier A

    # make a signal of the correct shape
    signal = np.arange(0, (p + q - 1) * m)
    signal = signal.reshape(p + q - 1, m)

    return signal, p, q, m, s



def test_block_hankel_direct():
    # make the signal
    signal, p, q, m, s = _setup_signal()

    # make the Hankel block matrix representation
    hankel_direct = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False)
    hankel_naive = hankel_direct.materialize()

    # run the checks
    blg.check_matrix_operations(hankel_direct, hankel_naive, s)


def test_block_hankel_fft():

    # make the signal
    signal, p, q, m, s = _setup_signal()

    # make the Hankel block matrix representation
    hankel_direct = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False)
    hankel_naive = hankel_direct.materialize()
    hankel_fft = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=True)

    # run the checks
    blg.check_matrix_operations(hankel_fft, hankel_naive, s)


def test_wrong_signal_shape():
    # make the signal
    signal, p, q, m, s = _setup_signal()

    # make the Hankel block matrix representation
    with pytest.raises(ValueError):
        hankel_direct = blg.BlockHankelRepresentation(signal[:, 0], p + q - 1, p, q, fast_hankel=False)


def test_multilevel_block_hankel_fft():

    # make the signal
    signal, p, q, m, s = _setup_signal()
    geng = np.random.default_rng(42)
    signal2 = signal + geng.random(size=signal.shape)

    # make the Hankel block matrix representation
    hankel_direct = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False)
    hankel_direct2 = blg.BlockHankelRepresentation(signal2, p + q - 1, p, q, fast_hankel=False)
    hankel_naive = hankel_direct.materialize()
    hankel_naive2 = hankel_direct2.materialize()
    hankel_fft = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=True)
    hankel_fft2 = blg.BlockHankelRepresentation(signal2, p + q - 1, p, q, fast_hankel=True)

    for axis in (0, 1):

        naive_multilevel = np.concatenate((hankel_naive, hankel_naive2), axis=axis)
        direct_multilevel = np.concatenate((hankel_direct, hankel_direct2), axis=axis)
        fft_multilevel = np.concatenate((hankel_fft, hankel_fft2), axis=axis)

        # make some checks using the functions
        blg.check_matrix_operations(fft_multilevel, naive_multilevel, s)
        blg.check_matrix_operations(direct_multilevel, naive_multilevel, s)

def test_wrong_axis_multilevel_block_hankel():
    # make the signal
    signal, p, q, m, s = _setup_signal()
    geng = np.random.default_rng(42)
    signal2 = signal + geng.random(size=signal.shape)

    # make the Hankel block matrix representation
    hankel_direct = blg.BlockHankelRepresentation(signal, p + q - 1, p, q, fast_hankel=False)
    hankel_direct2 = blg.BlockHankelRepresentation(signal2, p + q - 1, p, q, fast_hankel=False)

    # check whether we stop wrong axis
    with pytest.raises(ValueError):
        direct_multilevel = np.concatenate((hankel_direct, hankel_direct2), axis=2)