"""
This is a minimal example comparing the different approaches mentioned in the paper.
please run:
`pip install changepoynt tqdm`
before running the example

In with the parameters in our version, this example took around 2 minutes on a modern but standard laptop.
"""
import time

import numpy as np
from tqdm import tqdm

from changepoynt.algorithms.sst import SST

SIZE_FACTOR = 200

# simulate a signal that goes from exponential decline into a sine wave
# the signals is only for demonstration purposes and can be replaced by your signal
steady_before = np.ones(20*SIZE_FACTOR)
exp_signal = np.exp(-np.linspace(0, 5, 20*SIZE_FACTOR))
steady_after = np.exp(-5)*np.ones(15)
sine_after = 0.2*np.sin(np.linspace(0, 3*np.pi*10, 30*SIZE_FACTOR))
signal = np.concatenate((steady_before, exp_signal, steady_after, sine_after))
signal += 0.02*np.random.randn(signal.shape[0])  # add some minor noise


# create each of the detectors
# We enable the fast Hankel matrix product by: use_fast_hankel=True
# In order to save time, we only compute the change score at ever 50th sample (scoring_step=50)
window_size = 3*SIZE_FACTOR
scoring_step = 50
FFT_RSVD_SST = SST(window_size, method='rsvd', use_fast_hankel=True, scoring_step=scoring_step)
RSVD_SST = SST(window_size, method='rsvd', scoring_step=scoring_step)
FFT_IKA_SST = SST(window_size, method='ika', use_fast_hankel=True, scoring_step=scoring_step)
IKA_SST = SST(window_size, method='ika', scoring_step=scoring_step)
NAIVE_SVD = SST(window_size, method='naive', scoring_step=scoring_step)

# create an iterable with the detectors
detectors = {f'{NAIVE_SVD=}'.split('=')[0]: NAIVE_SVD, f'{RSVD_SST=}'.split('=')[0]: RSVD_SST, f'{FFT_RSVD_SST=}'.split('=')[0]: FFT_RSVD_SST, f'{IKA_SST=}'.split('=')[0]:IKA_SST, f'{FFT_IKA_SST=}'.split('=')[0]: FFT_IKA_SST}
namelength = max(len(key) for key in detectors.keys())

# go through each of the detectors (and keep the score of the naive method)
score = np.zeros_like(signal)
durations = {name: 0.0 for name in detectors.keys()}
errors = {name: -1 for name in detectors.keys()}
for name, detector in tqdm(detectors.items(), desc="Running detectors"):

    # we have to trigger the JIT compiler first (otherwise we measure compile time)
    if not name.startswith('NAIVE'):
        _ = detector.transform(signal)

    # apply the detectors
    # as the decompositions are applied for several steps, we already get a balanced time measurement for several
    # decompositions that accounts for processing time variations due to CPU scheduling
    start = time.perf_counter()
    method_score = detector.transform(signal)
    durations[name] = time.perf_counter() - start

    # create the ground truth from the naive method
    if name.startswith('NAIVE'):
        score = method_score

    # compute the error
    assert not np.all(score == 0), 'Naive method has to be the first.'
    errors[name] = np.mean(np.abs(score - method_score))

# print the results to console
for name in detectors.keys():
    duration = durations[name]
    error = errors[name]
    print(f"{name: <{namelength}} took {duration:.2f} seconds for {signal.shape[0]//scoring_step} samples and N={window_size}.")
    print(f"Mean absolute error to naive SST: {error:.3E}")
    print()