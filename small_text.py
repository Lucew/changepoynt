import time

import numpy as np
import matplotlib.pyplot as plt

from changepoynt.algorithms.sst import SST

normal = 0
ws = 120
cached = 0
series = np.random.random((5000,))

for ws in range(20, 400, 10):

    if normal == 0 or normal == 1:
        sst_normal = SST(ws, method='rsvd', use_fast_hankel=ws > 200)
        if cached < 2:
            sst_normal.transform(series)
            cached += 1
        start = time.perf_counter()
        score_normal = sst_normal.transform(series)
        normal_time = time.perf_counter() - start
        print('Normal time:', time.perf_counter() - start)
    if normal == 0 or normal == 2:
        sst_cached = SST(ws, method='cached rsvd', use_fast_hankel=ws > 200)
        if cached < 2:
            sst_cached.transform(series)
            cached += 1
        start = time.perf_counter()
        score_cached = sst_cached.transform(series)
        cached_time = time.perf_counter() - start
        print('Cached time:', time.perf_counter() - start)

    if normal == 0:
        print('Scores are equal:', np.allclose(score_normal, score_cached))
        print('Quotient:', normal_time / cached_time, 'Diff:', normal_time - cached_time)
        # plt.plot(score_normal)
        # plt.plot(score_cached)
        # plt.show()
    print()