import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import seaborn as sns
from changepoynt.algorithms.sst import SingularSpectrumTransformation
import fastsst
import time
import pandas as pd


# some plotting utilities
def plot_data_and_score(raw_data, own_score, other_score, normalize=True):
    # normalize the scores
    own_score = own_score / np.max(own_score)
    other_score = other_score / np.max(other_score)

    # plot the stuff
    f, ax = plt.subplots(2, 1, figsize=(20, 10))
    ax[0].plot(raw_data)
    ax[0].set_title("time series")
    ax[1].plot(own_score, "r", label='Own Scoring')
    ax[1].plot(other_score, "bx", label='Other Scoring')
    ax[1].set_title("Change score comparison")
    ax[1].legend()


def generate_functions(idx):
    if idx == 1:
        # synthetic (frequency change)
        x0 = np.sin(2 * np.pi * 1 * np.linspace(0, 10, 1000))
        x1 = np.sin(2 * np.pi * 2 * np.linspace(0, 10, 1000))
        x2 = np.sin(2 * np.pi * 8 * np.linspace(0, 10, 1000))
        x3 = np.sin(2 * np.pi * 4 * np.linspace(0, 10, 1000))
        x = np.hstack([x0, x1, x2, x3])
        x += np.random.rand(x.size)
    elif idx == 2:
        # make synthetic step function
        x0 = 1 * np.ones(1000) + np.random.rand(1000) * 1
        x1 = 3 * np.ones(1000) + np.random.rand(1000) * 2
        x2 = 5 * np.ones(1000) + np.random.rand(1000) * 1.5
        x = np.hstack([x0, x1, x2])
        x += np.random.rand(x.size)
    elif idx == 3:
        # rising and falling edges
        x = np.hstack([np.linspace(0, 0.5, num=1000), 0.5 * np.ones(1000), np.linspace(0.5, -0.5, num=1000)])
    else:
        raise NotImplementedError
    return x


def qualitative_comparison():
    # get a function
    x = generate_functions(3)

    # compute change score using sst
    own_score = SingularSpectrumTransformation(window_length=60, lag=10, rank=5, method='svd').transform(x)
    # compute change score using other implementation
    # other_score = fastsst.SingularSpectrumTransformation(win_length=60, lag=10, n_components=5).score_offline(x)
    other_score = SingularSpectrumTransformation(window_length=60, lag=10, rank=5, method='rsvd').transform(x)
    plot_data_and_score(x, own_score, other_score)
    plt.show()


def speed_comparison():

    # measure the two methods for different timings
    window_candidates = np.logspace(1, 3, 5).astype(int)

    # generate a function
    x = generate_functions(2)

    # trigger the jit compilation so comparison is fair
    SingularSpectrumTransformation(10, method='ika').transform(x)
    SingularSpectrumTransformation(10, method='svd').transform(x)
    SingularSpectrumTransformation(10, method='rsvd').transform(x)

    def time_per_window_length(window_length, method):
        # create the SST object
        sst = SingularSpectrumTransformation(window_length, method=method)

        # compute the scores and add to the timing
        start = time.time()
        sst.transform(x)
        return time.time() - start

    methods = []
    measured_times = []
    window_lengths = []

    for method in ("ika", "rsvd"):
        for wl in window_candidates:
            print(wl, window_candidates)
            for j in range(10):
                measured_times.append(time_per_window_length(wl, method=method))
                methods.append(method)
                window_lengths.append(wl)

    df = pd.DataFrame({"time[s]": measured_times, "method": methods, "window length": window_lengths})
    sns.barplot(x="window length", y="time[s]", hue="method", data=df)
    plt.show()


if __name__ == '__main__':
    # qualitative_comparison()
    speed_comparison()
