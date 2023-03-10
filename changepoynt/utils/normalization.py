# this file contains utility functions to scale and normalize signals
import numpy as np


def min_max_scaling(time_series: np.ndarray, min_val: float = 0.0, max_val: float = 1.0, inplace: bool = False)\
        -> np.ndarray:
    """
    This function applied min max scaling for an 1D-array. It is inspired by:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    but lighter and reimplemented in order to not introduce unnecessary dependencies for small functionality.

    Min max scaling as implemented here is sensitive to extreme outliers, but guarantees the value range not
    exceeding min_val and max_val.

    :param time_series: 1D array containing consecutive values for one feature
    :param min_val: the minimum value the scaled time series will reach
    :param max_val: the maximum value the scale time series will reach
    :param inplace: boolean to specify whether the input array will be scaled and changed in place

    :return: the scaled input array.
    """
    # make some assertion checks
    assert time_series.ndim == 1, 'Time series needs to be an 1D array.'

    # copy the time series if specified
    if not inplace: time_series = time_series.copy()

    # scale the time series to values between zero and one
    time_series = (time_series-np.min(time_series, axis=0)) / (np.max(time_series, axis=0)-np.min(time_series, axis=0))

    # scale into the wished value range
    time_series = time_series * (max_val - min_val) + min_val
    return time_series
