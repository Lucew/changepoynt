import typing

import numpy as np

from changepoynt.simulation import base
from changepoynt.simulation import noises
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends


class Signal:
    """
    This is the class for a Signal. The signal always consists of an Oscillation and Noise.
    """
    def __init__(self, oscillation: base.BaseOscillation, noise: base.BaseNoise = None, trend: base.BaseTrend = None):

        if len(oscillation.shape) != 1:
            raise AttributeError(f"Oscillation has to be a 1D array. Currently: {len(oscillation.shape)}.")

        # create the defaults if necessary
        if noise is None:
            noise = noises.NoNoise(oscillation.shape[0])
        if trend is None:
            trend = trends.ConstantOffset(oscillation.shape[0], offset=0.0)

        # check for the correct types
        if not isinstance(oscillation, base.BaseOscillation):
            raise ValueError(f"Oscillation has to be an instance of BaseOscillation. ")
        if not isinstance(noise, base.BaseNoise):
            raise ValueError(f"Noise has to be an instance of BaseNoise. ")
        if not isinstance(noise, base.BaseNoise):
            raise ValueError(f"Noise has to be an instance of BaseNoise. ")

        # check the shapes of the input parameters
        if oscillation.shape != noise.shape:
            raise ValueError(f"Shape of signal and noise must be the same. "
                             f"Currently: Signal shape {oscillation.shape} and Noise shape {noise.shape}.")
        if oscillation.shape != trend.shape:
            raise ValueError(f"Shape of signal and trend must be the same. "
                             f"Currently: Signal shape {oscillation.shape} and trend shape {noise.shape}.")
        if len(noise.shape) != 1 or len(trend.shape) != 1:
            raise ValueError(f"Trand and Noise have to have 1 dimension. "
                             f"Currently: Noise shape {noise.shape} and trend shape {trend.shape}.")

        # save the noise and the signal
        self.oscillation = oscillation
        self.noise = noise
        self.trend = trend

    def render(self) -> np.ndarray:
        return self.oscillation.render() + self.noise.render()

    @property
    def shape(self) -> tuple[int,]:
        # we have tested noise and signal to be the same shape when constructing this class
        return self.oscillation.shape

    def __eq__(self, other: "Signal") -> bool:

        # check that we are only compared to another Signal
        if not isinstance(other, type(self)):
            raise TypeError(f"Signals can only be compared to other Signals.")

        # compare the two oscillations for equality, they are equal if trend and oscillation are the same
        return self.oscillation == other.oscillation and self.trend == other.trend

    @staticmethod
    def translate_length(length: typing.Union[int, "Signal"]) -> int:
        if isinstance(length, int):
            return length
        elif isinstance(length, Signal):
            return length.shape[0]
        else:
            raise ValueError(f"Length must be of class Signal or Tuple. Not: {type(length)}.")


class ChangeSignal:
    """
    This is the class for a Change signal. The change signal is always a concatenation of several Signals.
    """
    def __init__(self, signals: list[Signal], trend: base.BaseTrend = None, noise: base.BaseNoise = None):

        # check whether all signals have a dimension of one
        if any(len(signal.shape) != 1 for signal in signals):
            raise ValueError(f"Signal must have 1 dimension.")

        # check whether all signals are of type Signal
        if any(not isinstance(signal, Signal) for signal in signals):
            raise ValueError(f"All Signals within a ChangeSignal have to be of type Signal.")

        # save the variables
        self.signals = signals

        # check for the trend (after we save the signals, as we need their shape to check)
        if trend is None:
            trend = trends.ConstantOffset(self.shape[0], offset=0.0)
        else:
            if trend.shape != self.shape:
                raise ValueError(f"Shape of trend and change signal must be the same. Change Signal: {self.shape},"
                                 f"Trend: {trend.shape}.")
        self.trend = trend

        # check for the noise (after we save the signals, as we need their shape to check)
        if noise is None:
            noise = noises.NoNoise(self.shape[0])
        else:
            if noise.shape != self.shape:
                raise ValueError(f"Shape of trend and change signal must be the same. Change Signal: {self.shape},"
                                 f"Trend: {noise.shape}.")
        self.noise = noise

        # infer the change points by using the shapes, by definition the change points
        # are always the first sample after a signal has ended
        #
        # we also check whether two signals are different so the change points are guaranteed
        self.change_points_list = []
        for idx, (signal1, signal2) in enumerate(zip(self.signals, self.signals[1:])):

            # check that we actually have a change point
            if signal1 == signal2:  # no change
                raise ValueError(f"signals[{idx}] and signals[{idx+1}] are similar and show no change.")
            self.change_points_list.append(signal1.shape[0]+1)


    def render(self):
        return (np.concatenate([signal.render() for signal in self.signals], axis=0)
                + self.trend.render()
                + self.noise.render())

    @property
    def shape(self) -> tuple[int,]:
        # we already tested that all signal are one-dimensional in the constructor
        return (sum(signal.shape[0] for signal in self.signals), )

    @property
    def change_points(self) -> list[int]:
        return self.change_points_list


class ChangeSignalMultivariate:
    """
    This is the class for a Change system. The system has multiple Change signals stacked along the zeroth axis.
    """

    def __init__(self, signals: list[ChangeSignal], signal_names:list[str] = None):

        # check whether all signals have the same shape, we already checked whether they are one-dimensional
        # when constructing the ChangeSignal
        if any(not isinstance(signal, ChangeSignal) for signal in signals):
            raise ValueError(f"All signals of a ChangeSignalMultivariate have to be of type ChangeSignal.")

        # check whether that the names have equal length to the signals if they are defined
        if signal_names is not None and len(signal_names) != len(signals):
            raise ValueError(f"Signals must have the same length as signal_names. ")

        # save the variables
        self.signals = signals
        self.signal_names = signal_names

        # get the change points
        self.change_points_list = [signal.change_points for signal in signals]

    def render(self) -> np.ndarray:
        return np.stack([signal.render() for signal in self.signals], axis=0)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.signals), self.signals[0].shape[0]

    @property
    def change_points(self) -> list[list[int]]:
        return self.change_points_list

if __name__ == "__main__":
    print('Parts')
    print("Grouped", base.SignalPart.get_registered_signal_parts_grouped())
    print("All", base.SignalPart.get_registered_signal_parts())
    print()