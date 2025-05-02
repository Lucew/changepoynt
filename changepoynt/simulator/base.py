import typing
import abc

import numpy as np


class BaseOscillation:
    """
    This class builds the base for an Oscillation, which we see as a base signal. For example, this can be constant,
    a sine, or a triangle function. Every Oscillation has to have a render function to be called when compiling the
    final signal.
    """

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        return np.ndarray([])

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:
        return ()

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class BaseTrend:
    """
    This class builds the base for a Trend. For example, this can be constant, a line with a slope. The trend
    is always additive.
    """

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        return np.ndarray([])

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:
        return ()

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class NoTrend(BaseTrend):
    """
    A class that is the default trend: No trend, adding an offset of zero.
    """
    def __init__(self, shape: tuple[int]):

        # save the intended shape
        self.shape_var = shape
        self.offset = 0.0

    def render(self, *args, **kwargs) -> float:
        return self.offset

    @property
    def shape(self) -> tuple:
        return self.shape_var

    def __eq__(self, other: object) -> bool:

        # check that we are compared to another BaseTrend
        if not isinstance(other, BaseTrend):
            raise TypeError(f'Cannot compare {type(other)} with {type(self)}.')

        # check whether the other is a different trend then no equality
        # if other is NoTrend, we have equality
        return isinstance(other, type(self))



class BaseNoise:
    """
    This class builds the base for a Noise signal. Noise will be always additive and is assigned to every
    Signal. It has to have a render function to be called when compiling the signal.
    """
    def __init__(self, render_function: typing.Callable):
        self.render_function = render_function

    def render(self, *args, **kwargs) -> np.ndarray:
        return self.render_function(*args, **kwargs)

    @property
    @abc.abstractmethod
    def shape(self) -> tuple:
        return ()


class NoNoise:
    """
    This class is the default if no noise is specified. It adds no noise to the signal.
    """

    def __init__(self, shape: tuple[int]):
        # save the intended shape
        self.shape_var = shape
        self.offset = 0.0

    def render(self, *args, **kwargs) -> float:
        return self.offset

    @property
    def shape(self) -> tuple:
        return self.shape_var

    def __eq__(self, other: object) -> bool:
        # check that we are compared to another BaseTrend
        if not isinstance(other, BaseNoise):
            raise TypeError(f'Cannot compare {type(other)} with {type(self)}.')

        # check whether the other is a different trend then no equality
        # if other is NoTrend, we have equality
        return isinstance(other, type(self))


class Signal:
    """
    This is the class for a Signal. The signal always consists of an Oscillation and Noise.
    """
    def __init__(self, oscillation: BaseOscillation, noise: BaseNoise = None, trend: BaseTrend = None):

        # create the defaults if necessary
        if noise is None:
            noise = NoNoise(oscillation.shape)
        if trend is None:
            trend = NoTrend(oscillation.shape)

        # check for the correct types
        if not isinstance(oscillation, BaseOscillation):
            raise ValueError(f"Oscillation has to be an instance of BaseOscillation. ")
        if not isinstance(noise, BaseNoise):
            raise ValueError(f"Noise has to be an instance of BaseNoise. ")
        if not isinstance(noise, BaseNoise):
            raise ValueError(f"Noise has to be an instance of BaseNoise. ")

        # check the shapes of the input parameters
        if oscillation.shape != noise.shape:
            raise ValueError(f"Shape of signal and noise must be the same. "
                             f"Currently: Signal shape {oscillation.shape} and Noise shape {noise.shape}.")
        if oscillation.shape != trend.shape:
            raise ValueError(f"Shape of signal and trend must be the same. "
                             f"Currently: Signal shape {oscillation.shape} and trend shape {noise.shape}.")
        if len(oscillation.shape) != 1 or len(noise.shape) != 1 or len(trend.shape) != 1:
            raise ValueError(f"Signal and Noise have to have 1 dimension. "
                             f"Currently: Oscillation shape {len(oscillation.shape)},  Noise shape {noise.shape}, "
                             f"and trend shape {trend.shape}.")

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


class ChangeSignal:
    """
    This is the class for a Change signal. The change signal is always a concatenation of several Signals.
    """
    def __init__(self, signals: list[Signal]):

        # check whether all signals have a dimension of one
        if any(len(signal.shape) != 1 for signal in signals):
            raise ValueError(f"Signal must have 1 dimension.")

        # check whether all signals are of type Signal
        if any(not isinstance(signal, Signal) for signal in signals):
            raise ValueError(f"All Signals within a ChangeSignal have to be of type Signal.")

        # save the variables
        self.signals = signals

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
        return np.concatenate([signal.render() for signal in self.signals], axis=0)

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
