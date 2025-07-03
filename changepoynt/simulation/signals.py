import typing
import itertools

import numpy as np

from changepoynt.simulation import base
from changepoynt.simulation import noises
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends
from changepoynt.simulation import transitions


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

    def render_parts(self) -> dict[str: tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return {'oscillation': self.oscillation.render(), 'trend': self.trend.render(), 'noise': self.noise.render()}

    @staticmethod
    def parts_to_signal(signal_parts: dict[str: tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
        return sum(signal_parts.values())

    def get_randomizeable_parameter_values(self) -> dict[str: dict[str: typing.Any]]:
        return {'oscillation': self.oscillation.get_parameters_for_randomizations_values(),
                'trend': self.trend.get_parameters_for_randomizations_values(),
                'noise': self.trend.get_parameters_for_randomizations_values()}

    def render(self) -> np.ndarray:
        return self.parts_to_signal(self.render_parts())

    def get_signal_parts(self):
        return {'oscillation': self.oscillation, 'trend': self.trend, 'noise': self.noise}

    @property
    def shape(self) -> tuple[int,]:
        # we have tested noise and signal to be the same shape when constructing this class
        return self.oscillation.shape

    @classmethod
    def from_dict(cls, params: dict[str: base.SignalPart]) -> 'Signal':
        return cls(**params)

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
    def __init__(self, signal_list: list[Signal],
                 oscillation_transition_list: list[typing.Union[base.BaseTransition, None]] = None,
                 trend_transition_list: list[typing.Union[base.BaseTransition, None]]= None,
                 general_trend: base.BaseTrend = None, general_noise: base.BaseNoise = None):

        # make default value for the transition_lists
        if oscillation_transition_list is None:
            oscillation_transition_list = [None] * (len(signal_list)-1)
        if trend_transition_list is None:
            trend_transition_list = [None] * (len(signal_list)-1)

        # check for the length of the transition lists
        if len(oscillation_transition_list) != len(signal_list)-1:
            raise ValueError(f"The length of 'oscillation_transition_list' ({len(oscillation_transition_list)}) must be one smaller than the length of 'signal_list' ({len(signal_list)}). ")
        if len(trend_transition_list) != len(signal_list)-1:
            raise ValueError(f"The length of 'trend_transition_list' ({len(oscillation_transition_list)}) must be one smaller than the length of 'signal_list' ({len(signal_list)}). ")

        # go through both transition lists, update default Nones and
        for idx, (trend_transition, oscillation_transition, from_signal, to_signal) in enumerate(zip(trend_transition_list, oscillation_transition_list, signal_list, signal_list[1:])):

            # create the defaults
            if oscillation_transition is None:
                oscillation_transition_list[idx] = transitions.NoTransition(from_signal.oscillation, to_signal.oscillation)
                oscillation_transition = oscillation_transition_list[idx]
            if trend_transition is None:
                trend_transition_list[idx] = transitions.NoTransition(from_signal.trend, to_signal.trend)
                trend_transition = trend_transition_list[idx]

            # check for non defaults whether the connected signals are equal
            if not oscillation_transition.check_objects(from_signal.oscillation, to_signal.oscillation):
                raise ValueError(f"'oscillation_transition[{idx}]' does not connect the two signal oscillations 'signal_list[{idx}].oscillation' and 'signal_list[{idx+1}].oscillation' from the 'signal_list'.")
            if not trend_transition.check_objects(from_signal.trend, to_signal.trend):
                raise ValueError(f"'oscillation_transition[{idx}]' does not connect the two signal trends 'signal_list[{idx}].trend' and 'signal_list[{idx+1}].trend' from the 'signal_list'.")

            # check that the transitions actually are transitions
            if not isinstance(oscillation_transition, base.BaseTransition):
                raise TypeError(f'oscillation_transition[{idx}] is not a BaseTransition.')
            if not isinstance(trend_transition, base.BaseTransition):
                raise TypeError(f'trend_transition[{idx}] is not a BaseTransition.')

        # save the transitions
        self.oscillation_transitions: list[base.BaseTransition] = oscillation_transition_list
        self.trend_transitions: list[base.BaseTransition] = trend_transition_list

        # save the signals
        if not all(isinstance(signal, Signal) for signal in signal_list):
            raise TypeError('All signals must be of type Signal.')
        self.signals = signal_list

        # check for the trend (after we save the signals, as we need their shape to check)
        if general_trend is None:
            general_trend = trends.ConstantOffset(self.shape[0], offset=0.0)
        else:
            if general_trend.shape != self.shape:
                raise ValueError(f"Shape of trend and change signal must be the same. Change Signal: {self.shape},"
                                 f"Trend: {general_trend.shape}.")
        self.trend = general_trend

        # check for the noise (after we save the signals, as we need their shape to check)
        if general_noise is None:
            general_noise = noises.NoNoise(self.shape[0])
        else:
            if general_noise.shape != self.shape:
                raise ValueError(f"Shape of trend and change signal must be the same. Change Signal: {self.shape},"
                                 f"Trend: {general_noise.shape}.")
        self.noise = general_noise

    def render(self):

        # create the list of signals
        rendered_signals = [signal.render_parts() for signal in self.signals]

        # apply the trend and oscillation transitions
        for idx, (signal_from, signal_to, oscillation_transition, trend_transition) in enumerate(zip(rendered_signals, rendered_signals[1:], self.oscillation_transitions, self.trend_transitions)):
            oscillation_transition.apply(signal_from['oscillation'], signal_to['oscillation'])
            trend_transition.apply(signal_from['trend'], signal_to['trend'])

        # sum the signal parts for a complete signals
        rendered_signals = [Signal.parts_to_signal(signal_parts) for signal_parts in rendered_signals]

        # concatenate the signals and add the general trend and noise
        return np.concatenate(rendered_signals, axis=0) + self.trend.render() + self.noise.render()

    @property
    def shape(self) -> tuple[int,]:
        # we already tested that all signal are one-dimensional in the constructor
        return (sum(signal.shape[0] for signal in self.signals), )

    @property
    def changepoints(self) -> list[int]:

        # go through the signals and collect their length
        return list(itertools.accumulate(signal.shape[0] for signal in self.signals))


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
        self.change_points_list = [signal.changepoints for signal in signals]

    def render(self) -> np.ndarray:
        return np.stack([signal.render() for signal in self.signals], axis=0)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.signals), self.signals[0].shape[0]

    @property
    def changepoints(self) -> list[list[int]]:
        return self.change_points_list


if __name__ == "__main__":
    print('Parts')
    print("Grouped", base.SignalPart.get_registered_signal_parts_grouped())
    print("All", base.SignalPart.get_registered_signal_parts())
    print()