import typing
import itertools
import json

import numpy as np

from changepoynt.simulation import base
from changepoynt.simulation import noises
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends
from changepoynt.simulation import transitions


class Signal(base.SignalPartCollection):
    """
    This is the class for a Signal. The signal always consists of an Oscillation and Noise.
    """
    def __init__(self, oscillation: base.BaseOscillation, noise: base.BaseNoise = None, trend: base.BaseTrend = None):
        super().__init__()

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
                'noise': self.noise.get_parameters_for_randomizations_values()}

    def render(self) -> np.ndarray:
        return self.parts_to_signal(self.render_parts())

    def get_signal_parts(self):
        return {'oscillation': self.oscillation, 'trend': self.trend, 'noise': self.noise}

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

    def __str__(self):
        return f"{self.oscillation}\n{self.trend}\n{self.noise}"

    @classmethod
    def from_dict(cls, params: dict[str: base.SignalPart]) -> typing.Self:
        return cls(**params)

    def to_json_dict(self) -> dict[str: typing.Any]:
        return {self.__class__.__name__:
                    {part_name: part.to_json_dict() for part_name, part in self.get_signal_parts().items()}
                }

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> typing.Self:

        # check that we received the dict for a single signal
        parts_dict = cls.process_dict(parts_dict)

        # instantiate the parts using the common base class (where each class is registered)
        instantiated_parts_dict = {part_type: base.SignalPart.from_json_dict(part_dict)
                                   for part_type, part_dict in parts_dict.items()}
        return cls.from_dict(instantiated_parts_dict)


class ChangeSignal(base.SignalPartCollection):
    """
    This is the class for a Change signal. The change signal is always a concatenation of several Signals.
    """
    def __init__(self, signal_list: list[Signal],
                 oscillation_transition_list: list[typing.Union[base.BaseTransition, None]] = None,
                 trend_transition_list: list[typing.Union[base.BaseTransition, None]]= None,
                 general_trend: base.BaseTrend = None, general_noise: base.BaseNoise = None):
        super().__init__()

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
            try:
                oscillation_transition.apply(signal_from['oscillation'], signal_to['oscillation'])
            except ValueError as e:
                print(f'We stopped {self.__class__.__name__} rendering at oscillation transition {idx} (type {oscillation_transition.__class__.__name__}) because of an error.')
                raise e
            try:
                trend_transition.apply(signal_from['trend'], signal_to['trend'])
            except ValueError as e:
                print(f'We stopped {self.__class__.__name__} rendering at trend transition {idx} (type {trend_transition.__class__.__name__}) because of an error.')
                raise e

        # sum the signal parts for a complete signals
        rendered_signals = [Signal.parts_to_signal(signal_parts) for signal_parts in rendered_signals]

        # concatenate the signals and add the general trend and noise
        return np.concatenate(rendered_signals, axis=0) + self.trend.render() + self.noise.render()

    @property
    def shape(self) -> tuple[int,]:
        # we already tested that all signal are one-dimensional in the constructor
        return (sum(signal.shape[0] for signal in self.signals), )

    def __eq__(self, other: "ChangeSignal") -> bool:

        # check that we are only compared to another Signal
        if not isinstance(other, type(self)):
            raise TypeError(f"Signals can only be compared to other Signals.")

        # compare the two oscillations for equality, they are equal if trend and oscillation are the same
        signal_equal = all(ele1 == ele2 for ele1, ele2 in zip(self.signals, other.signals))
        transitions_equal = all(ele1 == ele2 for ele1, ele2 in zip(self.trend_transitions, other.trend_transitions))
        noise_equal = self.noise == other.noise
        trend_equal = self.trend == other.trend
        return signal_equal and transitions_equal and noise_equal and trend_equal

    def __str__(self):

        # go through the signals and print them
        output = []
        for idx, signal in enumerate(self.signals):
            if idx&1:
                output.append(f'Transition {idx-1}:')
                output.append(str(self.trend_transitions[idx-1]))
                output.append(str(self.oscillation_transitions[idx-1]))
                output.append('')
            output.append(f'Signal {idx}:')
            output.append(str(signal))
            output.append('')
        return '\n'.join(output)

    @property
    def changepoints(self) -> list[int]:

        # go through the signals and collect their length
        return list(itertools.accumulate(signal.shape[0] for signal in self.signals))

    def to_json_dict(self) -> dict[str: typing.Any]:

        # initialize the dict
        result_dict = {'signal_list': [signal.to_json_dict() for signal in self.signals],
                       'oscillation_transition_list': [trans.to_json_dict() for trans in self.oscillation_transitions],
                       'trend_transition_list': [trans.to_json_dict() for trans in self.trend_transitions],
                       'general_trend': self.trend.to_json_dict(), 'general_noise': self.noise.to_json_dict()}

        # make the signal list
        return {self.__class__.__name__: result_dict}

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> typing.Self:

        # check that we received the dict for a single signal
        parts_dict = cls.process_dict(parts_dict)

        # get the list of things we can construct
        instantiated_parts_dict = {}
        for key_name, value_object in parts_dict.items():
            if isinstance(value_object, list):
                if key_name == 'signal_list':
                    instantiated_parts_dict[key_name] = [Signal.from_json_dict(list_obj) for list_obj in value_object]
                else:
                    instantiated_parts_dict[key_name] = [base.SignalPart.from_json_dict(list_obj)
                                                         if list_obj is not None else list_obj
                                                         for list_obj in value_object]
            else:
                instantiated_parts_dict[key_name] = base.SignalPart.from_json_dict(value_object) if value_object is not None else value_object


        # check the length of the lists
        expected_length = len(instantiated_parts_dict['signal_list'])-1
        oscillation_transition_length = len(instantiated_parts_dict['oscillation_transition_list'])
        trend_transition_length = len(instantiated_parts_dict['trend_transition_list'])
        if oscillation_transition_length != expected_length:
            raise ValueError(f'The serialized transition list needs {expected_length} elements but has {oscillation_transition_length} elements.')
        if trend_transition_length != expected_length:
            raise ValueError(f'The serialized transition list needs {expected_length} elements but has {trend_transition_length} elements.')

        # register the signals in the transitions
        element_iterator = zip(instantiated_parts_dict['signal_list'],
                               instantiated_parts_dict['signal_list'][1:],
                               instantiated_parts_dict['oscillation_transition_list'],
                               instantiated_parts_dict['trend_transition_list']
                               )
        for signal1, signal2, oscillation_transition, trend_transition in element_iterator:
            oscillation_transition.register_from_to_objects(signal1.oscillation, signal2.oscillation)
            trend_transition.register_from_to_objects(signal1.trend, signal2.trend)

        return cls(**instantiated_parts_dict)

    def _check_and_copy_other_signal(self, other_change_signal: typing.Self):

        other_signal = other_change_signal.copy()

        # check whether the signal has no general noise and trend
        # otherwise concatenation makes no sense
        if self.noise != other_signal.noise or not isinstance(self.noise, noises.NoNoise):
            raise ValueError(f'Two signals with noises can not be concatenated. Only NoNoise is supported.')
        if self.trend != other_signal.trend:
            raise ValueError(f'Two signals with different trends can not be concatenated.')
        return other_signal


    def concatenate(self, other_change_signal: typing.Self,
                    trend_transition: typing.Optional[base.BaseTransition] = None,
                    oscillation_transition: typing.Optional[base.BaseTransition] = None) -> typing.Self:
        """
        This function concatenates a change signal to another. It makes a copy of both.
        :param other_change_signal: the other signal to concatenate
        :param trend_transition: a trend transition between the concatenated signals
        :param oscillation_transition: an oscillation transition between the concatenated signals
        :return: The newly concatenated change signal.
        """

        # use our own "extend" function with a one element list
        return self.extend([other_change_signal],
                           trend_transition_list=[trend_transition],
                           oscillation_transition_list=[oscillation_transition])

    def extend(self, other_change_signal_list: list[typing.Self],
               trend_transition_list: typing.Optional[list[base.BaseTransition]] = None,
               oscillation_transition_list: typing.Optional[list[base.BaseTransition]] = None) -> typing.Self:
        """
        This function concatenates a list change signal to another. It makes a copy of all of them.
        :param other_change_signal_list: a list of objects of class ChangeSignal
        :param trend_transition_list: a trend transition between the concatenated signals
        :param oscillation_transition_list: an oscillation transition between the concatenated signals
        :return: The newly concatenated change signal.
        """
        # make a copy of all the signals (including the self signal)
        other_signal_list = [self.copy()]
        other_signal_list.extend(self._check_and_copy_other_signal(other_signal)
                                 for other_signal in other_change_signal_list)

        # check whether we received a list of oscillation transitions
        if oscillation_transition_list is None:
            oscillation_transition_list = [None]*len(other_signal_list)
        else:
            if len(oscillation_transition_list) != len(other_signal_list)-1:
                raise ValueError(f'There are not enough oscillation transitions. There are {len(oscillation_transition_list)} oscillation transitions but {len(other_signal_list)-1} signals to extend.')

        # check whether we received a list of trend transitions
        if trend_transition_list is None:
            trend_transition_list = [None]*len(other_signal_list)
        else:
            if len(trend_transition_list) != len(other_signal_list)-1:
                raise ValueError(f'There are not enough trend transitions. There are {len(oscillation_transition_list)} trend transitions but {len(other_signal_list)-1} signals to extend.')

        for idx, (prev_signal, curr_signal, oscillation_transition, trend_transition) in enumerate(zip(other_signal_list, other_signal_list[1:], oscillation_transition_list, trend_transition_list)):

            # add a transition between the two oscillations
            if oscillation_transition is None:
                # default is no transition
                oscillation_transition = transitions.NoTransition(prev_signal.signals[-1].oscillation,
                                                                  curr_signal.signals[0].oscillation)
            elif isinstance(oscillation_transition, base.BaseTransition):
                oscillation_transition.register_from_to_objects(prev_signal.signals[-1].oscillation,
                                                                curr_signal.signals[0].oscillation)
            else:
                raise ValueError(f'oscillation_transition_list[{idx=}] must be of type BaseTransition. Currently {type(oscillation_transition)}.')
            # append the oscillation transition to the end of the previous transition
            prev_signal.oscillation_transitions.append(oscillation_transition)

            # add a transition between the two trends
            if trend_transition is None:
                # default is no transition
                trend_transition = transitions.NoTransition(prev_signal.signals[-1].trend, curr_signal.signals[0].trend)
            elif isinstance(trend_transition, base.BaseTransition):
                trend_transition.register_from_to_objects(prev_signal.signals[-1].trend, curr_signal.signals[0].trend)
            else:
                raise ValueError(f'trend_transition_list[{idx=}] must be of type BaseTransition. Currently {type(trend_transition)}.')
            # append the trend transition to the end of the previous transition
            prev_signal.trend_transitions.append(trend_transition)

        # take the first signal as the sentinel/seed/start
        first_signal = other_signal_list[0]

        # extend the transitions
        first_signal.trend_transitions.extend(__trend_transition for __sig in other_signal_list[1:] for __trend_transition in __sig.trend_transitions)
        first_signal.oscillation_transitions.extend(__oscillation_transition for __sig in other_signal_list[1:] for __oscillation_transition in __sig.oscillation_transitions)

        # extend the size of the trend and of the noise
        first_signal.trend.length = first_signal.trend.length + sum(other_signal.trend.length for other_signal in other_signal_list[1:])
        first_signal.noise.length = first_signal.noise.length + sum(other_signal.noise.length for other_signal in other_signal_list[1:])

        # extend the signal list
        first_signal.signals.extend((__sig for __other_change_signal in other_signal_list[1:] for __sig in __other_change_signal.signals))
        return first_signal



class ChangeSignalMultivariate(base.SignalPartCollection):
    """
    This is the class for a Change system. The system has multiple Change signals stacked along the zeroth axis.
    """

    def __init__(self, signals: list[ChangeSignal], signal_names:list[str] = None):
        super().__init__()

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

    def to_json_dict(self) -> dict[str: typing.Any]:

        # initialize the dict
        result_dict = {'signals': [signal.to_json_dict() for signal in self.signals],
                       'signal_names': self.signal_names}

        # make the signal list
        return {self.__class__.__name__: result_dict}

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> typing.Self:

        # check that we received the dict for a single signal
        parts_dict = cls.process_dict(parts_dict)

        # get the list of things we can construct
        instantiated_parts_dict = {}
        for key, value in parts_dict.items():
            if key == 'signals':
                instantiated_parts_dict[key] = [ChangeSignal.from_json_dict(signal) for signal in value]
            else:
                instantiated_parts_dict[key] = value
        return cls(**instantiated_parts_dict)

    @property
    def changepoints(self) -> list[list[int]]:
        return self.change_points_list


if __name__ == "__main__":
    print('Parts')
    print("Grouped", base.SignalPart.get_registered_signal_parts_grouped())
    print("All", base.SignalPart.get_registered_signal_parts())
    print()