import itertools
import typing

from changepoynt.simulation.base import SignalPartCollection

# for self type annotations
try:  # Python 3.11+
    from typing import Self
except ImportError:  # Python 3.9–3.10
    from typing_extensions import Self

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
        self.verbose = False

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
        if self.verbose and not (self.oscillation == other.oscillation and self.trend == other.trend):
            print()
            print('CMP Signal', self.oscillation == other.oscillation, self.trend == other.trend)
            print(self.trend)
            print(other.trend)
            print()
        return self.oscillation == other.oscillation and self.trend == other.trend

    def __str__(self):
        return f"{self.oscillation}\n{self.trend}\n{self.noise}"

    @classmethod
    def from_dict(cls, params: dict[str: base.SignalPart]) -> Self:
        return cls(**params)

    def to_json_dict(self) -> dict[str: typing.Any]:
        return {self.__class__.__name__:
                    {part_name: part.to_json_dict() for part_name, part in self.get_signal_parts().items()}
                }

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> Self:

        # check that we received the dict for a single signal
        parts_dict = cls.process_dict(parts_dict)

        # instantiate the parts using the common base class (where each class is registered)
        instantiated_parts_dict = {part_type: base.SignalPart.from_json_dict(part_dict)
                                   for part_type, part_dict in parts_dict.items()}
        return cls.from_dict(instantiated_parts_dict)


class ChangePoint(SignalPartCollection):

    def __init__(self, from_signal: Signal, to_signal: Signal,
                 oscillation_transition: base.BaseTransition,
                 trend_transition: base.BaseTransition,
                 global_position: typing.Optional[int] = None,
                 general_trend: typing.Optional[base.BaseTrend] = None,
                 general_noise: typing.Optional[base.BaseNoise] = None):
        super().__init__()

        # check the signals
        if not isinstance(from_signal, Signal):
            raise TypeError(f"{type(from_signal)=} needs to be a {Signal}.")
        if not isinstance(to_signal, Signal):
            raise TypeError(f"{type(to_signal)=} needs to be a {Signal}.")

        # check and handle the two transitions
        self.__handle_transitions(oscillation_transition, trend_transition, from_signal, to_signal)

        # if not global position is defined set it to the start of the to object (second signal)
        if global_position is None:
            global_position = trend_transition.from_object.shape[0]

        # check the type of the global position
        if not isinstance(global_position, int):
            raise TypeError(f"{type(global_position)=} must be {int}.")

        # check that we need to have a global position if general noise or trend are not none
        any_global_specified = (not general_trend is None or not general_noise is None)
        if any_global_specified and (global_position is None):
            raise ValueError(f"If you specify general_trend or general_noise, we also need a global position.")

        # check that the global position is within the general_trend or noise
        if general_trend is not None:
            if global_position >= general_trend.shape[0]:
                raise ValueError(f"{global_position=} has to be smaller than {general_trend.shape[0]=}.")
        if general_noise is not None:
            if global_position >= general_noise.shape[0]:
                raise ValueError(f"{global_position=} has to be smaller than {general_noise.shape[0]=}.")

        # save the signals
        self.from_signal = from_signal
        self.to_signal = to_signal
        self.global_position = global_position
        self.oscillation_transition = oscillation_transition
        self.trend_transition = trend_transition
        self.general_trend = general_trend
        self.general_noise = general_noise

    @staticmethod
    def __check_transition(transition: base.BaseTransition):
        # check whether the input is indeed a transition, and from and to object are registered
        if not isinstance(transition, base.BaseTransition):
            raise TypeError(f"{type(transition)=} must be {base.BaseTransition}.")

        # check that the transition has registered objects
        if not transition.has_registered_objects():
            raise ValueError(f"Input signal_transition must have registered objects.")

    def __handle_transitions(self, oscillation_transition, trend_transition, from_signal: Signal, to_signal: Signal):

        # check both the transitions
        self.__check_transition(oscillation_transition)
        self.__check_transition(trend_transition)

        # check the oscillation transition
        if oscillation_transition.from_object != from_signal.oscillation:
            raise ValueError(f'oscillation_transition.from_object does not match the from_signal.oscillation.')
        if oscillation_transition.to_object != to_signal.oscillation:
            raise ValueError(f'oscillation_transition.to_object does not match the to_signal.oscillation.')

        # check the trend transition
        if trend_transition.from_object != from_signal.trend:
            raise ValueError(f'trend_transition.from_object does not match the from_signal.trend.')
        if trend_transition.to_object != to_signal.trend:
            raise ValueError(f'trend_transition.to_object does not match the to_signal.trend.')

    def render(self) -> np.ndarray:

        # render the transition itself by creating a ChangeSignal from it
        signal = ChangeSignal([self.from_signal, self.to_signal],
                              oscillation_transition_list=[self.oscillation_transition],
                              trend_transition_list=[self.trend_transition])

        # render the signal
        rendered_array = signal.render()

        # add the general noise and trend
        start_idx = self.global_position-self.from_signal.shape[0]
        end_idx = self.global_position+self.to_signal.shape[0]
        general_noise = self.general_noise.render()[start_idx:end_idx]
        general_trend = self.general_trend.render()[start_idx:end_idx]
        return rendered_array + general_noise + general_trend

    def __eq__(self, other):

        # compare the signals
        from_signals_equal = self.from_signal == other.from_signal
        to_signals_equal = self.to_signal == other.to_signal
        signal_equal = from_signals_equal and to_signals_equal

        # compare the transitions
        trend_transition_equal = self.trend_transition == other.trend_transition
        oscillation_transition_equal = self.oscillation_transition == other.oscillation_transition
        transitions_equal = trend_transition_equal and oscillation_transition_equal

        # compare the general noise and trend
        general_noise_equal = self.general_noise == other.general_noise
        general_trend_equal = self.general_trend == other.general_trend
        general_equal = general_noise_equal and general_trend_equal

        # compare the global position
        position_equal = self.global_position == other.global_position
        return signal_equal and transitions_equal and general_equal and position_equal

    def to_json_dict(self) -> dict[str: typing.Any]:

        # copy the transitions so we do not alter our own transitions
        oscillation_transition = self.oscillation_transition.copy()
        trend_transition = self.trend_transition.copy()

        # deregister the signals from the copies so we do not save them multiple times
        oscillation_transition.deregister_objects()
        trend_transition.deregister_objects()

        # initialize the dict
        general_trend_json = self.general_trend.to_json_dict() if self.general_trend is not None else None
        general_noise_json = self.general_noise.to_json_dict() if self.general_noise is not None else None
        information_dict = {'from_signal': self.from_signal.to_json_dict(),
                            'to_signal': self.to_signal.to_json_dict(),
                            'oscillation_transition': oscillation_transition.to_json_dict(),
                            'trend_transition': trend_transition.to_json_dict(),
                            'global_position': self.global_position,
                            'general_trend': general_trend_json,
                            'general_noise': general_noise_json,}

        return {self.__class__.__name__: information_dict}

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> Self:

        # check that we received the dict for a single signal
        parts_dict = cls.process_dict(parts_dict)

        # instantiate the from and to signal
        from_signal = Signal.from_json_dict(parts_dict['from_signal'])
        to_signal = Signal.from_json_dict(parts_dict['to_signal'])

        # instantiate the transitions
        oscillation_transition = base.SignalPart.from_json_dict(parts_dict['oscillation_transition'])
        trend_transition = base.SignalPart.from_json_dict(parts_dict['trend_transition'])

        # register the from and to signals
        oscillation_transition.register_from_to_objects(from_signal.oscillation, to_signal.oscillation)
        trend_transition.register_from_to_objects(from_signal.trend, to_signal.trend)

        # instantiate the general noise and general trend
        if parts_dict['general_trend'] is not None:
            general_trend = base.SignalPart.from_json_dict(parts_dict['general_trend'])
        else:
            general_trend = None
        if parts_dict['general_noise'] is not None:
            general_noise = base.SignalPart.from_json_dict(parts_dict['general_noise'])
        else:
            general_noise = None

        # extract the global position
        global_position = parts_dict['global_position']

        # instantiate the parts using the common base class (where each class is registered)
        instantiated_parts_dict = {'from_signal' : from_signal,
                                   'to_signal' : to_signal,
                                   'oscillation_transition': oscillation_transition,
                                   'trend_transition': trend_transition,
                                   'global_position': global_position,
                                   'general_trend': general_trend,
                                   'general_noise': general_noise}
        return cls(**instantiated_parts_dict)

    def local_interval(self) -> tuple[int, int, int]:
        max_transition_length = max(self.oscillation_transition.transition_length, self.trend_transition.transition_length)
        local_position = self.from_signal.shape[0]
        return local_position-max_transition_length, local_position, local_position+max_transition_length

    def global_interval(self) -> tuple[int, int, int]:
        max_transition_length = max(self.oscillation_transition.transition_length, self.trend_transition.transition_length)
        return self.global_position-max_transition_length, self.global_position, self.global_position+max_transition_length


class ChangeSignal(base.SignalPartCollection):
    """
    This is the class for a Change signal. The change signal is always a concatenation of several Signals.
    """
    def __init__(self, signal_list: list[Signal],
                 oscillation_transition_list: list[typing.Union[base.BaseTransition, None]] = None,
                 trend_transition_list: list[typing.Union[base.BaseTransition, None]]= None,
                 general_trend: base.BaseTrend = None, general_noise: base.BaseNoise = None):
        super().__init__()
        self.verbose = False

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

        if self.verbose:
            for idx, (ele1, ele2) in enumerate(zip(self.signals, other.signals)):
                if ele1 != ele2:
                    print('FALLLLLLLLLLLLLLLLSE', idx)
                    print(ele1)
                    print(ele2)
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

    def get_change_points(self) -> list[ChangePoint,]:
        """
        The discrete changepoint is always an index to the first sample of the signal.
        This function returns the change points as an interval (with the transition length)
        :return: list of change point intervals in the form (transition start, changepoint, transition end)
        """

        # go through the signals and accumulate their length
        # if you would use this as an index, it would index the first sample of the new signal
        change_idx = list(itertools.accumulate(signal.shape[0] for signal in self.signals))

        # go through the transitions and the change points to adapt their intervals
        change_intervals = []
        for global_position, from_signal, to_signal, oscillation_transition, trend_transition in zip(change_idx, self.signals, self.signals[1:], self.oscillation_transitions, self.trend_transitions):
            change_intervals.append(ChangePoint(from_signal, to_signal, oscillation_transition, trend_transition, global_position, self.trend, self.noise))
        return change_intervals

    def to_json_dict(self) -> dict[str: typing.Any]:

        # initialize the dict
        result_dict = {'signal_list': [signal.to_json_dict() for signal in self.signals],
                       'oscillation_transition_list': [trans.to_json_dict() for trans in self.oscillation_transitions],
                       'trend_transition_list': [trans.to_json_dict() for trans in self.trend_transitions],
                       'general_trend': self.trend.to_json_dict(), 'general_noise': self.noise.to_json_dict()}

        # make the signal list
        return {self.__class__.__name__: result_dict}

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> Self:

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

    def _check_and_copy_other_signal(self, other_change_signal: Self):

        other_signal = other_change_signal.copy()

        # check whether the signal has no general noise and trend
        # otherwise concatenation makes no sense
        if self.noise != other_signal.noise or not isinstance(self.noise, noises.NoNoise):
            raise ValueError(f'Two signals with noises can not be concatenated. Only NoNoise is supported.')
        if self.trend != other_signal.trend:
            raise ValueError(f'Two signals with different trends can not be concatenated.')
        return other_signal


    def concatenate(self, other_change_signal: Self,
                    trend_transition: typing.Optional[base.BaseTransition] = None,
                    oscillation_transition: typing.Optional[base.BaseTransition] = None) -> Self:
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

    def extend(self, other_change_signal_list: list[Self],
               trend_transition_list: typing.Optional[list[base.BaseTransition]] = None,
               oscillation_transition_list: typing.Optional[list[base.BaseTransition]] = None) -> Self:
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
        self.verbose = False
        self.had_names = True

        # check whether all signals have the same shape, we already checked whether they are one-dimensional
        # when constructing the ChangeSignal
        if any(not isinstance(signal, ChangeSignal) for signal in signals):
            raise ValueError(f"All signals of a ChangeSignalMultivariate have to be of type ChangeSignal.")

        # check whether that the names have equal length to the signals if they are defined
        if signal_names is not None and len(signal_names) != len(signals):
            raise ValueError(f"Signals must have the same length as signal_names.")
        if signal_names is None:
            self.had_names = False
            signal_names = [f'Signal {idx}' for idx in range(len(signals))]

        # save the variables
        self.signals = signals
        self.signal_names = signal_names

    def render(self) -> np.ndarray:
        return np.stack([signal.render() for signal in self.signals], axis=0)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.signals), self.signals[0].shape[0]

    def to_json_dict(self) -> dict[str: typing.Any]:

        # initialize the dict
        result_dict = {'signals': [signal.to_json_dict() for signal in self.signals],
                       'signal_names': self.signal_names if self.had_names else None}

        # make the signal list
        return {self.__class__.__name__: result_dict}

    @classmethod
    def from_json_dict(cls, parts_dict: dict[str: typing.Any]) -> Self:

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

    def __eq__(self, other):

        # check the signals for equality
        signals_equal = all(sig1 == sig2 for sig1, sig2 in zip(self.signals, other.signals))
        names_equal = all(name1 == name2 for name1, name2 in zip(self.signal_names, other.signal_names))

        # verbose equality check
        if self.verbose:
            for idx, (name1, name2, signal1, signal2) in enumerate(zip(self.signal_names, other.signal_names, self.signals, other.signals)):
                if signal1 != signal2:
                    print(f'ChangeSignals at index {idx} are not equal.')
                    print(signal1)
                    print(signal2)
                if name1 != name2:
                    print(f'Names at index {idx} are not equal.')
                    print(name1, name2)
        return signals_equal and names_equal

    def to_array_dict(self):
        return {name: signal.render() for name, signal in zip(self.signal_names, self.signals)}

    def get_signals(self, idx: int) -> list[ChangeSignal]:
        return self.signals

    def get_names(self):
        return self.signal_names

    def get_change_points(self) -> list[list[ChangePoint]]:
        return [signal.get_change_points() for signal in self.signals]


if __name__ == "__main__":
    print('Parts')
    print("Grouped", base.SignalPart.get_registered_signal_parts_grouped())
    print("All", base.SignalPart.get_registered_signal_parts())
    print()