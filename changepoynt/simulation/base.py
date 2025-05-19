import typing
import abc
import warnings

import numpy as np

# TODO: enable custom limit error messages when registering a parameter
# TODO: somehow test that all children of SignalPart which also have a render function can be initialized without specifying values
# TODO: test that
class SignalPart:
    """
    This class is the highest level interface for all elements that will end up in a change signal. It takes care of
    checking parameters and keeping track of modifiable parameters.

    The modifiable parameters are important as they will be used to randomly assign changes.
    """
    def __init__(self, length: int, minimum_length: int = 100):

        # create a dictionary, where we save limits of the variables
        self.__parameter_limits = dict()
        self.__parameter_tolerances = dict()

        # create a dictionary for deactivated modifiable parameters
        self.__modifiable_parameters = set()

        # create a dictionary for the parameter that have to be used for comparison
        self.__comparison_parameters = set()

        # create a dictionary for the limit error explanation
        self.__limit_error_explanations = dict()

        # save the length
        length = Signal.translate_length(length)
        if length < minimum_length:
            raise ValueError(f'Length is too short: {length} < {minimum_length}.')
        self.length = length

    @property
    def shape(self) -> tuple[int,]:
        return (self.length,)

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        pass

    def check_and_get_parameter(self, parameter_name: str):
        # get the value from the object
        value = getattr(self, parameter_name, None)
        if value is None:
            raise AttributeError(f"No parameter {parameter_name} defined in the current object.")
        if not (isinstance(value, int) or isinstance(value, float)):
            raise TypeError(f"Parameter {parameter_name} must be a number.")
        return value

    def check_and_get_parameter_limit(self, parameter_name: str):
        if parameter_name in self.__parameter_limits:
            return self.__parameter_limits[parameter_name]
        else:
            raise AttributeError(f"parameter {parameter_name} has no defined limit in the current object.")

    def check_and_get_parameter_tolerance(self, parameter_name: str):
        if parameter_name in self.__parameter_tolerances:
            return self.__parameter_tolerances[parameter_name]
        else:
            raise AttributeError(f"parameter {parameter_name} has no defined tolerance in the current object.")

    def get_parameter(self, parameter_name: str):
        return self.check_and_get_parameter(parameter_name)

    def get_limit(self, parameter_name: str):
        return self.check_and_get_parameter_limit(parameter_name)

    def get_tolerance(self, parameter_name: str):
        return self.check_and_get_parameter_tolerance(parameter_name)

    def is_modifiable(self, parameter_name: str):
        if parameter_name not in self.__modifiable_parameters:
            raise AttributeError(f"Parameter {parameter_name} is not modifiable.")
        return True

    def get_modifiable_limit(self, parameter_name: str):
        self.is_modifiable(parameter_name)
        return self.get_limit(parameter_name)

    def get_modifiable_tolerance(self, parameter_name: str):
        self.is_modifiable(parameter_name)
        return self.get_tolerance(parameter_name)

    @property
    def modifiable_parameters(self) -> list[str,]:
        return list(self.__modifiable_parameters)

    @property
    def parameters(self) -> list[str,]:
        return list(self.__parameter_limits.keys())

    def deactivate_modifiable_parameters(self, parameter_name: str):
        self.check_and_get_parameter(parameter_name)
        self.__modifiable_parameters.discard(parameter_name)

    def activate_modifiable_parameters(self, parameter_name: str):
        self.check_and_get_parameter(parameter_name)
        self.__modifiable_parameters.add(parameter_name)

    def _register_parameter(self, parameter_name: str,
                            limit: tuple[typing.Union[float, int], typing.Union[float, int]] = None,
                            tolerance: float = None, modifiable: bool = False, used_for_comparison: bool = True,
                            limit_error_explanation: str = ""):
        """
        This function allows to register parameters of a signal part. The parameters are used for:
        - Comparisons when checking for equality
        - Modifiable parameters can be used to randomly generate changes

        :param parameter_name: the name of the parameter. has to be the same name as defined in the signal part
        :param limit: a minimum and maximum value (inclusive). default: (-inf, inf)
        :param tolerance: the maximum absolute difference that still counts as equality between parameters. default: inf
        :param modifiable: whether the parameter is can be modified to generate random signals. default: False
        :param used_for_comparison: whether the parameter is used as a comparison. default: True
        :param limit_error_explanation: optional explanation why a parameter has a limit. default: ""
        :return: None
        """
        # check for existence of the parameter -------------------------------------------------------------------------

        # get the current parameter value (and check that it exists)
        self.check_and_get_parameter(parameter_name)

        # check that we not already have this parameter registered
        if parameter_name in self.__parameter_limits:
            warnings.warn(f"Parameter {parameter_name} has already been registered in the current object (lim).")

        # check whether the parameter is modifiable --------------------------------------------------------------------
        if modifiable:
            # register parameter as modifiable
            self.__modifiable_parameters.add(parameter_name)

        # take care of the limit ---------------------------------------------------------------------------------------

        # make the default value
        if limit is None:
            limit = (-float('inf'), float('inf'))

        # check that the limit is valid
        if limit[0] > limit[1]:
            raise AttributeError(f"Limit has to from small to big. Current limit: {limit}.")

        # register the limit for the parameter
        self.__parameter_limits[parameter_name] = limit

        # check the value for the limit
        self.check_value_in_limits(parameter_name)

        # save the explanations
        self.__limit_error_explanations[parameter_name] = limit_error_explanation

        # take care of the tolerance -----------------------------------------------------------------------------------

        # make the default value
        if tolerance is None:
            tolerance = float('inf')

        # check that we do not have a tolerance for this parameter
        if parameter_name in self.__parameter_tolerances:
            warnings.warn(f"Parameter {parameter_name} has already been registered in the current object (tol).")

        # check that the tolerance is valid
        if tolerance < 0:
            raise AttributeError(f"Tolerance hast to be >= 0. Current tolerance: {tolerance}.")

        # register the tolerance for this parameter
        self.__parameter_tolerances[parameter_name] = tolerance

        # register the parameter for comparison
        if used_for_comparison:
            self.__comparison_parameters.add(parameter_name)

    def check_value_in_limits(self, parameter_name: str) -> bool:

        # get the limit from the object
        limit = self.check_and_get_parameter_limit(parameter_name)

        # get the value from the object
        value = self.check_and_get_parameter(parameter_name)

        # check whether the condition is fulfilled
        if limit[0] <= value <= limit[1]:
            return True

        # throw error if outside of definition
        else:
            error_message = (f"{parameter_name} must be between {limit[0]} and {limit[1]}. Currently it is: {value}. "
                             f"{f'Explanation: {self.__limit_error_explanations[parameter_name]}' if self.__limit_error_explanations[parameter_name] else ''}")
            raise ValueError(error_message)

    def __eq__(self, other):
        """
        This function allows comparison of two different signal parts. Currently, it only compares signals
        if they have the same class. For comparison, it uses the registered parameters with the tolerances.

        :param other: the other object to compare with
        :return: boolean
        """
        print(f'I am {type(self).__name__} and am compared to {type(other).__name__}.')
        # check if both signals have the same class
        if isinstance(other, type(self)):
            return all(abs(self.get_parameter(name)-other.get_parameter(name)) < self.get_tolerance(name)
                       for name in self.__comparison_parameters)
        return False

    def __sub__(self, other):

        """
        This allows to subtract the registered parameters of two signal parts having the same class.

        :param other: the other object to subtract from us
        :return: boolean
        """

        # check if both signals have the same class
        if isinstance(other, type(self)):
            return {name: self.get_parameter(name)-other.get_parameter(name) for name in self.parameters}
        else:
            raise NotImplementedError(f"Cannot compare self object of class {type(self).__name__} "
                                      f"to other object of class {type(other).__name__}.")

    def __setattr__(self, key, value):
        print(f'I am {type(self).__name__} and your are setting {key}={value}.')

        # check whether the attribute is modifiable
        if not key in self.__modifiable_parameters:
            raise KeyError(f"Parameter {key} is not a modifiable parameter.")

        # check whether the new value is within its limits
        setattr(self, key, value)
        self.check_value_in_limits(key)


class BaseOscillation(SignalPart):
    """
    This class builds the base for an Oscillation, which we see as a base signal. For example, this can be constant,
    a sine, or a triangle function. Every Oscillation has to have a render function to be called when compiling the
    final signal.
    """

    def __init__(self, length: int):
        super().__init__(length)

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        pass


class NoOscillation(BaseOscillation):
    """
    This will implement a simple line. Offset will be done using a trend.
    """
    def __init__(self, length: int):
        super().__init__(length)

    def render(self) -> int:
        return 0


class BaseTrend(SignalPart):
    """
    This class builds the base for a Trend. For example, this can be constant, a line with a slope. The trend
    is always additive.
    """
    def __init__(self, length: int):
        super().__init__(length)


class ConstantTrend(BaseTrend):

    def __init__(self, offset: float, length: int, tolerance: float):
        super().__init__(length)

        # save the variables
        self.offset = offset
        self._register_parameter('offset', limit=(-float('inf'), float('inf')), tolerance=tolerance,
                                 modifiable=True)

    def render(self):
        return self.offset


class NoTrend(ConstantTrend):
    """
    A class that is the default trend: No trend, adding an offset of zero.
    """
    def __init__(self, length: typing.Union[int, "Signal"], tolerance: float = 0.01):
        super().__init__(0, length, tolerance)



class BaseNoise(SignalPart):
    """
    This class builds the base for a Noise signal. Noise will be always additive and is assigned to every
    Signal. It has to have a render function to be called when compiling the signal.
    """

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        pass


class NoNoise(BaseNoise):
    """
    This class is the default if no noise is specified. It adds no noise to the signal.
    """

    def __init__(self, length: int):
        super().__init__(length)

    def render(self, *args, **kwargs) -> float:
        return 0


class Signal:
    """
    This is the class for a Signal. The signal always consists of an Oscillation and Noise.
    """
    def __init__(self, oscillation: BaseOscillation, noise: BaseNoise = None, trend: BaseTrend = None):

        if len(oscillation.shape) != 1:
            raise AttributeError(f"Oscillation has to be a 1D array. Currently: {len(oscillation.shape)}.")

        # create the defaults if necessary
        if noise is None:
            noise = NoNoise(oscillation.shape[0])
        if trend is None:
            trend = NoTrend(oscillation.shape[0])

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
    def __init__(self, signals: list[Signal], trend: BaseTrend = None, noise: BaseNoise = None):

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
            trend = NoTrend(self.shape[0])
        else:
            if trend.shape != self.shape:
                raise ValueError(f"Shape of trend and change signal must be the same. Change Signal: {self.shape},"
                                 f"Trend: {trend.shape}.")
        self.trend = trend

        # check for the noise (after we save the signals, as we need their shape to check)
        if noise is None:
            noise = NoNoise(self.shape[0])
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
    a = BaseOscillation(100)
    b = a.render()
    print(b is None)
