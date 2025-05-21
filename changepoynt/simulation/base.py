import typing
import abc

import numpy as np


class Parameter:
    def __init__(self, param_type: typing.Union[type, tuple[type,...]],
                 default_value: typing.Union[int, float] = None,
                 limit: tuple[typing.Union[int, float], ...] = None,
                 tolerance: typing.Union[int, float] = None,
                 derived: bool = False,
                 modifiable: bool = True,
                 use_for_comparison: bool = True,
                 use_random: bool = True,
                 limit_error_explanation: str = "",
                 doc: str = ""):

        # save the parameter type as a tuple
        if isinstance(param_type, tuple):
            self.param_type = param_type
        else:
            self.param_type = (param_type, )

        # check the limit and set a default if limit is None
        if limit is None:
            limit = (-float("inf"), float("inf"))
        if len(limit) < 2:
            raise TypeError(f"Parameter limit must have minimum length of 2. Currently: {len(limit)}.")
        if any(limit1 >= limit2 for limit1, limit2 in zip(limit, limit[1:])):
            raise ValueError(f"Parameter limits must be strictly increasing. Currently: {limit}.")
        self.limit = limit

        # check the tolerance and save default
        if tolerance is None:
            tolerance = 0
        if tolerance < 0:
            raise ValueError(f"Parameter tolerances must be positive. Currently: {tolerance}.")
        self.tolerance = tolerance

        # save derived and modifiable and make them exclusive
        self.derived = derived
        self.modifiable = modifiable
        if self.derived and self.modifiable:
            raise ValueError("Derived parameters cannot be modifiable. derived=True -> modifiable=False.")
        if self.derived and use_random:
            raise ValueError("Derived parameters cannot be used for randomization. derived=True -> use_random=False.")

        # validate the default value
        if default_value is not None:
            if not isinstance(default_value, param_type):
                raise TypeError(f"Default value {default_value} does not match type {param_type.__name__}.")
            if not self.limit[0] <= default_value <= self.limit[1]:
                raise ValueError(f"Default value {default_value} out of range {limit}.")
        self.default_value = default_value

        # save whether we are ignoring the parameter for comparison
        self.use_for_comparison = use_for_comparison

        # save the error explanation
        self.limit_error_explanation = limit_error_explanation

        # check that if a parameter is not allowed for randomization it either has to be derived
        # or it has to have a default
        if not use_random and not (self.derived or self.default_value is not None):
            raise ValueError("If the parameter should not be used for randomization (use_random=False). "
                             "It either has to have a default value or has to be derived.")
        self.use_random = use_random

        # the name will be set by the SignalPartMetaclass
        self.name = None

        # save the docstring
        self.doc = doc

    def get_information(self):
        return {'limit': self.limit, 'tolerance': self.tolerance, 'default_value': self.default_value}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        if self.derived:
            # create the function name
            function_name = f"compute_{self.name}"

            # Derived values must be computed by a method/property
            function_value = getattr(instance, function_name)()

            # check that we get the specified type
            if not any(isinstance(function_value, paramtype) for paramtype in self.param_type):
                raise TypeError(f"The function for derived parameter {self.name} of class {instance.__class__} has to return a value of {self.param_type}. Currently it returns: {type(function_value)}.")
            return function_value
        return instance._values.get(self.name)

    def __set__(self, instance, value):

        # check whether it is derived
        if self.derived:
            raise AttributeError(f"'{self.name}' is derived and cannot be set.")

        # check whether it is modifiable, and we already have an instance
        if not self.modifiable and self.name in instance._values:
            raise AttributeError(f"Parameter '{self.name}' is not modifiable.")

        # check the limit
        if not (self.limit[0] <= value <= self.limit[-1]):
            raise ValueError(f"Parameter '{self.name}' with value {value} out of range: {self.limit}.{f' {self.limit_error_explanation}.' if self.limit_error_explanation else ''}")

        # check the illicit values
        for illicit_val in self.limit[1:-1]:
            if abs(value - illicit_val) <= self.tolerance:
                raise ValueError(f"Parameter '{self.name}' with value {value} not allowed within tolerance ({self.tolerance}) of illicit value '{illicit_val}' in limit {self.limit}.")

        # check the type of the set value
        if not any(isinstance(value, paramtype) for paramtype in self.param_type):
            raise TypeError(f"Parameter '{self.name}' must be one of {self.param_type}. Currently: {type(value)}.")

        # set the value of the instance and not of the parameter
        # otherwise it will be set globally for all instances of this class
        instance._values[self.name] = value


class SignalPartMeta(type):

    def __new__(cls, name, bases, namespace):

        # Collect parameters defined in this class
        parameters = {
            key: val for key, val in namespace.items()
            if isinstance(val, Parameter)
        }

        # set the name of the parameters
        for param_name, param in parameters.items():
            param.name = param_name

        # inherit parameters from base classes (base classes are build first!, so the parameter is always
        # the latest defined one)
        for base in bases:
            if hasattr(base, "_parameters"):
                for param_name, param in base._parameters.items():
                    if param_name not in parameters:
                        parameters[param_name] = param

        # save the parameters in the current class
        namespace["_parameters"] = parameters

        # instantiate the super class
        new_cls = super().__new__(cls, name, bases, namespace)

        # Check for derived parameters and verify compute method exists
        # also check that no parameter length is defined
        for param in parameters.values():

            # check for the parameter length
            if param.name == 'length':
                raise NameError(f'Class {name}: Every signal has a length and it can not be defined as a parameter.')

            # check that the name is not containing any upper case characters
            if not param.name.islower():
                raise NameError(f"'Class {name}: {param.name} must be all lowercase letter.")

            # check that all derived parameters have a computation function
            if param.derived and not hasattr(new_cls, f"compute_{param.name}"):
                raise TypeError(f"'Class {name}: Derived parameter '{param.name}' is missing compute method 'compute_{param.name}'.")

        # put the class into the registry
        if name != "SignalPart" and has_concrete_render(new_cls):
            SignalPart._registry[name] = new_cls

        # return the new class
        return new_cls


def has_concrete_render(cls):
    render_method = getattr(cls, "render", None)
    if not callable(render_method):
        return False
    try:
        # Check if method raises NotImplementedError when called
        # for the self argument we use a None class
        render_method(None)
    except NotImplementedError:
        return False
    except Exception:
        # Don't assume render() runs cleanly; just ensure it's not abstract
        return True
    return True


class SignalPart(metaclass=SignalPartMeta):
    _registry = dict()

    def __init__(self, length: int, **kwargs):

        # save and check the length
        minimum_length = 100
        length = length
        if length < minimum_length:
            raise ValueError(f'Length is too short: {length} < {minimum_length}.')
        self.length = length

        # make a dict that the parameter can fill to save information
        self._values = {}
        self.parameter_info = {}

        # go through the parameters and check whether they have been specified in the keywords
        for name, param in self._parameters.items():

            # save the default values and the limits
            self.parameter_info[name] = param.get_information()

            # if the parameter is derived we do not have to set it
            if param.derived:
                continue

            # check whether the user defined the parameter or we have a default value
            if name in kwargs:
                setattr(self, name, kwargs[name])
            elif param.default_value is not None:
                setattr(self, name, param.default_value)
            else:
                raise ValueError(f"Missing parameter: {name}.")


        # check that all kwargs are parameters
        unknown_parameters = kwargs.keys() - self._parameters.keys()
        if unknown_parameters:
            raise ValueError(f"Your specified unknown parameters: {unknown_parameters}.")

        # whether we specified derived parameters
        derived_parameters = kwargs.keys() & {name for name, param in self._parameters.items() if param.derived}
        if derived_parameters:
            raise ValueError(f"Your specified derived parameters: {derived_parameters}.")

    @classmethod
    def get_registered_signal_parts(cls):
        return {key: val for key, val in cls._registry.items()}

    @classmethod
    def get_registered_signal_parts_group(cls, group):
        return {key: val for key, val in cls._registry.items() if issubclass(val, group)}

    @classmethod
    def get_registered_signal_parts_grouped(cls):

        # get all the immediate subclasses
        subclasses = cls.__subclasses__()

        # find the accompanying classes
        return {group.__name__: cls.get_registered_signal_parts_group(group) for group in subclasses}

    @classmethod
    def get_parameters(cls):
        return cls._parameters

    @classmethod
    def get_parameters_for_randomizations(cls):
        return {name: param for name, param in cls._parameters.items() if param.use_random}

    @property
    def shape(self) -> tuple[int,]:
        return (self.length,)

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def __eq__(self, other):

        # check whether both classes are of the same type
        if not isinstance(other, type(self)):
            return False

        # go through the parameters and check for tolerances
        for name, param in self._parameters.items():

            # check whether we want to skip the parameters
            if not param.use_for_comparison:
                continue

            # get both parameter values
            self_param = getattr(self, name)
            other_param = getattr(other, name)

            # check for the tolerance
            if abs(self_param - other_param) > param.tolerance:
                return False
        return True

    def to_dict(self):
        data = {}
        for name in self._parameters:
            value = getattr(self, name)
            data[name] = value
        return data

    @classmethod
    def from_dict(cls, data):
        # Only non-derived parameters go in constructor
        ctor_args = {
            name: param
            for name, param in cls._parameters.items()
            if not param.derived and name in data
        }
        instance = cls(**ctor_args)

        # Optional: allow updating modifiable fields post-construction
        for name, param in cls._parameters.items():
            if param.derived or not param.modifiable:
                continue
            if name in data:
                setattr(instance, name, data[name])
        return instance


class BaseOscillation(SignalPart):
    """
    This class builds the base for an Oscillation, which we see as a base signal. For example, this can be constant,
    a sine, or a triangle function. Every Oscillation has to have a render function to be called when compiling the
    final signal.
    """

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class NoOscillation(BaseOscillation):
    """
    This will implement a simple line. Offset will be done using a trend.
    """
    def render(self) -> int:
        return 0


class BaseTrend(SignalPart):
    """
    This class builds the base for a Trend. For example, this can be constant, a line with a slope. The trend
    is always additive.
    """

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class BaseNoise(SignalPart):
    """
    This class builds the base for a Noise signal. Noise will be always additive and is assigned to every
    Signal. It has to have a render function to be called when compiling the signal.
    """

    @abc.abstractmethod
    def render(self) -> np.ndarray:
        raise NotImplementedError


if __name__ == "__main__":
    a = BaseOscillation(100)
    b = a.render()
    print(b is None)
