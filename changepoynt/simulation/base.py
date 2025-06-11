import typing
import abc
from inspect import isclass

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
                raise TypeError(f"The function '{function_name}' for derived parameter '{self.name}' of class {instance.__class__} has to return a value of {self.param_type}. Currently it returns: {type(function_value)}.")
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


class ParameterDistribution:

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_parameter(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        raise NotImplementedError


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

        # the current class is a transition and not the base class
        if bases and any(base.__name__ == 'BaseTransition' for base in bases):

            # check that the every transition has allowed_from and allowed_to class attributes
            if 'allowed_from' not in namespace or 'allowed_to' not in namespace:
                raise NotImplementedError(
                    f"Class '{name}' is a Transition and needs to have 'allowed_from' 'allowed_to' as class variables.")

            # test whether we properly defined the allowed_from and allowed_to attributes of the transition class
            from_to_objects = {'allowed_from': namespace['allowed_from'], 'allowed_to': namespace['allowed_to']}
            for object_name, object_list in from_to_objects.items():
                if not isinstance(object_list, tuple):
                    raise TypeError(f"Attribute '{object_name}' of class '{cls.__name__}' has to be of type tuple. Currently: {type(object_list)}.")
                if len(object_list) == 0:
                    raise ValueError(f"Attribute '{object_name}' of class '{cls.__name__}' has to specify at least one object.")
                if not all(isclass(objects) and issubclass(objects, SignalPart) and isclass(objects) for objects in object_list):
                    raise TypeError(f"All objects {object_name} of class '{cls.__name__}' have to be a subclass of SignalPart. Currently: {object_list}.")

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

        # check that the render function returns a numpy array
        has_render = has_concrete_render(new_cls)
        if has_render:
            return_type = typing.get_type_hints(getattr(new_cls, "render"))
            if 'return' not in return_type:
                raise NotImplementedError(f"The render function of class {name} is missing a return type annotation.")

            return_type = return_type['return']
            if return_type != np.ndarray:
                raise TypeError(f"The render function of class {name} has to return np.ndarray. Currently the type hint says: {return_type}.")

        # put the class into the registry
        if has_render and SignalPart not in bases:
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
    _minimum_length = 30

    def __init__(self, length: int, **kwargs):

        # save and check the length
        self.minimum_length = self._minimum_length if not isinstance(self, BaseTransition) else 0
        length = length
        if length < self.minimum_length:
            raise ValueError(f'Instance class {self.__class__} length is too short: {length} < {self.minimum_length}.')
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
    def get_registered_signal_parts_group(cls, group: typing.Type['SignalPart']):
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

    def get_parameter_value(self, name):
        return self._values[name]

    @classmethod
    def get_parameters_for_randomizations(cls):
        return {name: param for name, param in cls._parameters.items() if param.use_random}

    @classmethod
    def get_all_registered_parameters_for_randomization(cls):
        return {(clsname, paramname): parameter for clsname, clsinstance in cls.get_registered_signal_parts_group(cls).items() for paramname, parameter in clsinstance.get_parameters_for_randomizations().items()}

    @classmethod
    def get_possible_transitions(cls, from_object: 'SignalPart', to_object: 'SignalPart'):
        possible_transitions = {}
        for name, transition_class in cls.get_registered_signal_parts_group(BaseTransition).items():
            try:
                transition = transition_class(transition_length=100, from_object=from_object, to_object=to_object)
                possible_transitions[name] = transition
            except TypeRestrictedError:
                continue
        return possible_transitions

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

    @classmethod
    def get_minimum_length(cls):
        return cls._minimum_length


class BaseOscillation(SignalPart):
    """
    This class builds the base for an Oscillation, which we see as a base signal. For example, this can be constant,
    a sine, or a triangle function. Every Oscillation has to have a render function to be called when compiling the
    final signal.
    """

    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


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


class TypeRestrictedError(TypeError):
    pass


class BaseTransition(SignalPart):
    """
    This class builds the base for a Transition. A transition always connects two classes of oscillations.
    """
    allowed_from: tuple[typing.Type[SignalPart], ...]
    allowed_to: tuple[typing.Type[SignalPart], ...]

    @property
    @abc.abstractmethod
    def allowed_from(self):
        """
        This property is the modern way of specifying an abstract attribute
        https://stackoverflow.com/a/41897823
        :return:
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def allowed_to(self):
        """
        This property is the modern way of specifying an abstract attribute
        https://stackoverflow.com/a/41897823
        :return:
        """
        raise NotImplementedError

    def __init__(self, transition_length: typing.Union[int, float], from_object: SignalPart, to_object: SignalPart):

        # the length of a transition is the two signals combined
        super().__init__(from_object.shape[0]+to_object.shape[0])

        # check whether the from and to object are allowed
        if not any(isinstance(from_object, allowed_from) for allowed_from in self.allowed_from):
            raise TypeRestrictedError(f"'from_object' must be one of types {self.allowed_from}.")
        if not any(isinstance(from_object, allowed_to) for allowed_to in self.allowed_to):
            raise TypeRestrictedError(f"'from_object' must be one of types {self.allowed_to}.")

        # check that the length is at least one
        if isinstance(transition_length, int):
            if transition_length < 1:
                raise ValueError(f"In case of an 'int' type 'length' must be greater than 0. Currently: '{transition_length}'.")
            self.transition_length = transition_length
        elif isinstance(transition_length, float):
            if not 0 < transition_length < 0.5:
                raise ValueError(f"In case of a 'float' type 'length' must be in interval (0, 0.5) exclusively. Currently: '{transition_length}'.")
            self.transition_length = int(min(from_object.shape[0], to_object.shape[0])*transition_length)
        else:
            raise TypeError(f"Length must be either 'float' or 'int'. Currently: '{type(transition_length)}'.")

        # check that to and from object are different
        if from_object == to_object:
            raise ValueError(f"From object and to object cannot be equal for a transition.")


        # save the object we are coming from
        self.from_object = from_object
        self.to_object = to_object

        # check that both from and to objects are longer or equal to the transition length
        if from_object.shape[0]//2 < self.transition_length:
            raise ValueError(f"The specified 'transition_length' ({transition_length}) has to be shorter than half the length of the 'from_object' ({from_object.shape[0]//2}).")
        if to_object.shape[0]//2 < self.transition_length:
            raise ValueError(f"The 'transition_length' ({transition_length}) has to be shorter than half the length of the 'to_object' ({from_object.shape[0]//2}).")

        # get the start y-values and end y-values by rendering both objects
        self.start_y = from_object.render()[-self.transition_length:]
        self.end_y = to_object.render()[:self.transition_length]

    @abc.abstractmethod
    def get_transition_values(self) -> np.ndarray:
        """
        Every transition subclass must implement this method.

        The function has to fulfill the following criteria:
        - return a numpy array with one dimension and length of exactly 2*self.transition_length
        - can use the from and to object
        :return:
        """
        raise NotImplementedError

    def check_objects(self, from_object: SignalPart, to_object: SignalPart):
        """
        This function checks whether two input objects are the ones that were used to specify the transition.

        :param from_object: The SignalPart object we want to transition from
        :param to_object: The SignalPart object we want to transition to
        :return: Boolean, True if the two objects are the ones specified when creating the transition
        """
        return from_object is self.from_object and to_object is self.to_object

    def render(self) -> np.ndarray:
        """
        This function creates the transition values.
        :return: Tuple[np.ndarray, np.ndarray], (TransitionValuesFrom, TransitionValuesTo)
        """

        # get the transition values from the actual implementation
        transition_values = self.get_transition_values()

        # check whether the implementation works as expected
        info_string = f"Problematic Class is '{self.__class__.__name__}'."
        if not isinstance(transition_values, np.ndarray):
            raise TypeError(f"{info_string} The return value of 'get_transition_values' must be a numpy array. Current type: {type(transition_values)}.")
        if transition_values.ndim != 1:
            raise np.exceptions.AxisError(f"{info_string} The return array of 'get_transition_values' must be a 1D numpy array. Current shape: {transition_values.shape}.")
        if transition_values.shape[0] != self.transition_length*2:
            raise np.exceptions.AxisError(f"{info_string} The return array of 'get_transition_values' must have the length of 2*'transition_length' ({self.transition_length}). Current length: {transition_values.shape[0]}.")

        return transition_values

    def apply(self, rendered_from_signal: np.ndarray, rendered_to_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This function applies the transition function from 'from_object' to 'to_object'.
        !IMPORTANT! This operation acts inplace.

        :param rendered_from_signal: the array containing the signal values the transition starts from
        :param rendered_to_signal: the array containing the signal values the transition goes to
        :return: Changed numpy arrays, modified inplace!
        """

        # check that the signals have been rendered by the objects specified (and not altered)
        from_rendered = self.from_object.render()[-self.transition_length:]
        if not np.all(np.equal(rendered_from_signal[-self.transition_length:], from_rendered)):
            raise ValueError(f"The 'rendered_from_signal[-self.transition_length]' unequal to the rendered from_object. Is the input array different than rendered from the 'from_object'?")
        to_rendered = self.to_object.render()[:self.transition_length]
        if not np.all(np.equal(rendered_to_signal[:self.transition_length], to_rendered)):
            raise ValueError(f"The 'rendered_to_signal[:self.transition_length]' unequal to the rendered to_object. Is the input array different than rendered from the 'to_object'?")

        # render the current overlap signal
        transition_values = self.render()
        rendered_from_signal[-self.transition_length:] = transition_values[:self.transition_length]
        rendered_to_signal[:self.transition_length] = transition_values[self.transition_length:]
        return rendered_from_signal, rendered_to_signal



if __name__ == "__main__":
    a = BaseOscillation(100)
    b = a.render()
    print(b is None)
