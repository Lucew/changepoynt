import typing
import abc
import inspect
import functools
import sys
import collections

import numpy as np


class ParameterDistribution:

    def __init__(self,
                 minimum: typing.Optional[typing.Union[float, int]] = None,
                 maximum: typing.Optional[typing.Union[float, int]] = None,
                 random_generator: typing.Optional[np.random.Generator] = None,
                 verbose: bool = False):

        # save the values
        self.minimum = minimum
        if self.minimum is None:
            self.minimum = -np.inf
        self.maximum = maximum
        if self.maximum is None:
            self.maximum = np.inf
        self.verbose = verbose

        # make the default random state
        self.random_generator = random_generator
        if self.random_generator is None:
            self.random_generator = np.random.default_rng()

        # check whether minimum is smaller than the maximum
        if self.minimum > self.maximum:
            raise ValueError(f"{minimum=} must be smaller than maximum {maximum=}.")

        # create some variables to save which class and parameter the distribution is attached too
        self.class_name = "??"
        self.parameter_name = "??"

    def set_random_generator(self, random_generator: np.random.Generator) -> 'ParameterDistribution':
        # protect from setting arbitrary random generator
        if random_generator is None:
            return self

        if not isinstance(random_generator, np.random.Generator):
            raise TypeError(f"{type(random_generator)=} must be an instance of np.random.Generator")
        self.random_generator = random_generator
        return self

    @abc.abstractmethod
    def generate_random_number(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        raise NotImplementedError

    def get_parameter(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:

        # get the random number from the implementation
        random_val = self.generate_random_number(prev_value)

        # limit the random number
        prev_str = "" if self.class_name == "??" else f"Class {self.class_name}, Parameter {self.parameter_name}. "
        if random_val < self.minimum:
            if self.verbose:
                f"{prev_str}Generated {random_val=} for is too low. Setting to minimum value '{self.minimum=}'."
            random_val = self.minimum
        if random_val > self.maximum:
            if self.verbose:
                f"{prev_str}Generated {random_val=} for is too high. Setting to minimum value '{self.maximum=}'."
            random_val = self.maximum
        return random_val


class RandomSelector:

    def __init__(self, choices: list[str], random_generator: typing.Optional[np.random.Generator] = None):

        # get the signal parts that we allow
        registered_signal_parts = set(SignalPart.get_registered_signal_parts().keys())

        # check the type
        if not isinstance(choices, list):
            raise TypeError(f"{choices=} is not a list.")
        if not all(isinstance(ele, str) for ele in choices):
            raise TypeError(f"Not all elements in {choices=} are a string.")

        # check the choices
        if all(choice in registered_signal_parts for choice in choices):
            self.choices = choices
        else:
            raise ValueError(f"Not all elements of {choices=} are registered signal parts: {registered_signal_parts=}.")

        # save the random generator
        self.random_generator = random_generator

        # make the default random state
        if self.random_generator is None:
            self.random_generator = np.random.default_rng()
        elif not isinstance(self.random_generator, np.random.Generator):
            raise TypeError(f"{self.random_generator=} must be an instance of np.random.Generator")

    @abc.abstractmethod
    def get_selection(self, prev_signal_part: str) -> str:
        raise NotImplementedError


class Parameter:
    def __init__(self, param_type: typing.Union[type, tuple[type,...]],
                 default_value: typing.Union[int, float] = None,
                 limit: tuple[typing.Union[int, float], ...] = None,
                 tolerance: typing.Union[int, float] = None,
                 derived: bool = False,
                 modifiable: bool = True,
                 use_for_comparison: bool = True,
                 use_random: bool = True,
                 default_parameter_distribution: ParameterDistribution = None,
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
        if any((not type(limitval) or limitval == np.inf) in self.param_type for limitval in limit):
            raise TypeError(f"All values in limit must be of the same type as the parameter ('{self.param_type}'). Currently: {list(type(limitval) for limitval in limit)} for {limit=}.")
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
            if not self.limit[0] <= default_value <= self.limit[-1]:
                raise ValueError(f"Default value {default_value} out of range {limit}.")
            # check the illicit values
            for illicit_val in self.limit[1:-1]:
                if abs(default_value - illicit_val) <= self.tolerance:
                    raise ValueError(f"Default value {default_value} not allowed within tolerance ({self.tolerance}) of illicit value '{illicit_val}' in limit {self.limit}.")

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
        self.name = "??"
        self.class_name = "??"

        # save the docstring
        self.doc = doc

        # check the parameter distribution (only if it is a random parameter)
        self.default_parameter_distribution = None
        if self.use_random:

            # check whether there is one
            if default_parameter_distribution is None:
                raise ValueError(f"'default_parameter_distribution' cannot be None for a randomize-able parameter ({self.use_random=}). Please specify a 'default_parameter_distribution' (see traceback).")
            if not isinstance(default_parameter_distribution, ParameterDistribution):
                raise TypeError(f"Default parameter distribution must be of type {ParameterDistribution.__name__}. Currently: {type(default_parameter_distribution)=}.")

            # take a guess using the default value
            try:
                random_val = default_parameter_distribution.get_parameter(self.default_value)
            except BaseException as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                raise ValueError(f"We tried to run the get_parameter method from the 'default_parameter_distribution' using the specified '{self.default_value=}' and an error occurred. Please only specify 'default_parameter_distribution' that runs using the default value.\nOriginal Exception: {e}.\nFile {exc_tb.tb_frame.f_code.co_filename}, line {exc_tb.tb_lineno}")

            # check the random_val for the correct type
            if type(random_val) not in self.param_type:
                raise TypeError(f"The '{random_val=}' created by the 'default_parameter_distribution' does not return the correct type ({type(random_val)=}). Possible types: {self.param_type}.")

            # check whether the limits are within the limits of the parameter
            if default_parameter_distribution.minimum < self.limit[0] or self.limit[-1] < default_parameter_distribution.maximum:
                raise ValueError(f"The 'default_parameter_distribution' minimum and maximum is outside of the current {self.limit=}: From '{default_parameter_distribution.minimum}' to '{default_parameter_distribution.maximum}'.")

            # safe the parameter distribution
            self.default_parameter_distribution = default_parameter_distribution

            # set the names of the default parameter distribution
            self.default_parameter_distribution.parameter_name = self.name
            self.default_parameter_distribution.class_name = self.class_name

    def get_information(self):
        return {'limit': self.limit, 'tolerance': self.tolerance, 'default_value': self.default_value, 'type': self.param_type}

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
            raise TypeError(f"Parameter '{self.name}' for class '{instance.__class__}' must be one of {self.param_type}. Currently: {type(value)}.")

        # set the value of the instance and not of the parameter
        # otherwise it will be set globally for all instances of this class
        instance._values[self.name] = value

    def __str__(self):
        return f"{self.__class__.__name__} {self.name}({self.param_type=}, {self.default_value=}, {self.limit=}, {self.tolerance=}, {self.derived=}, {self.modifiable=}, {self.use_for_comparison=}, {self.use_random}, {self.default_parameter_distribution=})"


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
            param.class_name = name

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
                if not all(inspect.isclass(objects) and issubclass(objects, SignalPart) and inspect.isclass(objects) for objects in object_list):
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

        # check whether there are any methods for acquiring a final parameter that start with _final_
        # if that is, it has to return a value of the same type as the parameter allowed types
        for finalfunc in dir(new_cls):

            # find the parts of the new class that have a certain naming scheme
            if finalfunc.startswith('_final_'):

                # get the actual object
                finalobj = getattr(new_cls, finalfunc)

                # check that it is indeed a function
                if not callable(finalobj):
                    raise TypeError(f"All properties of class {name} that start with '_final_' are reserved for functions that compute final parameter values. Currently it is not a callable, but {type(finalobj)}.")

                # check that it returns the correct type
                return_type = typing.get_type_hints(finalobj)
                if 'return' not in return_type:
                    raise NotImplementedError(f"The render function of class {name} is missing a return type annotation.")

                # get the name of the parameter it returns
                param_name = finalfunc.replace('_final_', '')

                # check whether the paramater exists
                if param_name not in parameters:
                    raise NotImplementedError(f"The function {finalfunc} of class {name} aims to compute {param_name=}, but this parameter does not exist.")

                # get the parameter itself
                curparam = parameters[param_name]

                # check that the returned type is correct
                return_type = return_type['return']
                if return_type not in curparam.param_type:
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


def check_first_arg_class():
    """
    Decorator to check if the first argument (typically 'self') is an instance of expected_class.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise ValueError("Function called without positional arguments.")
            if args[0] != SignalPart:
                raise TypeError(f"First argument must be class {SignalPart}, got {args[0]} instead.")
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SignalPart(metaclass=SignalPartMeta):
    _registry = dict()
    _minimum_length = 30
    type_name: str

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
        # self._parameters will be set by the metaclass
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
                raise ValueError(f"Missing parameter: {name}. Please specify when constructing the object of type {type(self).__name__}.")


        # check that all kwargs are parameters
        unknown_parameters = kwargs.keys() - self._parameters.keys()
        if unknown_parameters:
            raise ValueError(f"Your specified unknown parameters: {unknown_parameters}.")

        # whether we specified derived parameters
        derived_parameters = kwargs.keys() & {name for name, param in self._parameters.items() if param.derived}
        if derived_parameters:
            raise ValueError(f"Your specified derived parameters: {derived_parameters}.")

    @classmethod
    @check_first_arg_class()
    def get_immediate_subclasses(cls):
       return cls.__subclasses__()

    @classmethod
    @check_first_arg_class()
    def get_registered_signal_parts(cls):
        return {key: val for key, val in cls._registry.items()}

    @classmethod
    @check_first_arg_class()
    def get_registered_signal_parts_group(cls, group: typing.Type['SignalPart']):
        return {key: val for key, val in cls._registry.items() if issubclass(val, group)}

    @classmethod
    @check_first_arg_class()
    def get_registered_signal_parts_grouped(cls):

        # get all the immediate subclasses
        subclasses = cls.get_immediate_subclasses()

        # find the accompanying classes
        return {group.type_name: cls.get_registered_signal_parts_group(group) for group in subclasses}

    @classmethod
    @check_first_arg_class()
    def get_possible_transitions(cls, from_class: typing.Type['SignalPart'], to_class: typing.Type['SignalPart']):

        # get the possible transitions
        possible_transitions = {}
        for name, transition_class in cls.get_registered_signal_parts_group(BaseTransition).items():
            # make a type annotation so the linter is not confused
            transition_class: BaseTransition

            # check which transitions allow us to transition between the two classes
            if not any(issubclass(from_class, allowed) for allowed in transition_class.allowed_from):
                continue
            if not any(issubclass(to_class, allowed) for allowed in transition_class.allowed_to):
                continue

            # check whether both from and to belong to the same signal part type (trend, oscillation, noise)
            if not any(issubclass(from_class, target_class) and issubclass(to_class, target_class) for target_class in cls.get_immediate_subclasses()):
                continue

            # save the transition class
            possible_transitions[name] = transition_class
        return possible_transitions

    @classmethod
    @check_first_arg_class()
    def get_all_possible_transitions(cls):

        # get all the SignalParts
        signal_parts = cls.get_registered_signal_parts_grouped()

        # iterate over all possible classes
        possible_transitions = collections.defaultdict(dict)
        for clsname1, cls1 in ((clsname, class1) for clses in signal_parts.values() for clsname, class1 in
                               clses.items()):
            # skip transitions
            if issubclass(cls1, BaseTransition):
                continue

            for clsname2, cls2 in ((clsname, class2) for clses in signal_parts.values() for clsname, class2 in
                                   clses.items()):
                # skip transitions
                if issubclass(cls2, BaseTransition):
                    continue

                # check whether both from and to belong to the same signal part type (trend, oscillation, noise)
                found_target_class = None
                for target_class in cls.get_immediate_subclasses():
                    if issubclass(cls1, target_class) and issubclass(cls2, target_class):
                        found_target_class = target_class.type_name
                if found_target_class is None:
                    continue

                # get the possible transitions
                possible_transitions[found_target_class][(clsname1, clsname2)] = cls.get_possible_transitions(cls1, cls2)

                # check whether something is off
                if len(possible_transitions[found_target_class][(clsname1, clsname2)]) == 0:
                    raise NotImplementedError(f'There is no possible transition between {cls1} and {cls2}, both of type {found_target_class}.')
        # create dict from the default dict
        return dict(possible_transitions)

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
        instance = cls(**data)
        return instance

    @classmethod
    def get_parameters(cls):
        return cls._parameters

    def get_parameter_value(self, name):
        return self._values[name]

    @classmethod
    def get_parameters_for_randomizations(cls):
        return {name: param for name, param in cls._parameters.items() if param.use_random}

    def get_parameters_for_randomizations_values(self):

        # decide which function to use (either use a calculated value or the parameter value)
        # print(self.__class__.__name__)
        return {name: getattr(self, name) if not hasattr(self, f'_final_{name}') else getattr(self, f'_final_{name}')()
                for name in self.get_parameters_for_randomizations().keys()}

    @classmethod
    def get_all_registered_parameters_for_randomization(cls) -> dict[tuple[str, str, str]: Parameter]:
        return {(parttype, clsname, paramname): parameter
                for parttype, typeclasses in cls.get_registered_signal_parts_grouped().items()
                for clsname, clsinstance in typeclasses.items()
                for paramname, parameter in clsinstance.get_parameters_for_randomizations().items()}

    @classmethod
    def get_minimum_length(cls):
        return cls._minimum_length

    def __str__(self):
        return_str = [f"Class {self.__class__.__name__} of type {self.type_name} with parameters:"]
        for paramname, parameter in self.get_parameters().items():
            return_str.append(f"\t{paramname} = {getattr(self,paramname)}")
        return "\n".join(return_str)


class BaseOscillation(SignalPart):
    """
    This class builds the base for an Oscillation, which we see as a base signal. For example, this can be constant,
    a sine, or a triangle function. Every Oscillation has to have a render function to be called when compiling the
    final signal.
    """
    type_name = "oscillation"
    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class BaseTrend(SignalPart):
    """
    This class builds the base for a Trend. For example, this can be constant, a line with a slope. The trend
    is always additive.
    """
    type_name = "trend"
    @abc.abstractmethod
    def render(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


class BaseNoise(SignalPart):
    """
    This class builds the base for a Noise signal. Noise will be always additive and is assigned to every
    Signal. It has to have a render function to be called when compiling the signal.
    """
    type_name = "noise"
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
    type_name = "transition"
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
        if not any(isinstance(to_object, allowed_to) for allowed_to in self.allowed_to):
            raise TypeRestrictedError(f"'to_object' must be one of types {self.allowed_to}.")

        # check that the length is at least one
        if isinstance(transition_length, int):
            if transition_length < 1:
                raise ValueError(f"In case of an 'int' type 'length' must be greater than 0. Currently: '{transition_length}'.")
            self.transition_length = transition_length
        elif isinstance(transition_length, float):
            if not 0 < transition_length < 0.5:
                raise ValueError(f"In case of a 'float' type 'length' must be in interval (0, 0.5) exclusively. Currently: '{transition_length}'.")
            self.transition_length = int(min(from_object.shape[0], to_object.shape[0])*transition_length)
            # take care that the transition length is at least one sample
            self.transition_length = max(self.transition_length, 1)
        else:
            raise TypeError(f"Length must be either 'float' or 'int'. Currently: '{type(transition_length)}'.")

        # check that to and from object are different
        if id(from_object) == id(to_object):
            raise ValueError(f"From object and to object cannot be equal for a transition.")

        # check whether both from and to belong to the same signal part type (trend, oscillation, noise)
        if not any(isinstance(from_object, target_class) and isinstance(to_object, target_class) for target_class in SignalPart.get_immediate_subclasses()):
            raise TypeError(f"'{from_object.__class__=}' and '{to_object.__class__=}' have to both be of the same SignalPartType (One of: {SignalPart.get_immediate_subclasses()}).")

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

    def __str__(self):
        return f"{self.__class__.__name__}(length={self.transition_length}, from={self.from_object.__class__.__name__}, to={self.to_object.__class__.__name__})"



if __name__ == "__main__":
    a = BaseOscillation(100)
    b = a.render()
    print(b is None)
