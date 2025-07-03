import typing

import numpy as np

from changepoynt.simulation import base


class ContinuousConditionalGaussianDistribution(base.ParameterDistribution):
    """
    This class models a conditional gaussian that is conditioned using the previous value as a mean.
    The user can specify the standard deviation around this mean. If no condition is given the previous value is assumed
    to be zero.
    """

    def __init__(self, standard_deviation: float, minimum: float = None, maximum: float = None, default_mean: float = 0.0,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

        # save the variables
        self.std = standard_deviation
        self.default_mean = default_mean

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> float:
        # set the default value if necessary
        if previous_value is None:
            previous_value = self.default_mean
        return self.random_generator.normal(previous_value, self.std)


class ContinuousGaussianDistribution(base.ParameterDistribution):
    """
    This class models a conditional gaussian with a specifiable mean.
    """

    def __init__(self, standard_deviation: float, minimum: float = None, maximum: float = None, default_mean: float = 0.0,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

        # save the variables
        self.std = standard_deviation
        self.default_mean = default_mean

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> float:
        # set the default value if necessary
        if previous_value is None:
            previous_value = self.default_mean
        return self.random_generator.normal(previous_value, self.std)


class DiscreteConditionalGaussianDistribution(base.ParameterDistribution):
    """
    This class models a conditional gaussian that is conditioned using the previous value as a mean.
    The user can specify the standard deviation around this mean. If no condition is given the previous value is assumed
    to be zero.
    """

    def __init__(self, standard_deviation: float, minimum: float = None, maximum: float = None,
                 default_mean: float = 0.0, random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

        # save the variables
        self.std = standard_deviation
        self.default_mean = default_mean

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> int:
        # set the default value if necessary
        if previous_value is None:
            previous_value = self.default_mean
        return round(self.random_generator.normal(previous_value, self.std))


class DiscretePoissonDistribution(base.ParameterDistribution):
    """
    This class models a conditional gaussian that is conditioned using the previous value as a mean.
    The user can specify the standard deviation around this mean. If no condition is given the previous value is assumed
    to be zero.
    """

    def __init__(self, rate: float, minimum: float, maximum: float,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

        # save the variables
        self.rate = rate

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> int:
        return self.random_generator.poisson(self.rate)


class ContinuousConditionalUniformDistribution(base.ParameterDistribution):
    """
    This class models a uniform distribution that is conditioned on the previous value as a mean.
    The user can specify a range around the mean value. If no condition is given the previous value is assumed
    to be zero.
    """

    def __init__(self, uniform_range: float,  minimum: float, maximum: float,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

        # save the standard deviation
        self.range = uniform_range

    def generate_random_number(self, previous_value: typing.Union[float, int] = 0.0) -> float:
        return self.random_generator.uniform(previous_value-self.range/2, previous_value+self.range/2)


class ContinuousUniformDistribution(base.ParameterDistribution):
    """
    This class models a uniform distribution. The values will be between minimum and maximum.
    """

    def __init__(self, minimum: float, maximum: float,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

    def generate_random_number(self, previous_value: typing.Union[float, int] = 0.0) -> typing.Union[float, int]:
        return self.random_generator.uniform(self.minimum, self.maximum)


class DiscreteConditionalUniformDistribution(base.ParameterDistribution):

    """
    This class models a uniform distribution. The user can specify a range around a previous value.
    If no condition is given the distribution will output any values between minimum and maximum.
    """
    def __init__(self, uniform_range: int, minimum: float, maximum: float, random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)
        self.range = uniform_range

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> int:
        if previous_value is None:
            return self.random_generator.randint(self.minimum, self.maximum)
        return self.random_generator.randint(int(max(previous_value-self.range, self.minimum)), int(min(previous_value + self.range+1, self.maximum+1)))


class DiscreteUniformDistribution(base.ParameterDistribution):

    """
    This class models a uniform distribution. The user can specify a range around a previous value.
    If no condition is given the distribution will output any values between minimum and maximum.
    """
    def __init__(self, minimum: float, maximum: float, random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(minimum, maximum, random_generator)

    def generate_random_number(self, previous_value: typing.Union[float, int] = None) -> int:
        return self.random_generator.integers(self.minimum, self.maximum, endpoint=True, dtype=int)


class NoDistribution(base.ParameterDistribution):
    """
    This class models no distribution. The previous value is just returned.
    """
    def __init__(self, value: typing.Union[float, int]):
        super().__init__(minimum=value, maximum=value)
        self.value = value

    def generate_random_number(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        return self.value


class ConditionalRandomSelector(base.RandomSelector):
    """
    This class models a conditional selector of discrete values. The user can specify one probability which is the
    probability that the choice stays the same as before. The leftover probability is uniformly distributed among all
    other choices.
    """
    def __init__(self, choices: list[str], keep_probability: float,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(choices, random_generator)
        # check and keep the probability
        if not (0.0 <= keep_probability <= 1.0):
            raise ValueError(f"'{keep_probability=}' must be between 0.0 and 1.0.")
        self.keep_probability = keep_probability

    def get_selection(self, previous_value: str) -> str:

        # check that the previous value is in the choices
        if previous_value not in self.choices:
            raise ValueError(f"'{previous_value=}' is not in '{self.choices=}'.")

        probabilities = [0]*len(self.choices)
        other_prob = (1-self.keep_probability)/(len(self.choices)-1)
        for idx, choice in enumerate(self.choices):
            if choice != previous_value:
                probabilities[idx] = other_prob
            else:
                probabilities[idx] = self.keep_probability

        return self.random_generator.choice(self.choices, p=probabilities)


class PreferenceRandomSelector(base.RandomSelector):
    """
    This class models a conditional selector of discrete values. The user can specify a preferred choice and the
    corresponding probability.
    """
    def __init__(self, choices: list[str], preferred_choice: str, preferred_probability: float,
                 random_generator: typing.Optional[np.random.Generator] = None):
        super().__init__(choices, random_generator)

        # check that the preferred choice is a possible choice
        if preferred_choice not in self.choices:
            raise ValueError(f"'{preferred_choice=}' is not in '{self.choices=}'.")
        self.preferred_choice = preferred_choice

        # check and keep the probability
        if not (0.0 <= preferred_probability <= 1.0):
            raise ValueError(f"'{preferred_probability=}' must be between 0.0 and 1.0.")
        self.preferred_probability = preferred_probability

    def get_selection(self, previous_value: str) -> str:

        probabilities = [0]*len(self.choices)
        other_prob = (1-self.preferred_probability)/(len(self.choices)-1)
        for idx, choice in enumerate(self.choices):
            if choice != self.preferred_choice:
                probabilities[idx] = other_prob
            else:
                probabilities[idx] = self.preferred_probability

        return self.random_generator.choice(self.choices, p=probabilities)


class RandomSelector(base.RandomSelector):
    """
    This class models a conditional selector of discrete values. It chooses randomly using a specified probability
    distribution (array of values). If no distribution is given, it will use a uniform distribution.
    """
    def __init__(self, choices: list[str], probabilities: list[float] = None,
                 random_generator: typing.Optional[np.random.Generator] = None):
        # initialize the super class
        super().__init__(choices, random_generator)

        # check the list of probabilities
        if probabilities is not None:
            if abs(sum(probabilities)-1) > 1e-6:
                raise ValueError(f"Sum of 'probabilities' must be equal to 1.0. Currently {sum(probabilities)=}.")
            if any(ele < 0 for ele in probabilities):
                raise ValueError(f"Every element of 'probabilities' must be greater than 0.")
        self.probabilities = probabilities

        # check the length of the probabilities
        if self.probabilities is not None and len(self.choices) != self.probabilities:
            raise ValueError(f"'{len(self.probabilities)=}' must be equal to {len(choices)=}.")

    def get_selection(self, previous_value: str) -> str:

        # use the probabilities if we have them
        if self.probabilities is None:
            return self.random_generator.choice(self.choices)
        return self.random_generator.choice(self.choices, p=self.probabilities)


class StaticSelector(base.RandomSelector):

    def __init__(self, output: str):
        super().__init__([], None)
        self.output = output

    def get_selection(self, previous_value: str) -> str:
        return self.output