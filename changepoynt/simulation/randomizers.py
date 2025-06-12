import typing

import numpy as np

from changepoynt.simulation import base


class ConditionalGaussianDistribution(base.ParameterDistribution):
    """
    This class models a conditional gaussian that is conditioned using the previous value as a mean.
    The user can specify the standard deviation around this mean. If no condition is given the previous value is assumed
    to be zero.
    """

    def __init__(self, standard_deviation: float, random_generator: typing.Optional[np.random.Generator] = None):

        # save the standard deviation
        self.std = standard_deviation
        self.random_generator = random_generator

        # make the default random state
        if self.random_generator is None:
            self.random_generator = np.random.default_rng()

    def get_parameter(self, previous_value: typing.Union[float, int] = 0.0) -> typing.Union[float, int]:
        return self.random_generator.normal(previous_value, self.std)


class NoDistribution(base.ParameterDistribution):
    """
    This class models no distribution. The previous value is just returned.
    """
    def __init__(self):
        super().__init__()

    def get_parameter(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        return prev_value


class ConditionalContinuousSelector(base.RandomSelector):
    """
    This class models a conditional selector of discrete values. The user can specify one probability which is the
    probability that the choice stays the same as before. The leftover probability is uniformly distributed among all
    other choices.
    """
    def __init__(self, keep_probability: float, random_generator: typing.Optional[np.random.Generator] = None):

        # check and keep the probability
        if not (0.0 <= keep_probability <= 1.0):
            raise ValueError(f"'{keep_probability=}' must be between 0.0 and 1.0.")
        self.keep_probability = keep_probability

        # save the random generator
        self.random_generator = random_generator

        # make the default random state
        if self.random_generator is None:
            self.random_generator = np.random.default_rng()

    def get_selection(self, previous_value: str, choices: list[str]) -> str:

        # check that the previous value is in the choices
        if previous_value not in choices:
            raise ValueError(f"'{previous_value=}' is not in '{choices=}'.")

        probabilities = [0]*len(choices)
        other_prob = (1-self.keep_probability)/(len(choices)-1)
        for idx, choice in enumerate(choices):
            if choice != previous_value:
                probabilities[idx] = other_prob
            else:
                probabilities[idx] = self.keep_probability

        return self.random_generator.choice(choices, p=probabilities)


class RandomSelector(base.RandomSelector):
    """
    This class models a conditional selector of discrete values. It chooses randomly using a specified probability
    distribution (array of values). If no distribution is given, it will use a uniform distribution.
    """
    def __init__(self, probabilities: list[float] = None,
                 random_generator: typing.Optional[np.random.Generator] = None):

        # check the list of probabilities
        if probabilities is not None:
            if abs(sum(probabilities)-1) > 1e-6:
                raise ValueError(f"Sum of 'probabilities' must be equal to 1.0. Currently {sum(probabilities)=}.")
            if any(ele < 0 for ele in probabilities):
                raise ValueError(f"Every element of 'probabilities' must be greater than 0.")
        self.probabilities = probabilities

        # save the random generator
        self.random_generator = random_generator

        # make the default random state
        if self.random_generator is None:
            self.random_generator = np.random.default_rng()

    def get_selection(self, previous_value: str, choices: list[str]) -> str:

        # check that the choices have the same shape as the probabilities
        if self.probabilities is None:
            return self.random_generator.choice(choices)
        else:
            if len(choices) != self.probabilities:
                raise ValueError(f"'{len(self.probabilities)=}' must be equal to {len(choices)=}.")
            return self.random_generator.choice(choices, p=self.probabilities)


class StaticSelector(base.RandomSelector):

    def __init__(self, output: str):
        self.output = output

    def get_selection(self, previous_value: str, choices: list[str]) -> str:
        return self.output