import abc
import collections
import typing

import numpy as np

# import all the necessary parts
from changepoynt.simulation import base
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends
from changepoynt.simulation import noises
from changepoynt.simulation import signals


class ConditionalGaussianDistribution(base.ParameterDistribution):

    def __init__(self, standard_deviation: float, random_state: typing.Optional[np.random.RandomState] = None):

        # save the standard deviation
        self.std = standard_deviation
        self.random_state = random_state

        # make the default random state
        if self.random_state is None:
            self.random_state = np.random.RandomState()

    def get_parameter(self, previous_value: typing.Union[float, int] = 0.0) -> typing.Union[float, int]:
        return self.random_state.normal(previous_value, self.std)


class NoDistribution(base.ParameterDistribution):

    def __init__(self):
        super().__init__()

    def get_parameter(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        return prev_value


class SignalGenerator:

    def __init__(self, oscillations_weights: dict[str: int] = None,
                 trend_weights: dict[str: int] = None,
                 noise_weights: dict[str: int] = None,
                 parameter_distributions: dict[tuple[str, str]: base.ParameterDistribution] = None,
                 default_parameter_distribution: base.ParameterDistribution = None,
                 random_state: np.ndarray = None) -> None:

        # get all the oscillations that are implemented
        self.possible_oscillations = base.SignalPart.get_registered_signal_parts_group(base.BaseOscillation)
        self.possible_trends = base.SignalPart.get_registered_signal_parts_group(base.BaseTrend)
        self.possible_noises = base.SignalPart.get_registered_signal_parts_group(base.BaseNoise)
        print('Oscillations', self.possible_oscillations)
        print('Trends', self.possible_trends)
        print('Noises', self.possible_noises)

        # make default weights or check the existing ones
        self.oscillations_weights = self.process_input_weights(oscillations_weights, self.possible_oscillations)
        self.trend_weights = self.process_input_weights(trend_weights, self.possible_trends)
        self.noise_weights = self.process_input_weights(noise_weights, self.possible_noises)

        # save the random state
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            raise TypeError("'random_state' must be either None or a np.random.RandomState.")

        # create the default or save the default_parameter_distributions
        if default_parameter_distribution is None:
            self.default_parameter_distribution = ConditionalGaussianDistribution(standard_deviation=20.0,
                                                                                  random_state=self.random_state)
        elif isinstance(default_parameter_distribution, base.ParameterDistribution):
            self.default_parameter_distribution = default_parameter_distribution
        else:
            raise TypeError("'default_parameter_distribution' must be either None or a ParameterDistribution.")

        # save the default parameter distribution
        self.parameter_distributions = collections.defaultdict(lambda : default_parameter_distribution)

        # get all the parameters that are registered
        self.possible_parameters = base.SignalPart.get_all_registered_parameters_for_randomization()
        print('Parameters', self.possible_parameters)

        # create the default value
        if parameter_distributions is None:
            parameter_distributions = {}

        # go through the specified parameter distributions and put them into
        for key, val in parameter_distributions.items():

            # check whether this (class, parameter) combination exists in the possible ones
            if key in self.possible_parameters:

                # check whether the distribution actually is of the right type
                if isinstance(val, base.ParameterDistribution):
                    self.parameter_distributions[key] = val
                else:
                    raise TypeError(f"The distribution for combo (class, parameter): '{key}' in 'parameter_distributions' has to be of type 'ParameterDistribution'. Currently: {type(val)}.")
            else:
                raise KeyError(f"You specified a distribution for the combo (class, parameter): '{key}' in the 'parameter_distributions'. This combo is not registered.")

    @staticmethod
    def create_default_weights(possible_scenarios: dict):
        return collections.defaultdict(lambda: 100//len(possible_scenarios))

    def process_input_weights(self, input_weights: typing.Union[dict[str: int], None],
                              possible_scenarios: dict) -> dict[str: int]:

        # make the default scenario
        if input_weights is None:
            return self.create_default_weights(possible_scenarios)

        # check whether
        return input_weights


if __name__ == '__main__':
    SignalGenerator()
