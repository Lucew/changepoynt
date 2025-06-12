import collections
import typing
import warnings

import numpy as np

# import all the necessary parts (we have to import everything, so all classes can be inferred)
from changepoynt.simulation import base
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends
from changepoynt.simulation import noises
from changepoynt.simulation import signals


class ConditionalGaussianDistribution(base.ParameterDistribution):

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

    def __init__(self):
        super().__init__()

    def get_parameter(self, prev_value: typing.Union[float, int]) -> typing.Union[float, int]:
        return prev_value


class ChangeGenerator:

    def __init__(self,
                 length: int,
                 minimum_length: typing.Optional[int] = None,
                 samples_before_change: typing.Optional[typing.Union[int, float]] = None,
                 rate: typing.Optional[typing.Union[int, float]] = None,
                 random_generator: np.random.Generator = None):

        # save the length
        self.length = length

        # save the minimum time between events
        if minimum_length is None:
            minimum_length = base.SignalPart.get_minimum_length()
        if minimum_length < base.SignalPart.get_minimum_length():
            raise ValueError(
                f"'minimum_length={minimum_length}' of between changes must be greater than the minimum length of the signal parts ('base.SignalPart.get_minimum_length()'={base.SignalPart.get_minimum_length()}).")
        self.minimum_length = minimum_length

        # parametrize the gamma distribution for choosing the change events
        # samples_before_change (shape parameter, sometimes called k) corresponds to the number of events we
        # are waiting for.
        # if samples_before_change is large, we have fewer events
        #
        # rate (rate or inverse scale parameter) is the rate at which events occur.
        # if the rate is high then we have to wait less event for a larger samples_before_change
        if samples_before_change is None:
            samples_before_change = 3
        if rate is None:
            rate = 1 / self.minimum_length
        self.samples_before_change = samples_before_change
        self.rate = rate

        # save the random state
        if random_generator is None:
            self.random_generator = np.random.default_rng()
        elif isinstance(random_generator, np.random.Generator):
            self.random_generator = random_generator
        else:
            raise TypeError("'random_state' must be either None or a np.random.Generator.")

    @staticmethod
    def max_points(maximum_value: int, minimum_distance: int):
        return maximum_value // minimum_distance - 1

    def generate_random_points(self, maximum_value: int, minimum_distance: int,
                               number_of_points: typing.Optional[int] = None) -> list[int,]:

        # check whether we need a default value of points
        if number_of_points is None:
            number_of_points = self.random_generator.integers(0, self.max_points(maximum_value, minimum_distance),
                                                              endpoint=True)

        # increase the number of points by two, as we need to first and the last point to be at minimum
        # and maximum value (there is a "change" at the beginning and the end of the signal)
        number_of_points += 2

        # get the minimum length of the signal that we have to have
        min_total_length = (number_of_points - 1) * minimum_distance

        # compute the available space that is left to distribute
        available_space = maximum_value - min_total_length

        # check whether we have enough space
        if available_space < 0:
            raise ValueError(
                f"Not enough space for the given '{maximum_value=}', '{minimum_distance=}', and 'number_of_points={number_of_points - 2}'. We can only have a maximum of: {self.max_points(maximum_value, minimum_distance)}).")

        # create the gaps to distribute the values
        gaps = np.zeros(number_of_points - 1, dtype=int)

        # distribute the available space (instead of a loop we create random integers
        idces = self.random_generator.integers(low=0, high=number_of_points - 1, size=available_space)
        for idx in idces:
            gaps[idx] += 1

        # Generate the points from the gaps
        points = [0]
        current = 0
        for gap in gaps:
            current += minimum_distance + gap
            points.append(current)

        # throw away start and end point (they are fixed)
        assert points[0] == 0 and points[-1] == maximum_value, 'Something with the random points is off.'
        points = points[1:-1]

        return points

    def generate_dependent_list(self, original_lists: list[list[int]], linking_weights: list[float,]):
        """
        Generate a new list of events given the original lists and linking weights.

        Parameters:
        - original_lists: List[List[int]] of original event times.
        - linking_weights: List[float] of weights (sum to 1) for each original list.
        - max_time: Maximum event time for new list.
        - min_distance: Minimum distance between events in the new list.
        - max_events: Optional maximum number of events to sample.

        Returns:
        - new_list: Sorted list of integer event times.
        """

        # get the number of steering signals that we have
        signal_number = len(original_lists)

        # check that we have weights to all the signals
        assert len(
            linking_weights) == signal_number, f"We require a linking value for each original_list. Currently. '{len(linking_weights)=}' not equal to '{signal_number=}'."

        # check that the weights sum up to one
        assert abs(
            sum(linking_weights) - 1.0) < 1e-6, f"Linking weights must sum to 1. Current sum: {sum(linking_weights)}."

        # go through the lists and pick change events with a probability
        new_event_list = []
        for cplist, linkage in zip(original_lists, linking_weights):
            # transform the cplist into numpy array for faster processing
            cplist = np.array(cplist)

            # choose the values using a uniform distribution
            choose = self.random_generator.uniform(low=0.0, high=1.0, size=cplist.shape[0])

            # choose with probability
            cplist = cplist[choose <= linkage]

            # transform back to list and append to the new events
            new_event_list.extend(cplist)

        # sort the new event list
        new_event_list = sorted(new_event_list)

        return new_event_list

    def generate_number_of_events(self):

        # check how many events we can create at a maximum
        max_events = self.max_points(self.length, self.minimum_length)

        # sample from the gamma distribution
        gamma_val = self.random_generator.gamma(self.samples_before_change, 1 / self.rate)
        sampled_event_number = int(self.length / gamma_val)

        # now divide the length by the number of events
        if sampled_event_number > max_events:
            warnings.warn(
                f'The sampled event number ({sampled_event_number}) if larger than the allowed events. Defaulting to {max_events=}.')
        num_events = min(max_events, sampled_event_number)
        return num_events

    def generate_independent_lists(self, num_lists: int):

        # generate the number of events we aim for
        num_events = self.generate_number_of_events()
        print('Number of events:', num_events)

        # check whether we have enough events to get our num_lists
        if num_events < num_lists:
            raise ValueError(
                f"We could not sample enough change points for the specified number of lists '{num_lists=}'. Either decrease 'num_lists', decrease 'samples_before_change', or increase 'rate'.")

        # generate the events point
        events = self.generate_random_points(self.length, self.minimum_length, num_events)

        # shuffle the events so we can assign them randomly
        self.random_generator.shuffle(events)

        # get the split points
        split_points = self.generate_random_points(len(events), 1, num_lists - 1)
        split_points = [0] + split_points + [self.length]

        # get the split event lists
        event_lists = []
        for start, stop in zip(split_points, split_points[1:]):
            event_lists.append(sorted(events[start:stop]))
        return event_lists

    def generate_independent_list_disturbed(self, num_lists: int, extra_lists: typing.Optional[int] = None):

        # get more independent lists than we need
        if extra_lists is None:
            extra_lists = max(1, int(num_lists * 0.25))

        # try to sample the lists
        try:
            lists = self.generate_independent_lists(num_lists + extra_lists)
        except ValueError as excp:
            raise ValueError(
                f"We tried to generate {num_lists + extra_lists=} independent lists. Most likely these were to much. Original error below.\n\n{excp}.")

        # keep the lists and fuse the extra ones
        independent_lists = lists[:num_lists]
        disturbances = [ele for ls in lists[num_lists:] for ele in ls]
        return independent_lists, disturbances


class ChangeSignalGenerator:

    def __init__(self,
                 oscillations_weights: dict[str: int] = None,
                 trend_weights: dict[str: int] = None,
                 noise_weights: dict[str: int] = None,
                 parameter_distributions: dict[tuple[str, str]: base.ParameterDistribution] = None,
                 default_parameter_distribution: base.ParameterDistribution = None,
                 random_generator: np.random.Generator = None) -> None:

        # get all the oscillations that are implemented
        self.possible_oscillations = base.SignalPart.get_registered_signal_parts_group(base.BaseOscillation)
        self.possible_trends = base.SignalPart.get_registered_signal_parts_group(base.BaseTrend)
        self.possible_noises = base.SignalPart.get_registered_signal_parts_group(base.BaseNoise)

        # make default weights or check the existing ones
        self.oscillations_weights = self.process_input_weights(oscillations_weights, self.possible_oscillations)
        self.trend_weights = self.process_input_weights(trend_weights, self.possible_trends)
        self.noise_weights = self.process_input_weights(noise_weights, self.possible_noises)

        # save the random state
        if random_generator is None:
            self.random_generator = np.random.default_rng()
        elif isinstance(random_generator, np.random.Generator):
            self.random_generator = random_generator
        else:
            raise TypeError("'random_state' must be either None or a np.random.Generator.")

        # create the default or save the default_parameter_distributions
        if default_parameter_distribution is None:
            self.default_parameter_distribution = ConditionalGaussianDistribution(standard_deviation=20.0,
                                                                                  random_generator=self.random_generator)
        elif isinstance(default_parameter_distribution, base.ParameterDistribution):
            self.default_parameter_distribution = default_parameter_distribution
        else:
            raise TypeError("'default_parameter_distribution' must be either None or a ParameterDistribution.")

        # save the default parameter distribution
        self.parameter_distributions = collections.defaultdict(lambda : default_parameter_distribution)

        # get all the parameters that are registered
        self.possible_parameters = base.SignalPart.get_all_registered_parameters_for_randomization()

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

    def generate_from_length(self, min_length: int, max_length: int, parts: int) -> signals.ChangeSignal:
        pass


if __name__ == '__main__':
    cpg = ChangeGenerator(length=10000)
    _pts = cpg.generate_random_points(maximum_value=100, minimum_distance=10)
    print(_pts)

    # generate some change point lists
    _indep = cpg.generate_independent_lists(6)
    print('Printing independent lists:')
    for _id in _indep:
        print(_id)
    print()

    # generate dependent values from the independent lists
    print('A dependent list')
    print(cpg.generate_dependent_list(_indep, [0.9, 0.0, 0.1, 0.0, 0.0, 0.0]))
    print()

    # generate list with disturbances
    _indep, _dist = cpg.generate_independent_list_disturbed(4, 3)
    print('Printing independent lists:')
    for _id in _indep:
        print(_id)
    print('Printing disturbances:', _dist)
    print()
