import collections
import typing
import warnings

import numpy as np

# import all the necessary parts (we have to import everything, so all classes can be inferred)
from changepoynt.simulation import base
from changepoynt.simulation import oscillations
from changepoynt.simulation import trends
from changepoynt.simulation import noises
from changepoynt.simulation import randomizers as rds
from changepoynt.simulation import signals
from changepoynt.simulation import errors as ers
from changepoynt.simulation.signals import ChangeSignal


class ChangeGenerator:

    def __init__(self,
                 length: int,
                 minimum_length: typing.Optional[int] = None,
                 samples_before_change: typing.Optional[typing.Union[int, float]] = None,
                 rate: typing.Optional[typing.Union[int, float]] = None,
                 random_generator: np.random.Generator = None,
                 verbose: bool = False,):

        # save the length
        self.length = length
        self.verbose = verbose

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
            # also get rid of start and end (as those are zero and self.length)
            cplist = np.array(cplist[1:-1])

            # choose the values using a uniform distribution
            choose = self.random_generator.uniform(low=0.0, high=1.0, size=cplist.shape[0])

            # choose with probability
            cplist = cplist[choose <= linkage]

            # transform back to list and append to the new events
            new_event_list.extend(cplist)

        # sort the new event list
        new_event_list = [0] + sorted(new_event_list) + [self.length]

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
                f'The sampled event number ({sampled_event_number}) is larger than the allowed events. Defaulting to {max_events=}.')
        num_events = min(max_events, sampled_event_number)
        return num_events

    def generate_independent_lists(self, num_lists: int):

        # generate the number of events we aim for
        num_events = self.generate_number_of_events()
        print('Number of events:', num_events)

        # check whether we have enough events to get our num_lists
        if num_events < num_lists:
            raise ValueError(
                f"We could not sample enough change points for the specified number of lists '{num_lists=}'. Either decrease 'num_lists', decrease '{self.samples_before_change=}', or increase '{self.rate=}'.")

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
            sorted_events = sorted(events[start:stop])
            sorted_events = [0] + sorted_events + [self.length]
            event_lists.append(sorted_events)
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

        # keep the lists and fuse the extra ones, make sure to get rid of zeros and self.length
        # for the disturbances (therefore only from 1 to element -1)
        independent_lists = lists[:num_lists]
        disturbances = [ele for ls in lists[num_lists:] for ele in ls[1:-1]]
        return independent_lists, disturbances


class ChangeSignalGenerator:

    def __init__(self,
                 oscillation_selectors: dict[str: base.RandomSelector] = None,
                 initial_oscillation_selector: base.RandomSelector = None,
                 default_oscillation_selector: base.RandomSelector = None,
                 trend_selectors: dict[str: base.RandomSelector] = None,
                 initial_trend_selector: base.RandomSelector = None,
                 default_trend_selector: base.RandomSelector = None,
                 noise_selectors: dict[str: base.RandomSelector] = None,
                 initial_noise_selector: base.RandomSelector = None,
                 default_noise_selector: base.RandomSelector = None,
                 parameter_distributions: dict[tuple[str, str, str]: base.ParameterDistribution] = None,
                 transition_selectors: dict[tuple[str, str]: base.RandomSelector] = None,
                 default_transition_selector: base.RandomSelector = None,
                 random_generator: typing.Optional[typing.Union[np.random.Generator, int]] = None,
                 verbose: bool = False,
                 retries: int = 100) -> None:

        # save whether we are verbose
        self.verbose = verbose
        self.retries = retries

        # check the retries so it is correct
        if not isinstance(self.retries, int) or self.retries <= 0:
            raise TypeError(f"'{retries=}' must be a positive integer.")

        # initialize general variables ---------------------------------------------------------------------------------
        # save the random state
        if random_generator is None:
            self.random_generator = np.random.default_rng()
        elif isinstance(random_generator, np.random.Generator):
            self.random_generator = random_generator
        elif isinstance(random_generator, int):
            self.random_generator = np.random.default_rng(random_generator)
        else:
            raise TypeError("'random_state' must be either None or a np.random.Generator.")

        # initialize the random signal selection -----------------------------------------------------------------------

        # get the possible classes for each signal part
        self.possible_parts = {'trend': base.SignalPart.get_registered_signal_parts_group(base.BaseTrend),
                               'noise': base.SignalPart.get_registered_signal_parts_group(base.BaseNoise),
                               'oscillation': base.SignalPart.get_registered_signal_parts_group(base.BaseOscillation),
                               'transition': base.SignalPart.get_registered_signal_parts_group(base.BaseTransition)}

        # create the default values for the oscillation initial selector
        if initial_oscillation_selector is None:
            initial_oscillation_selector = rds.PreferenceRandomSelector(random_generator=self.random_generator,
                                                                        preferred_choice='NoOscillation',
                                                                        preferred_probability=0.8,
                                                                        choices=list(self.possible_parts['oscillation'].keys()))
        # create the default values for the default oscillation selector
        if default_oscillation_selector is None:
            default_oscillation_selector = rds.ConditionalRandomSelector(keep_probability=0.75,
                                                                         random_generator=self.random_generator,
                                                                         choices=list(self.possible_parts['oscillation'].keys()))
        # create the default value for the oscillation selectors
        if oscillation_selectors is None:
            oscillation_selectors = {}
        # create the oscillation selector
        oscillation_selectors = self.handle_signal_part_selectors(oscillation_selectors, initial_oscillation_selector,
                                                                  default_oscillation_selector, base.BaseOscillation)


        # create the default values for the trend initial selector
        if initial_trend_selector is None:
            initial_trend_selector = rds.PreferenceRandomSelector(random_generator=self.random_generator,
                                                                  preferred_choice='ConstantOffset',
                                                                  preferred_probability=0.9,
                                                                  choices=list(self.possible_parts['trend'].keys()))
        # create the default values for the default trend selector
        if default_trend_selector is None:
            default_trend_selector = rds.ConditionalRandomSelector(keep_probability=0.85,
                                                                   random_generator=self.random_generator,
                                                                   choices=list(self.possible_parts['trend'].keys()))
        # create the default value for the oscillation selectors
        if trend_selectors is None:
            trend_selectors = {}
        # create the trend selector
        trend_selectors = self.handle_signal_part_selectors(trend_selectors, initial_trend_selector,
                                                            default_trend_selector, base.BaseTrend)


        # create the default values for the noise initial selector
        if initial_noise_selector is None:
            initial_noise_selector = rds.StaticSelector('GaussianNoise')
        # create the default values for the default noise selectors
        if default_noise_selector is None:
            default_noise_selector = rds.StaticSelector('GaussianNoise')
        # create the default value for the oscillation selectors
        if noise_selectors is None:
            noise_selectors = {}
        # create the noise selector
        noise_selectors = self.handle_signal_part_selectors(noise_selectors, initial_noise_selector,
                                                            default_noise_selector, base.BaseNoise)

        # save all the selectors using their names
        self.selectors = {'trend': trend_selectors, 'noise': noise_selectors, 'oscillation': oscillation_selectors}

        # initialize random parameter selection ------------------------------------------------------------------------

        # get all the parameters that are registered
        self.possible_parameters = base.SignalPart.get_all_registered_parameters_for_randomization()

        # go through all the parameters and get the default distributions (while setting the random generator)
        self.parameter_distributions = {key: param.default_parameter_distribution.set_random_generator(self.random_generator)
                                        for key, param in self.possible_parameters.items()}

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
        
        # initialize random transition selection -----------------------------------------------------------------------

        # set the default selector
        if default_transition_selector is None:
            default_transition_selector = rds.RandomSelector(random_generator=self.random_generator,
                                                             choices=list(self.possible_parts['transition'].keys()))
        elif isinstance(default_transition_selector, base.RandomSelector):
            default_transition_selector = default_transition_selector
        else:
            raise TypeError("'default_transition_selector' must be either None or a RandomSelector.")

        # get all the transitions that are registered
        self.possible_transitions = base.SignalPart.get_all_possible_transitions()

        # save the default parameter distribution
        self.transition_selectors = collections.defaultdict(lambda: default_transition_selector)

        # create the default transition_selectors
        if transition_selectors is None:
            transition_selectors = {}

        # overwrite the specified selectors
        for key, selector in transition_selectors.items():

            # the key has to be existing
            if key not in self.possible_transitions:
                raise KeyError(f"{key=} in 'transition_selectors' seems to specify an invalid transition.")

            # update the dictionary
            self.transition_selectors[key] = selector

    def handle_signal_part_selectors(self,
                                     selectors: dict[str: base.RandomSelector],
                                     initial_selector: base.RandomSelector,
                                     default_selector: base.RandomSelector,
                                     signal_part_subclass: typing.Type[base.SignalPart]) \
            -> dict[str: base.RandomSelector]:

        # check that the selectors have the right type
        if not isinstance(initial_selector, base.RandomSelector):
            raise TypeError(f"'initial_{signal_part_subclass.type_name}_selector' must be an instance of {base.RandomSelector}.")
        if not isinstance(default_selector, base.RandomSelector):
            raise TypeError(f"'initial_{signal_part_subclass.type_name}_selector' must be an instance of {base.RandomSelector}.")

        # get the possible signal parts
        possible_signal_parts = self.possible_parts[signal_part_subclass.type_name]

        # save the default parameter distribution
        output_selectors = {name: default_selector for name, classed in possible_signal_parts.items()}

        # create the initial_signal_selector and set it
        output_selectors[None] = initial_selector

        # overwrite the specified selectors
        for key, selector in selectors.items():

            if not isinstance(selector, base.RandomSelector):
                raise TypeError(f"{key=} in '{signal_part_subclass.type_name}_selectors' must be an instance of {base.RandomSelector}.")

            # check whether the key exists
            if key not in possible_signal_parts:
                raise KeyError(f"{key=} in 'signal_selectors' seems to specify an invalid SignalPart of type {signal_part_subclass}.")

            # update the dictionary
            output_selectors[key] = selector
        return output_selectors

    def choose_signal(self, previous_signal_parts: dict[str: str] = None) -> dict[str: str]:

        # set the default if no previous signal has been specified
        if previous_signal_parts is None:
            previous_signal_parts = collections.defaultdict(lambda : None)

        # get the signal parts conditioned on the previous signal parts
        signal_parts = {part_name: self.selectors[part_name][previous_signal_parts[part_name]].get_selection(previous_signal_parts[part_name]) for part_name in self.selectors.keys()}
        return signal_parts

    def instantiate_random_parameter(self, classname: str, parameter_name: str, parameter: base.Parameter,
                                     signal_part: str, previous_signal: signals.Signal):
        # make a print if verbose
        self.printer(f"Choosing parameter {parameter_name}.", level=1)

        # get the previous value (with default if no previous signal is specified)
        prev_val = None
        if previous_signal is not None:
            prev_val = previous_signal.get_randomizeable_parameter_values()[signal_part].get(parameter_name)

        # get the minimum value, disallowed values, and maximum value for the parameter
        mini, *dissalowed_vals, maxi = parameter.get_information()['limit']
        dissalowed_vals = set(dissalowed_vals)

        # get the random value from the parameter distribution
        random_val = self.parameter_distributions[(signal_part, classname, parameter_name)].get_parameter(prev_val)

        # get the parameter tolerance
        tolerance = parameter.get_information()['tolerance']

        # try to create the parameter multiple times
        for retry_index in range(self.retries):
            if any(abs(random_val-disallowed_val) <= tolerance for disallowed_val in dissalowed_vals):
                self.printer(
                    f"Chosen parameter {parameter_name} was disallowed: '{random_val=}'. {dissalowed_vals=}. Going again! Try: {retry_index + 1}/{self.retries}.",
                    level=2)
            else:
                break
            random_val = self.parameter_distributions[(signal_part, classname, parameter_name)].get_parameter(prev_val)
        if any(abs(random_val-disallowed_val) <= tolerance for disallowed_val in dissalowed_vals):
            raise ers.RetryError(f"For {parameter_name=} parameter '{random_val}' was disallowed. Retries maxed out.")

        return random_val

    def instantiate_signal(self, signal_parts: dict[str: str], signal_length: int,
                           previous_signal: signals.Signal = None) -> signals.Signal:

        # go through the parts, get the random parameters, choose them, and then instantiate the object
        instantiated_signal_parts = {}
        for signal_part, classname in signal_parts.items():

            # make a print if verbose
            self.printer(f"Creating signal part '{signal_part}' with class '{classname}'.")

            # get the class object
            class_obj = self.possible_parts[signal_part][classname]

            # get the parameters we have to choose
            possible_parameters = class_obj.get_parameters_for_randomizations()

            # go through the parameters and create randomized values
            params = {'length': signal_length}
            for parameter_name, parameter in possible_parameters.items():
                params[parameter_name] = self.instantiate_random_parameter(classname, parameter_name, parameter, signal_part, previous_signal)

            # create the signal part
            instantiated_signal_parts[signal_part] = class_obj(**params)

        # create the signal from the signal parts
        instantiated_signal = signals.Signal.from_dict(instantiated_signal_parts)
        return instantiated_signal

    def generate_signal(self, length: int, previous_signal: signals.Signal = None) -> signals.Signal:

        # get the dict of previous signal parts and their names
        previous_signal_parts = None
        if previous_signal is not None:
            previous_signal_parts = {typename: part_object.__class__.__name__ for typename, part_object in previous_signal.get_signal_parts().items()}

        # get the new signal parts
        signal_parts = self.choose_signal(previous_signal_parts)

        # instantiate the signal
        new_signal = self.instantiate_signal(signal_parts, length, previous_signal=previous_signal)
        return new_signal

    def generate_from_events(self, event_list: list[int,]) -> signals.ChangeSignal:

        print('Event List', event_list)

        # go through the list of events and create the signals
        previous_signal = None
        signal_list = []
        for event_start, event_end in zip(event_list, event_list[1:]):

            # compute the length of the signal
            signal_length = event_end - event_start

            # generate the signal
            previous_signal = self.generate_signal(signal_length, previous_signal=previous_signal)

            # save the signal
            signal_list.append(previous_signal)
        print(signal_list)
        return ChangeSignal(signal_list)

    def printer(self, msg: str, level: int = 0):
        if self.verbose:
            indents = "\t"*level
            print(f'{indents}{msg}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cpg = ChangeGenerator(length=1000, minimum_length=100, rate=1/5)
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

    # make a change signal generator
    csg = ChangeSignalGenerator(verbose=False)
    haha = csg.generate_signal(200)
    plt.plot(haha.render())

    # create a change signal from the event list
    plt.figure()
    chnggg = csg.generate_from_events(_indep[0])
    plt.plot(chnggg.render())
    plt.show()
