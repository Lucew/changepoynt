# import builtin packages
import typing
import os

# import foreign packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import own code
import changepoynt.simulation.generator as cpsim
import changepoynt.simulation.randomizers as rds
import changepoynt.simulation.base as base
import changepoynt.simulation.signals as cpsig
import changepoynt.simulation.serialization as cpser

from changepoynt.algorithms.esst import ESST
from changepoynt.visualization.score_plotting import plot_data_and_score

def get_transitions():
    transitions = cpsim.ChangeSignalGenerator.get_possible_transitions()
    transitions = {key: {sigsig: list(trans.keys()) for sigsig, trans in transis.items()} for key, transis in transitions.items()}
    return transitions

def get_possible_parts():
    possible_parts = {key: list(value.keys()) for key, value in cpsim.ChangeSignalGenerator.get_possible_parts().items()}
    return possible_parts

def create_oscillation_transition_selector(allowed_transitions: tuple[str], no_transition_prob: float,
                                           random_generator: np.random.Generator):
    allowed_transition_set = set(allowed_transitions)
    allowed_transition_set.add('NoTransition')

    possible_transitions = get_transitions()['oscillation']
    osctranselec = {
        sigsig: [ele for ele in val if ele in allowed_transition_set] for sigsig, val in possible_transitions.items()
    }

    osctranselec = {sigsig: rds.PreferenceRandomSelector(random_generator=random_generator,
                                                         preferred_choice='NoTransition',
                                                         preferred_probability=no_transition_prob,
                                                         choices=values)
                    for sigsig, values in osctranselec.items()}
    return osctranselec


def create_trend_transition_selector(allowed_transitions: tuple[str], no_transition_prob: float,
                                     random_generator: np.random.Generator):
    allowed_transition_set = set(allowed_transitions)
    allowed_transition_set.add('NoTransition')

    possible_transitions = get_transitions()['trend']
    trendtranselec = {
        sigsig: [ele for ele in val if ele in allowed_transition_set] for sigsig, val in possible_transitions.items()
    }

    trendtranselec = {sigsig: rds.PreferenceRandomSelector(random_generator=random_generator,
                                                           preferred_choice='NoTransition',
                                                           preferred_probability=no_transition_prob,
                                                           choices=values)
                      for sigsig, values in trendtranselec.items()}
    return trendtranselec


def create_oscillation_selector(allowed_oscillations: tuple[str], no_oscillation_prob: float,
                                random_generator: np.random.Generator):
    allowed_oscillations = list(allowed_oscillations)
    if 'NoOscillation' not in allowed_oscillations:
        allowed_oscillations.append('NoOscillation')
    oscgen = rds.PreferenceRandomSelector(random_generator=random_generator, preferred_choice='NoOscillation',
                                          preferred_probability=no_oscillation_prob, choices=allowed_oscillations)
    return oscgen


def create_trend_selector(allowed_trends: tuple[str], no_trend_prob: float,
                          random_generator: np.random.Generator):
    allowed_trends = list(allowed_trends)
    if 'ConstantOffset' not in allowed_trends:
        allowed_trends.append('ConstantOffset')
    trendgen = rds.PreferenceRandomSelector(random_generator=random_generator, preferred_choice='ConstantOffset',
                                            preferred_probability=no_trend_prob, choices=allowed_trends)
    return trendgen

def create_initial_trend_selector(allowed_trends: tuple[str], random_generator: np.random.Generator):
    allowed_trends = list(allowed_trends)
    if 'ConstantOffset' not in allowed_trends:
        allowed_trends.append('ConstantOffset')
    trend_gen = rds.RandomSelector(allowed_trends, random_generator=random_generator)
    return trend_gen


def create_initial_oscillation_selector(allowed_oscillations: tuple[str], random_generator: np.random.Generator):
    allowed_oscillations = list(allowed_oscillations)
    if 'NoOscillation' not in allowed_oscillations:
        allowed_oscillations.append('NoOscillation')
    trend_gen = rds.RandomSelector(allowed_oscillations, random_generator=random_generator)
    return trend_gen


def get_change_signals(signal_number: int, signal_length: int,
                       allowed_oscillations: tuple[str] = None,
                       allowed_trends: tuple[str] = None,
                       allowed_transitions: tuple[str] = None,
                       no_oscillation_prob: float = 0.8,
                       no_trend_prob: float = 0.9,
                       no_transition_prob: float = 0.25,
                       event_rate: typing.Optional[float] = None,
                       random_seed: typing.Optional[int] = None,
                       ) -> cpsig.ChangeSignalMultivariate:

    # compute an event rate from experience
    if event_rate is None:
        event_rate = 10 / signal_length * 1.3

    # set some default values
    if allowed_oscillations is None:
        allowed_oscillations = get_possible_parts()['oscillation']
    if allowed_trends is None:
        allowed_trends = get_possible_parts()['trend']
    if allowed_transitions is None:
        allowed_transitions = get_possible_parts()['transition']

    # make the random generator using the seed (reproducibility)
    randg = np.random.default_rng(random_seed)

    # get the initial trend selectors
    initial_trend_selector = create_initial_trend_selector(allowed_trends, randg)
    initial_oscillation_selector = create_initial_oscillation_selector(allowed_oscillations, randg)

    # get the oscillation and selectors
    oscselec = create_oscillation_selector(allowed_oscillations, no_oscillation_prob, randg)
    trenselect = create_trend_selector(allowed_trends, no_trend_prob, randg)

    # get the transition selectors
    osctranselec = create_oscillation_transition_selector(allowed_transitions, no_transition_prob, randg)
    trentranselec = create_trend_transition_selector(allowed_transitions, no_transition_prob, randg)

    # create the change signal generator
    csg = cpsim.ChangeSignalGenerator(default_oscillation_selector=oscselec, default_trend_selector=trenselect,
                                      oscillation_transition_selectors=osctranselec,
                                      trend_transition_selectors=trentranselec,
                                      random_generator=randg,
                                      initial_trend_selector=initial_trend_selector,
                                      initial_oscillation_selector=initial_oscillation_selector)

    # create the events
    minimum_length = base.SignalPart.get_minimum_length()
    random_change_point = randg.integers(minimum_length, signal_length-minimum_length, size=signal_number)
    independent_events = [[0, int(random_change_point[idx]), signal_length] for idx in range(signal_number)]

    # generate the signals
    independent_signals = [csg.generate_from_events(events) for events in independent_events]
    change_points = independent_signals[0].get_change_points()
    change_points[0].copy()


    # create a multivariate change signal
    multivariate_change_signal = cpsig.ChangeSignalMultivariate(independent_signals)

    # make a copy of the multivariate change signal
    return multivariate_change_signal

def signals_to_dataframe(change_signals: cpsig.ChangeSignalMultivariate) -> pd.DataFrame:
    # put the signals into a dataframe
    df = pd.DataFrame.from_dict(change_signals.to_array_dict())
    return df

def format_bytes(num: int) -> str:
    # human-friendly bytes
    result_str = ""
    for unit in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024 or unit == "TB":
            result_str = f"{num:.0f} {unit}" if unit == "bytes" else f"{num:.2f} {unit}"
            break
        num /= 1024.0
    return result_str

def main(persist: bool = True, test_persistence: bool = True):
    # create the signals with a random fixed random seed
    # change parameters or seed for different results
    signals = get_change_signals(1000, 1_000, random_seed=42)
    signal_df = signals_to_dataframe(signals)

    # plot the signals
    sns.lineplot(data=signal_df.iloc[:, :10])
    plt.show()

    # persist the data if necessary
    if persist:

        # persist the json file
        with open('simulation.json', 'w') as f:
            f.write(signals.to_json(compress=False))

        # persist the dataframe
        signal_df.to_parquet('simulation.parquet')

        # check how large the files are
        print('Simulation Package Json Serialization memory size:\t', format_bytes(os.path.getsize('simulation.json')))
        print('Rendered array dataframe to parquet memory size:\t', format_bytes(os.path.getsize('simulation.parquet')))
        print()

    # check the persistence
    if test_persistence:
        with open('simulation.json', 'r') as f:
            signals_copy = cpser.from_json(f.read())
        signal_df_copy = pd.read_parquet('simulation.parquet')

        np.testing.assert_array_equal(signals_copy.render(), signals.render())

        print('Check whether the persisted data is the same as in memory:')
        print('Simulation Package Json Serialization:\t', np.array_equal(signals_copy.render(), signals.render()))
        print('The rendered array parquet files:\t\t', pd.DataFrame.equals(signal_df, signal_df_copy))


if __name__ == '__main__':
    main()
