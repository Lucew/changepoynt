import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

import changepoynt.simulation.generator as cpsim
import changepoynt.simulation.randomizers as rds
import exampleExtension

@st.cache_data
def get_possible_parts():
    possible_parts = {key: list(value.keys()) for key, value in cpsim.ChangeSignalGenerator.get_possible_parts().items()}
    return possible_parts

@st.cache_resource
def get_random_generator():
    return np.random.default_rng()

@st.cache_data
def get_transitions():
    transitions = cpsim.ChangeSignalGenerator.get_possible_transitions()
    transitions = {key: {sigsig: list(trans.keys()) for sigsig, trans in transis.items()} for key, transis in transitions.items()}
    return transitions

@st.cache_data
def create_oscillation_selector(allowed_oscillations: tuple[str], no_oscillation_prob: float):
    allowed_oscillations = list(allowed_oscillations)
    if 'NoOscillation' not in allowed_oscillations:
        allowed_oscillations.append('NoOscillation')
    oscgen = rds.PreferenceRandomSelector(random_generator=get_random_generator(), preferred_choice='NoOscillation',
                                          preferred_probability=no_oscillation_prob, choices=allowed_oscillations)
    return oscgen

@st.cache_data
def create_trend_selector(allowed_trends: tuple[str], no_trend_prob: float):
    allowed_trends = list(allowed_trends)
    if 'ConstantOffset' not in allowed_trends:
        allowed_trends.append('ConstantOffset')
    trendgen = rds.PreferenceRandomSelector(random_generator=get_random_generator(), preferred_choice='ConstantOffset',
                                            preferred_probability=no_trend_prob, choices=allowed_trends)
    return trendgen

@st.cache_data
def create_oscillation_transition_selector(allowed_transitions: tuple[str], no_transition_prob: float):
    allowed_transition_set = set(allowed_transitions)
    allowed_transition_set.add('NoTransition')

    possible_transitions = get_transitions()['oscillation']
    osctranselec = {
        sigsig: [ele for ele in val if ele in allowed_transition_set] for sigsig, val in possible_transitions.items()
    }

    osctranselec = {sigsig: rds.PreferenceRandomSelector(random_generator=get_random_generator(),
                                                         preferred_choice='NoTransition',
                                                         preferred_probability=no_transition_prob,
                                                         choices=values)
                    for sigsig, values in osctranselec.items()}
    return osctranselec

@st.cache_data
def get_weights(indep_signal_number: int):
    return [round(get_random_generator().random()*100)/100 for _ in range(indep_signal_number)]


@st.cache_data
def create_trend_transition_selector(allowed_transitions: tuple[str], no_transition_prob: float):
    allowed_transition_set = set(allowed_transitions)
    allowed_transition_set.add('NoTransition')

    possible_transitions = get_transitions()['trend']
    trendtranselec = {
        sigsig: [ele for ele in val if ele in allowed_transition_set] for sigsig, val in possible_transitions.items()
    }

    trendtranselec = {sigsig: rds.PreferenceRandomSelector(random_generator=get_random_generator(),
                                                           preferred_choice='NoTransition',
                                                           preferred_probability=no_transition_prob,
                                                           choices=values)
                      for sigsig, values in trendtranselec.items()}
    return trendtranselec

@st.cache_resource
def get_change_generator(signal_length: int, event_rate: float):
    return cpsim.ChangeGenerator(length=signal_length, minimum_length=50, rate=event_rate, random_generator=get_random_generator())

@st.cache_data
def get_change_events(independent_signals: int, linking_weights: tuple[float,...], signal_length: int, event_rate: float):
    ceg = get_change_generator(signal_length, event_rate)
    _indep, _dist = ceg.generate_independent_list_disturbed(independent_signals, 1)
    dep = ceg.generate_dependent_list(_indep, list(linking_weights))
    return _indep, dep

def get_change_signals(independent_signals: int, linking_weights: tuple[float,], allowed_oscillations: tuple[str],
                       allowed_trends: tuple[str], allowed_transitions: tuple[str], no_oscillation_prob: float,
                       no_trend_prob: float, no_transition_prob: float, signal_length: int, event_rate: float):

    # normalize the linking weights
    linking_weights = tuple(ele/sum(linking_weights) for ele in linking_weights)

    # get the generated events
    indep_events, dep_events = get_change_events(independent_signals, linking_weights, signal_length, event_rate)

    # get the oscillation and selectors
    oscselec = create_oscillation_selector(allowed_oscillations, no_oscillation_prob)
    trenselect = create_trend_selector(allowed_trends, no_trend_prob)

    # get the transition selectors
    osctranselec = create_oscillation_transition_selector(allowed_transitions, no_transition_prob)
    trentranselec = create_trend_transition_selector(allowed_transitions, no_transition_prob)

    # create the change signal generator
    csg = cpsim.ChangeSignalGenerator(default_oscillation_selector=oscselec, default_trend_selector=trenselect,
                                      oscillation_transition_selectors=osctranselec,
                                      trend_transition_selectors=trentranselec,
                                      random_generator=get_random_generator(),
                                      initial_trend_selector=trenselect,
                                      initial_oscillation_selector=oscselec)

    # create the signals
    independent_signals = [csg.generate_from_events(indep) for indep in indep_events]
    dependent_signal = csg.generate_from_events(dep_events)

    # put the events into a dataframe
    df = pd.DataFrame.from_dict({f'Indep. {idx} Sig.': sig.render() for idx, sig in enumerate(independent_signals)})
    df['Depend. Sig.'] = dependent_signal.render()

    # put the weights into a dataframe
    weights = pd.DataFrame(dict(r=linking_weights, theta=[col for col in df.columns if col.startswith('Indep')]))
    return df, weights




def app():
    st.set_page_config(layout="wide")

    # make a sidebar to specify all the simulator options
    simulated_signals = None
    with st.sidebar:
        st.header("Simulation Options")

        st.subheader("General Settings")
        # get the signal length
        signal_length = st.number_input('Signal Length', min_value=1000, max_value=100000, value=5000)

        # specify the event rate
        event_rate = st.slider('Event Rate', min_value=0.0, max_value=1.0, value=0.05, step=0.01)

        # specify the number of independent signals
        independent_signals = st.number_input('Independent Signals', min_value=2, max_value=50, value=4)

        st.subheader("Specify the Allowed Signals")
        # get the allowed trends
        allowed_trends = st.multiselect('Allowed Trends', [ele for ele in get_possible_parts()['trend'] if not ele.startswith('Constant')])

        # get the allowed oscillations
        allowed_oscillations = st.multiselect('Allowed oscillations', [ele for ele in get_possible_parts()['oscillation'] if not ele.startswith('No')])

        # get the allowed transitions
        allowed_transitions = st.multiselect('Allowed transitions', [ele for ele in get_possible_parts()['transition'] if not ele.startswith('No')])

        # get the preferences for no oscillation, no trend and no transition
        no_osc = st.slider('Probability of NoOscillation', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        no_trend = st.slider('Probability of NoTrend', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
        n_trans = st.slider('Probability of NoTransition', min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        st.subheader("Specify the Connections")
        # create all sliders we have to specify for the dependent signal
        dependency_list = [0]*independent_signals
        weights = get_weights(independent_signals)
        for idx in range(independent_signals):
            dependency_list[idx] = st.slider(f'Weight to sig {idx}', min_value=0.0, max_value=1.0, value=weights[idx], step=0.01)

        # make a button that creates the data
        if st.button('Simulate'):
            simulated_signals, weights = get_change_signals(independent_signals, dependency_list, allowed_oscillations,
                                                            allowed_trends, allowed_transitions, no_osc, no_trend,
                                                            n_trans, signal_length, event_rate)

    # check whether we have simulated signals
    if simulated_signals is None:
        st.header('⚠ Please press the "Simulate" button! ⚠')
        return

    # make a plot of the simulated signals
    st.header("Simulated Signals")
    fig = px.line(simulated_signals, x=simulated_signals.index, y=simulated_signals.columns)
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig)

    # make a radar chart showing the weights
    st.header("Dependent Signal Linking")
    fig = px.line_polar(weights, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)



if __name__ == '__main__':
    app()
