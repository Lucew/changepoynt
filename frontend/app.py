import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST
from changepoynt.algorithms.fluss import FLUSS
from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.visualization.score_plotting import plot_data_and_score

# make a global dict for the algorithms
ALGS = {'IKA-SST': SST, 'ESST': ESST, 'FLUSS': FLUSS}


# Define a function to apply a running mean filter to a signal
def transform(transformer: Algorithm, signal: np.ndarray):

    # filter the signal with the algorithm
    with st.spinner('Wait for the algorithm to run...'):
        result = transformer.transform(signal)

        # plot the result of the algorithm
        fig = plot_data_and_score(signal, result)
    return fig


# Define the Streamlit app
def app():
    # Set the app title
    st.title('Change Point Detection App.')

    # Add a file uploader component to the app
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is None: return

    # Read the file contents into a Pandas dataframe
    df = pd.read_csv(uploaded_file)

    # check make a selection for the columns we have available
    column = st.sidebar.selectbox('Select CSV column with the signal in it', df.columns)
    if not column: return

    # check whether the column contains float values
    if not df[column].apply(lambda x: isinstance(x, float)).all():
        st.error('Selected CSV column could not be parsed as floats.', icon="🚨")
        return

    # Add a sidebar for selecting the filter type
    algorithm = st.sidebar.selectbox('Select Algorithm', list(ALGS.keys()))

    # add a slider for the window size
    window_size = st.slider("Select the window size", min_value=5, max_value=min(400, df.shape[0]//5), value=50, step=1)

    # add a slider for the step size
    if algorithm == 'ESST' or algorithm == 'IKA-SST':
        step_size = st.slider("Select the step size", min_value=1, max_value=100, value=10, step=2)
        transformer = ALGS[algorithm](window_length=window_size, scoring_step=step_size)
    else:
        transformer = ALGS[algorithm](window_size)

    # make a button to run the algorithm
    placeholder = st.empty()
    run_algorithm = placeholder.button("Run.", key=1)
    if run_algorithm:
        placeholder.button('Run.', disabled=True, key=2)
        result = transform(transformer, df[column].values)
        placeholder.button('Run.', disabled=False, key=3)
    else:
        return
    st.pyplot(result)


if __name__ == '__main__':
    app()
