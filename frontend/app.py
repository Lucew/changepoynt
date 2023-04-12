import setuptools.command.egg_info
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from changepoynt.algorithms.sst import SST
from changepoynt.algorithms.esst import ESST
from changepoynt.algorithms.fluss import FLUSS
from changepoynt.algorithms.clasp import CLASP
from changepoynt.algorithms.base_algorithm import Algorithm
from changepoynt.visualization.score_plotting import plot_data_and_score
import time

# make a global dict for the algorithms
ALGS = {'ESST': ESST, 'IKA-SST': SST, 'FLUSS': FLUSS, "CLASP": CLASP}


# Define a function to apply a running mean filter to a signal
def transform(transformer: Algorithm, signal: np.ndarray):

    # filter the signal with the algorithm
    with st.spinner('Wait for the algorithm to run...'):
        result = transformer.transform(signal)

        # plot the result of the algorithm
        fig = plot_data_and_score(signal, result)
    return fig


def create_example():
    df = pd.DataFrame(np.random.rand(2000, 4), columns=list('ABCD'))
    return df


# Define the Streamlit app
def app():
    # Set the app title
    st.title('Change Point Detection App.')

    # Add a file uploader component to the app
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is None:

        # write a text
        st.write("If you have no idea yet on what files to upload you can use our example file (currently loaded).")

        # make a button to use the example
        df = create_example()

        # make a download button for an example
        st.download_button(
            "Download Example",
            df.to_csv(index=False).encode('utf-8'),
            "file.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        # Read the file contents into a Pandas dataframe
        df = pd.read_csv(uploaded_file)

    # check make a selection for the columns we have available
    column = st.sidebar.selectbox('Select CSV column with the signal in it', df.columns)
    if not column: return

    # check whether the column contains float values
    if not df[column].apply(lambda x: isinstance(x, float)).all():
        st.error('Selected CSV column could not be parsed as floats. '
                 'Please select and appropriate column in the sidebar on the left.', icon="🚨")
        st.write('Quick visualization of the CSV-File you uploaded:')
        return

    # Add a sidebar for selecting the filter type
    algorithm = st.sidebar.selectbox('Select Algorithm', list(ALGS.keys()))

    # add a slider for the window size
    sampling = st.slider("Select the Down-sampling", min_value=1, max_value=100, value=1, step=1)

    # add a slider for the step size
    if algorithm == 'ESST' or algorithm == 'IKA-SST':
        step_size = st.slider("Select the Step Size", min_value=1, max_value=100, value=10, step=2)
        # add a slider for the window size
        window_size = st.slider("Select the Window Size", min_value=5, max_value=min(400, df.shape[0] // 5), value=50,
                                step=1)
        transformer = ALGS[algorithm](window_length=window_size, scoring_step=step_size)
    elif algorithm == 'CLASP':
        transformer = ALGS[algorithm]()
    else:
        # add a slider for the window size
        window_size = st.slider("Select the Window Size", min_value=5, max_value=min(400, df.shape[0] // 5), value=50,
                                step=1)
        transformer = ALGS[algorithm](window_size)

    # write the description of the algorithm as scrollable text
    st.sidebar.write(transformer.__doc__)

    # make a button to run the algorithm
    placeholder = st.empty()
    run_algorithm = placeholder.button("Run.", key=1)
    if run_algorithm:
        start = time.time()
        placeholder.button('Run.', disabled=True, key=2)
        result = transform(transformer, df[column].values[::sampling])
        placeholder.button('Run.', disabled=False, key=3)
        duration = time.time()-start
        st.write(f"Running the algorithm took: {duration:0.3f} s.")
        st.pyplot(result)


if __name__ == '__main__':
    app()
