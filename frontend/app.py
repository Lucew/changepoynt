import streamlit as st
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

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
import requests
import os


# make a global dict for the algorithms
ALGS = {'ESST': ESST, 'IKA-SST': SST, 'FLUSS': FLUSS, "CLASP": CLASP}


# Define a function to apply a running mean filter to a signal
def transform(transformer: Algorithm, signal: np.ndarray):

    # filter the signal with the algorithm
    with st.spinner('Wait for the algorithm to run...'):
        score = transformer.transform(signal)

        # plot the result of the algorithm
        fig = plot_data_and_score(signal, score)
    return fig, score


def get_remote_ip() -> str:
    """Get remote ip."""

    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return ""

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return ""
    except Exception as e:
        return ""

    return session_info.request.remote_ip


@st.cache_data
def create_example():

    # read the remote csv from the repository
    # the raw file is from: ECML/PKDD 2023 Human Activity Segmentation Challenge
    # https://github.com/patrickzib/human_activity_segmentation_challenge
    modulation = pd.read_csv("https://raw.githubusercontent.com/Lucew/changepoynt/master/frontend/example.csv")
    modulation = modulation["Signal"].values

    # make some modulation on the time series
    offset = 12
    df = pd.DataFrame(np.random.rand(modulation.shape[0], 4), columns=["Signal", "RandA", "RandB", "RandC"])
    x = np.arange(0, modulation.shape[0], 1, dtype=float)
    signal = np.ones_like(x) * offset
    end_index = int(signal.shape[0] * 0.65)
    signal[end_index:] = np.exp(np.log(1/offset)/(signal.shape[0]*0.7-end_index) * (x[end_index:]-end_index)) * offset
    signal[signal < 0] = 0
    signal += modulation
    df["Signal"] = signal
    return df


def telegram_bot_sendtext(bot_message):
    bot_token = os.environ.get("BOT_TOKEN")
    bot_chat_id = os.environ.get("CHAT_ID")
    if bot_token is None or bot_chat_id is None:
        st.error("Bot Token or Chat ID have not been set as environment variables.")
        return
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chat_id \
                + '&parse_mode=Markdown&text=' + bot_message

    # send the request
    try:
        response = requests.get(send_text)
    except requests.exceptions.ConnectionError:
        return ""

    # decode the request if we have one
    try:
        response = response.json()
    except requests.exceptions.JSONDecodeError:
        response = None
    return response


def create_score_download(signal: np.ndarray, score: np.ndarray):
    new = pd.DataFrame({"Signal": signal, "Changepoint-Score": score})
    return new.to_csv()


# Define the Streamlit app
def app():
    # Set the app title
    st.title('Change Point Detection App.')

    # get the ip of the user
    ip = str(get_remote_ip())

    # log the usage of our app
    if ip not in st.session_state:
        # send a telegram message
        telegram_bot_sendtext(f"We have a new user in our streamlit app: {ip}")

        # set the ip to usage
        st.session_state[ip] = True

    # Add a file uploader component to the app
    st.write("Caution with Data uploading! The connection might not be encrypted and the server is not secure.")
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

    # plot the signal
    fig, ax = plt.subplots(figsize=(20, 1))
    ax.plot(df[column].values)
    ax.axis('off')
    st.write("Minimalistic shape of your time series:")
    st.pyplot(fig)

    # Add a sidebar for selecting the filter type
    algorithm = st.sidebar.selectbox('Select Algorithm', list(ALGS.keys()))

    # add a slider for the window size
    sampling = st.slider("Select the Down-Sampling", min_value=1, max_value=100, value=1, step=1)

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

        # start time measurement
        start = time.time()

        # disable the run button while running the code
        placeholder.button('Run.', disabled=True, key=2)

        # create the signal from the dataframe
        signal = df[column].values[::sampling]

        # compute the change point score result
        figure, score= transform(transformer, signal)

        # enable the run button again
        placeholder.button('Run.', disabled=False, key=3)

        # check the elapsed time
        duration = time.time()-start

        # output the time it took to make the complete transformation
        st.write(f"Running the algorithm took: {duration:0.3f} s.")

        # plot the result onto the page
        st.pyplot(figure)

        btn = st.download_button(
            label="Download Score",
            data=create_score_download(signal, score),
            file_name=f"changepoint_score_{algorithm}_{column}.csv",
            mime="text/csv"
        )


if __name__ == '__main__':
    app()
