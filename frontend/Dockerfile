FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries while
RUN pip install changepoynt --no-deps
RUN pip install claspy numpy scipy matplotlib pandas streamlit fbpca numba stumpy

# workraound according to: https://github.com/streamlit/streamlit/issues/4842
WORKDIR /app

# copy the app file into the container
COPY . ./
RUN ls
CMD streamlit run app.py