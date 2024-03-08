FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries while
RUN pip install changepoynt --no-deps
RUN pip install claspy numpy scipy matplotlib pandas streamlit fbpca numba stumpy

# copy the app file into the container
COPY . ./

CMD streamlit run ./frontend/app.py