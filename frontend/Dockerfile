FROM python:3.11-slim
LABEL authors="lucas"

# install necessary python libraries while
RUN pip install changepoynt --no-deps
RUN pip install numpy scipy matplotlib pandas streamlit fbpca numba

# copy the app file into the container
COPY . ./

CMD python app.py