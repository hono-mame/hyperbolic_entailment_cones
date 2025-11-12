# Base: TensorFlow 1.15.5 (CPU, Python 3)
FROM tensorflow/tensorflow:1.15.5-py3

WORKDIR /app

RUN pip3 install numpy joblib click plotly prettytable smart_open autograd
RUN pip install gensim
RUN pip install "smart_open>=4.0.0,<5.0.0"
RUN pip install dataclasses