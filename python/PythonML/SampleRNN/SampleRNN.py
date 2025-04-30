import numpy as np
import io
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, Dropout
from keras.layers import Dense, LSTM, Dropout

# RNN sample
model = keras.models.Sequential([
        keras.layers.SimpleRNN\
        (1, input_shape=[None, 1])])

model = keras.models.Sequential\
        ([keras.layers.SimpleRNN\
        (20, return_sequences=True, input_shape=[None, 1]), \
        keras.layers.SimpleRNN(20, return_sequences=True), \
        keras.layers.SimpleRNN(1)])
