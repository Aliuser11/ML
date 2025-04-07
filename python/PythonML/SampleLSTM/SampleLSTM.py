
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

#lstm
data = pd.read_csv('NVDA.csv')
print()
print (data.head())
print()
data_training = data[data['Date']<'2019-01-01'].copy()

data_test = data[data['Date']>='2019-01-01'].copy()

training_data = data_training.drop(['Date', 'Adj Close'], axis = 1)
print(training_data.head())
print()
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
training_data
print(training_data)
print()
X_train = []
y_train = []
training_data.shape[0]

for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape, y_train.shape #((808, 60, 5), (808,))

print(X_train.shape, y_train.shape)
X_old_shape = X_train.shape
X_train = X_train.reshape(X_old_shape[0], \
X_old_shape[1]*X_old_shape[2])
print(X_train.shape)
print()
regressor = Sequential()
regressor.add(LSTM(units= 50, activation = 'relu', \
return_sequences = True, \
input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 60, activation = 'relu', \
return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units= 80, activation = 'relu', \
return_sequences = True))
regressor.add(Dropout(0.4))
regressor.add(LSTM(units= 120, activation = 'relu'))
regressor.add(Dropout(0.5))
regressor.add(Dense(units = 1))
regressor = Sequential()
print(regressor.summary())


