import numpy as np
import io
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout

# RNN with LSTM model

data = pd.read_csv('NVDA.csv')
print()
print (data.head())
print(data.tail())
print()
data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()

training_data = data_training.drop(['Date', 'Adj Close'], axis = 1)
print(training_data.head()) 
print()


scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data)
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

print()
regressor = Sequential()
regressor.add(LSTM(units= 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units= 60, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units= 80, activation = 'relu',return_sequences = True))
regressor.add(Dropout(0.4))
regressor.add(LSTM(units= 120, activation = 'relu'))
regressor.add(Dropout(0.5))
regressor.add(Dense(units = 1))
print(regressor.summary())

regressor.compile(optimizer='adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs=10, batch_size=32)

print(data_test.head())
print()
data_training.tail(60)
past_60_days = data_training.tail(60)

dff = past_60_days._append(data_test, ignore_index = True)
dff = dff.drop(['Date', 'Adj Close'], axis = 1)
print(data_test.head())
inputs = scaler.transform(dff)
X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape                                                                                                                                                                                                  

# predict
y_pred = regressor.predict(X_test)
scaler.scale_
scale = 1/3.70274364e-03
print(scale)
print()
y_pred = y_pred*scale
y_test = y_test*scale

#plot
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'black', label = "Real NVDA Stock Price")
plt.plot(y_pred, color = 'gray', label = 'Predicted NVDA Stock Price')
plt.title('NVDA Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('NVDA Stock Price')
plt.legend()
plt.show()
