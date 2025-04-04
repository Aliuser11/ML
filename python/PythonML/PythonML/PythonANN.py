import numpy as np
import io
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ANN model

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
regressor_ann = Sequential()
regressor_ann.add(Input(shape = (300,)))
regressor_ann.add(Dense(units = 512, activation = 'relu'))
regressor_ann.add(Dropout(0.2))

regressor_ann.add(Dense(units = 128, activation = 'relu'))
regressor_ann.add(Dropout(0.3))

regressor_ann.add(Dense(units = 64, activation = 'relu'))
regressor_ann.add(Dropout(0.4))

regressor_ann.add(Dense(units = 16, activation = 'relu'))
regressor_ann.add(Dropout(0.5))

regressor_ann.add(Dense(units = 1))

regressor_ann.summary()
print(regressor_ann.summary())
regressor_ann.compile(optimizer='adam', \
loss = 'mean_squared_error')
regressor_ann.fit(X_train, y_train, epochs=10, batch_size=32)
data_test.head()
print(data_test.head())
print()
past_60_days = data_training.tail(60)
dff = past_60_days._append(data_test, ignore_index = True)
dff = dff.drop(['Date', 'Adj Close'], axis = 1)
inputs = scaler.transform(dff)
X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_old_shape = X_test.shape
X_test = X_test.reshape(X_old_shape[0], \
X_old_shape[1] * X_old_shape[2])
X_test.shape, y_test.shape
y_pred = regressor_ann.predict(X_test)
scaler.scale_
scale = 1/3.70274364e-03
print(scale)
print()
y_pred = y_pred*scale
y_test = y_test*scale
plt.plot(y_test, color = 'black', label = "Real NVDA Stock Price")
plt.plot(y_pred, color = 'gray',\
label = 'Predicted NVDA Stock Price')
plt.title('NVDA Stock Price Prediction')
plt.xlabel('time')
plt.ylabel('NVDA Stock Price')
plt.legend()
plt.show()