import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt


data = pd.read_csv("household_power_consumption.csv")
data['Date'] = pd.to_datetime(data['Date'], format="%d/%m/%Y")
data['Datetime'] = data['Date'].dt.strftime('%Y-%m-%d') + ' ' + data['Time']
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.sort_values(['Datetime'])

# num_cols -> have numeric values 
num_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

#Convert 
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#missing values
for col in num_cols:
    data[col].fillna(data[col].mean(), inplace=True)

dff = data.drop(['Date', 'Time', 'Global_reactive_power', 'Datetime'], axis = 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dff)
X = []
y = []

#training dataset
for i in range(60, scaled_data.shape[0]):
    X.append(scaled_data [i-60:i])
    y.append(scaled_data [i, 0])
X, y = np.array(X), np.array(y)

#train and test sets by spliting given index 
X_train = X[:217440]
y_train = y[:217440]
X_test = X[217440:]
y_test = y[217440:]

# neural network, LSTM layers with given units.
regressor = Sequential()
regressor.add(LSTM(units= 20, activation = 'relu', return_sequences = True,input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units= 40, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units= 80, activation = 'relu'))
regressor.add(Dropout(0.5))
regressor.add(Dense(units = 1))

#compile()  -> configure  training model 
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

#fit and predict
regressor.fit(X_train, y_train, epochs=2, batch_size=32)
y_pred = regressor.predict(X_test)

plt.figure(figsize=(15,8))
plt.plot(y_test[-60:], color = 'black', label = "Real Power Consumption")
plt.plot(y_pred[-60:], color = 'gray', label = 'Predicted Power Consumption')
plt.title('Power Consumption Prediction')
plt.xlabel('time')
plt.ylabel('Power Consumption')
plt.legend()
plt.savefig("B16341_09_44.png", dpi=200)
plt.show()