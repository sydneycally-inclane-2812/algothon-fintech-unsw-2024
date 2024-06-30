import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
# Load the data
file_path = 'prices.txt'
data = pd.read_csv(file_path, sep=r"\s+", header=None)
# Display the first few rows of the dataframe to understand its structure
data.head()


def plot_instrument(data, column_index):
    # Plotting the selected instrument's prices
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data[column_index], label=f'Instrument {column_index + 1} Prices')
    plt.title(f'Price Trend for Instrument {column_index + 1} Over 500 Days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example of plotting data for the second instrument (column 1)
plot_instrument(data, 1)

# prepare data
prices = data[1].values  # Prices of the second instrument

# normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices = scaler.fit_transform(prices.reshape(-1, 1))

# create sequences
def create_sequences(data, n):
    X, y = [], []
    for i in range(n, len(data)):
        X.append(data[i-n:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

n_steps = 10  # Number of days to use for prediction
X, y = create_sequences(prices, n_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM

# Split data to training data and testing data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM Model


model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train and evaluate the model
model.fit(X_train, y_train, epochs=50, batch_size=32)
y_pred = model.predict(X_test)
# Optionally, invert scaling to compare against original prices
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_rescaled = scaler.inverse_transform(y_pred)




