import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Load datasets
neutral_df = pd.read_csv("ntdtset.txt")
resting_df = pd.read_csv("resting.txt")
holding_df = pd.read_csv("holding.txt")
gripping_df = pd.read_csv("gripping.txt")

# Ensure consistent number of features across datasets
min_features = min(neutral_df.shape[1], resting_df.shape[1], holding_df.shape[1], gripping_df.shape[1])

neutral_df = neutral_df.iloc[:, :min_features]
resting_df = resting_df.iloc[:, :min_features]
holding_df = holding_df.iloc[:, :min_features]
gripping_df = gripping_df.iloc[:, :min_features]

X = []
y = []
no_of_timesteps = 20

# Prepare data for LSTM
datasets = neutral_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

datasets = resting_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)

datasets = holding_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(2) 

datasets = gripping_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(3)  

# Convert lists to NumPy arrays
X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))  

# Compile model
model.compile(optimizer="adam", metrics=["accuracy"], loss="sparse_categorical_crossentropy")

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("lstm-hand-grasping.h5")
