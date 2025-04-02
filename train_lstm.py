import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# Load collected sensor data
data = pd.read_csv("combined_sensor_data_cleaned.csv")  

# Preprocess data
X_train, Y_train = data.iloc[:, :-1], data.iloc[:, -1]

# Create LSTM model
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Predict if road is rough (1) or smooth (0)
])

# Compile and train
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=50, batch_size=16)
model.save("road_condition_lstm.h5")  # Saves the model as 'road_condition_lstm.h5'
