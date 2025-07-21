import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sensor_acquisition import SensorAcquisition
from sensor_fusion import SensorFusion
import time

# User-configurable mode: 'mock' for simulation, 'real' for hardware
MODE = "mock"  # Change to 'real' when hardware is available

# Step 1: Acquire sensor data (batch for training, single for real-time)
def get_sensor_data(mode, n_samples=500):
    acq = SensorAcquisition(mode=mode)
    return acq.acquire_all(n_samples=n_samples)

# Step 2: Fuse and normalize sensor data
def fuse_data(raw_data):
    fusion = SensorFusion(normalization=True)
    return fusion.fuse(raw_data)

# Step 3: Prepare data for LSTM
def prepare_lstm_data(fused_data, labels):
    X = fused_data.values
    Y = labels
    X = np.expand_dims(X, axis=1)  # LSTM expects 3D input
    return X, Y

# Step 4: Build LSTM model
def build_lstm_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    # TRAINING PHASE (simulation)
    raw_data = get_sensor_data(MODE, n_samples=500)
    fused_data = fuse_data(raw_data)
    X, Y = prepare_lstm_data(fused_data, raw_data['Road_Condition'].values)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, Y, epochs=50, batch_size=16)
    model.save("road_hazard_lstm_model.h5")
    fused_data['Road_Condition'] = Y
    fused_data.to_csv("fused_sensor_data.csv", index=False)

    # REAL-TIME INFERENCE PHASE (can be simulated)
    print("\n--- Real-time Hazard Detection (Simulated) ---")
    for _ in range(10):
        test_data = get_sensor_data(MODE, n_samples=1)
        fused_test = fuse_data(test_data)
        X_test, _ = prepare_lstm_data(fused_test, [0])  # label dummy
        pred = model.predict(X_test)[0][0]
        hazard = pred > 0.5
        print(f"Hazard Probability: {pred:.2f} | Hazard Detected: {hazard}")
        # Simulated vehicle response
        if hazard:
            print("[ALERT] Road hazard detected! Slowing down vehicle and triggering alert.")
        else:
            print("Road is smooth. Proceeding at normal speed.")
        time.sleep(1)
