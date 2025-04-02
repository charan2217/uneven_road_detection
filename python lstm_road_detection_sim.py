import time  
import tensorflow as tf  
import numpy as np  

# Load LSTM model (Ensure lstm_model.h5 exists)
lstm_model = tf.keras.models.load_model("road_condition_lstm.h5")

# Compile the model to remove warnings (Only needed if you plan to train/evaluate)
lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

def get_fake_distance():
    """Simulate sensor data using random values"""
    return np.random.uniform(5, 100)  # Random distance between 5cm and 100cm

while True:
    distance = get_fake_distance()  # Use simulated distance

    # Convert distance into a NumPy array with correct shape
    distance_input = np.array([[distance]]).astype(np.float32)  

    # Make prediction
    prediction = lstm_model.predict(distance_input)

    if prediction > 0.5:  # Uneven road detected
        print(f"🚨 ALERT: Rough road detected at {distance:.2f} cm! Slowing down the vehicle.")
    else:
        print(f"✅ Smooth road detected at {distance:.2f} cm. No action needed.")

    time.sleep(1)
