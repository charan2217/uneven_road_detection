import numpy as np
import pandas as pd

np.random.seed(42)
data_size = 5000

# Simulated LiDAR height data
x_coords = np.random.uniform(low=0, high=10, size=data_size)
y_coords = np.random.uniform(low=0, high=10, size=data_size)
z_height = np.random.uniform(low=0, high=5, size=data_size)  # Road surface variation

# Label: 0 = Smooth road, 1 = Rough road
labels = np.where(z_height > 2, 1, 0)  # If height difference > 2cm, classify as rough road

df = pd.DataFrame({"X": x_coords, "Y": y_coords, "Z_Height": z_height, "Road_Condition": labels})
df.to_csv("synthetic_lidar_data.csv", index=False)

print("Synthetic LiDAR data saved as 'synthetic_lidar_data.csv'")
