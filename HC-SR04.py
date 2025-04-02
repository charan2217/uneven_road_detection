import pandas as pd

# Load all datasets
ultrasonic_data = pd.read_csv("synthetic_ultrasonic_data.csv")  
lidar_data = pd.read_csv("synthetic_lidar_data.csv")  
infrared_data = pd.read_csv("synthetic_infrared_data.csv")

# Ensure they have the same number of rows
print(len(ultrasonic_data), len(lidar_data), len(infrared_data))  # Should all be 5000

# Merge datasets column-wise
combined_data = pd.concat([ultrasonic_data, lidar_data.drop(columns=["Road_Condition"])], axis=1)
combined_data = pd.concat([combined_data, infrared_data.drop(columns=["Road_Condition"])], axis=1)

# Ensure only one "Road_Condition" column is present
combined_data.to_csv("combined_sensor_data.csv", index=False)

print("Combined dataset saved as 'combined_sensor_data.csv'")
