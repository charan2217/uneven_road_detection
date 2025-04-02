# Uneven Road Detection

## Overview
This project aims to detect uneven roads, potholes, and major/minor cracks using a combination of **LiDAR**, **Ultrasonic Sensors**, and **Infrared Cameras**. The system processes real-time data and predicts road conditions using **Deep Learning** and **Machine Learning** models.

## Technologies Used
- **Python** (for data processing and ML model development)
- **TensorFlow/Keras** (for deep learning models)
- **OpenCV** (for image processing and analysis)
- **Pandas & NumPy** (for data manipulation)
- **Matplotlib & Seaborn** (for data visualization)
- **Scikit-learn** (for machine learning models)
- **Arduino/C++** (for sensor integration and real-time data collection)
- **RPLIDAR A2** (for LiDAR-based road scanning)
- **HC-SR04** (for ultrasonic sensor-based depth measurement)
- **Infrared Sensors** (for detecting fine cracks and texture variations)
- **YOLOv8** (for object detection and road condition classification)

## Project Structure
```
├── data/
│   ├── combined_sensor_data_cleaned.csv
│   ├── synthetic_infrared_data.csv
│   ├── synthetic_lidar_data.csv
│   ├── synthetic_ultrasonic_data.csv
├── models/
│   ├── model.sav
│   ├── train_lstm.py
│   ├── train_yolov8.py
├── scripts/
│   ├── HC-SR04.py
│   ├── RPLIDAR_A2_LiDAR.py
│   ├── road_detection.py
│   ├── road_detection1.py
│   ├── python_lstm_road_detection_sim.py
├── LICENSE
├── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/charan2217/uneven_road_detection.git
   cd uneven_road_detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the detection script:
   ```bash
   python road_detection.py
   ```

## Model Training
To train the LSTM model for road condition prediction:
```bash
python train_lstm.py
```

To train the YOLOv8 model for object detection:
```bash
python train_yolov8.py
```

## Contributors
- **Charan Parava Reddy**

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any queries, contact [paravacharanreddy4@gmail.com](mailto:paravacharanreddy4@gmail.com).
