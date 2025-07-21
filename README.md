# 🚗 Uneven Road Detection System

> Real-time road hazard detection using Sensor Fusion + Deep Learning  
> Built by [Parava Charan Reddy](mailto:paravacharanreddy4@gmail.com) | [LinkedIn](https://linkedin.com/in/charan-parava-b4b164232) | [GitHub](https://github.com/charan2217)

---

## 📌 Project Overview

AI-powered system to detect **potholes, cracks, bumps**, and **surface anomalies** using:

- 🔄 Sensor Fusion (LiDAR + Ultrasonic + Infrared)
- 🧠 YOLOv8 + LSTM for real-time detection
- 📍 Extended Kalman Filter (EKF) for location/velocity tracking
- 🖥️ Visual Simulation + Alert system

This project bridges robotics, computer vision, and real-world embedded sensing — ideal for automotive AI and SLAM-based navigation.

---

## 🧠 Features

- ✅ Fuses 3 sensor streams to detect uneven roads
- ✅ Object detection via **YOLOv8**
- ✅ Route learning using **LSTM**
- ✅ Position estimation with **IMU + EKF**
- ✅ Real-time road warnings (console / GUI alerts)
- ✅ Extendable to **cloud hazard sharing networks**

---

## 🔧 Tech Stack

| Category       | Tools/Technologies |
|----------------|--------------------|
| Languages      | Python, Arduino (C++) |
| Libraries      | OpenCV, TensorFlow, Keras, Scikit-learn |
| Models         | YOLOv8, LSTM, EKF |
| Sensors        | RPLIDAR A2, HC-SR04, IR sensors |
| Visualization  | Matplotlib, Seaborn |
| Tools          | Git, GitHub |

---

## 🗂️ Project Structure


---

## ⚙️ Run Instructions

```bash
# 1. Clone the repo
git clone https://github.com/charan2217/uneven_road_detection.git
cd uneven_road_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run detection
python road_detection.py

# Optional: Run model training
python train_lstm.py         # LSTM
python train_yolov8.py       # YOLOv8



| Sensor     | Role                              |
| ---------- | --------------------------------- |
| LiDAR      | Detect surface depth variation    |
| Ultrasonic | Spot potholes & minor elevations  |
| Infrared   | Detect cracks or reflectance dips |



👨‍💻 About Me
Parava Charan Reddy
📧 paravacharanreddy4@gmail.com
🌐 GitHub • LinkedIn
