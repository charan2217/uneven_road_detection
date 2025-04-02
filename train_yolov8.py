from ultralytics import YOLO  

# Load YOLOv8 model (nano version for efficiency)
model = YOLO("yolov8n.pt")  

# Train the model with your dataset
model.train(data="road_damage.yaml", epochs=50, imgsz=640, batch=16)
