import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Replace with your camera's IP or RTSP stream URL
url = "url = "rtsp:/username:password@192.72.1.101:802/stream"
"

# Set timeout option
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to avoid long delays

if not cap.isOpened():
    print("Camera not detected or wrong URL!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Retrying...")
        break

    results = model(frame)

    for result in results:
        annotated_frame = result.plot()
        cv2.imshow("Agaro Camera Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
