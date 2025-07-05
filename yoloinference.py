from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO("C:\Users\poorn\OneDrive\Desktop\IDP-AnimalModel\model_- 13 june 2025 19_35.pt")  # Example: "best.pt"

# Load an image for inference
img = "path_to_image.jpg"  # Replace with image path or use cv2.imread for OpenCV images

# Run inference
results = model(img)

# Show results
results[0].show()  # Opens a window with bounding boxes
results[0].save(filename='output.jpg')  # Saves the result

# To get raw prediction data
for box in results[0].boxes:
    cls_id = int(box.cls[0])  # Class index
    conf = float(box.conf[0])  # Confidence
    xyxy = box.xyxy[0].tolist()  # Bounding box coordinates
    print(f"Class: {cls_id}, Confidence: {conf:.2f}, BBox: {xyxy}")
