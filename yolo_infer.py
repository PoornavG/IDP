from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best100.pt")  # Replace with your model path

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run inference
    results = model(frame)

    # Annotate image (bounding boxes, labels, etc.)
    annotated_frame = results[0].plot()  # Automatically draws boxes and labels

    # Display result
    cv2.imshow("YOLO Inference", annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
