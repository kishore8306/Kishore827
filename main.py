import sys
import cv2
import torch
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
from collections import deque, Counter

print("Python Executable:", sys.executable)
print("Python Path:", sys.path)

# Load YOLO model
model = YOLO('yolov8n.pt')  

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize deque to store last 5 gender predictions for smoothing
predictions = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            height, width = frame.shape[:2]
            pad = 15  # Padding to include a bit more around the face

            # Ensure safe bounding box with padding
            x1 = max(0, int(box.xyxy[0][0]) - pad)
            y1 = max(0, int(box.xyxy[0][1]) - pad)
            x2 = min(width, int(box.xyxy[0][2]) + pad)
            y2 = min(height, int(box.xyxy[0][3]) + pad)
            confidence = box.conf[0]

            if confidence > 0.6:
                face = frame[y1:y2, x1:x2]

                try:
                    # Run DeepFace analysis on the cropped face (no model_name param)
                    gender_result = DeepFace.analyze(
                        face,
                        actions=['gender'],
                        detector_backend='retinaface',  # Optional but useful
                        enforce_detection=False
                    )

                    gender = gender_result[0]['dominant_gender']

                    # Add to deque for smoothing
                    predictions.append(gender)

                    # Get the most common gender from recent frames
                    most_common_gender = Counter(predictions).most_common(1)[0][0]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, most_common_gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error in gender classification: {e}")

    cv2.imshow("Gender Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
