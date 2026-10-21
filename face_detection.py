"""
Task 1: Real-time Face Detection using OpenCV (Haar Cascade)
Features:
  - Webcam par live face detection
  - Face count display
  - 's' dabao -> detected faces crop karke save + attendance_log.csv mein entry
  - 'q' dabao -> exit
"""
import csv
import os
from datetime import datetime

import cv2

SAVE_DIR = "detected_faces"
LOG_FILE = "attendance_log.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# Haar Cascade model load
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Haar Cascade file load nahi hui.")

# CSV header (sirf pehli baar)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "faces_detected", "saved_files"])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam open nahi hui. Camera check karo.")

print("Controls: 's' = save faces + log attendance, 'q' = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "s: save | q: quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and len(faces) > 0:
        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        saved = []
        for i, (x, y, w, h) in enumerate(faces):
            path = os.path.join(SAVE_DIR, f"face_{stamp}_{i}.jpg")
            cv2.imwrite(path, frame[y:y + h, x:x + w])
            saved.append(path)
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow(
                [now.strftime("%Y-%m-%d %H:%M:%S"), len(faces), ";".join(saved)]
            )
        print(f"{len(faces)} face(s) saved, attendance logged.")

cap.release()
cv2.destroyAllWindows()
