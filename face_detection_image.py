"""
Image par face detection.
Use: python face_detection_image.py photo.jpg
Output: output.jpg
"""
import sys

import cv2

if len(sys.argv) < 2:
    print("Use: python face_detection_image.py <image_path>")
    sys.exit(1)

img = cv2.imread(sys.argv[1])
if img is None:
    print("Image nahi mili. Path check karo.")
    sys.exit(1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.putText(img, f"Faces: {len(faces)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("output.jpg", img)
print(f"{len(faces)} face(s) detected. Result: output.jpg")
