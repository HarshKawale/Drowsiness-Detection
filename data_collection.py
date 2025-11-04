import cv2
import os
import time
import mediapipe as mp

# === Setup ===
save_path = "dataset"
labels = ["drowsy", "alert", "yawn", "no_yawn"]
for label in labels:
    os.makedirs(os.path.join(save_path, label), exist_ok=True)

# === MediaPipe Face Detection ===
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# === Webcam ===
cap = cv2.VideoCapture(0)
print("Keys: 'd' = DROWSY, 'a' = ALERT, 'y' = YAWN, 'n' = NO_YAWN, 'q' = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(frame_rgb)

    face_crop = None
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(iw, x + w), min(ih, y + h)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(face_crop, (224, 224))
            cv2.imshow("Face", face_crop)
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if face_crop is not None:
        timestamp = time.time()
        if key == ord("d"):
            path = f"{save_path}/drowsy/image_{timestamp}.jpg"
            cv2.imwrite(path, face_crop)
            print(f"Saved DROWSY: {path}")
        elif key == ord("a"):
            path = f"{save_path}/alert/image_{timestamp}.jpg"
            cv2.imwrite(path, face_crop)
            print(f"Saved ALERT: {path}")
        elif key == ord("y"):
            path = f"{save_path}/yawn/image_{timestamp}.jpg"
            cv2.imwrite(path, face_crop)
            print(f"Saved YAWN: {path}")
        elif key == ord("n"):
            path = f"{save_path}/no_yawn/image_{timestamp}.jpg"
            cv2.imwrite(path, face_crop)
            print(f"Saved NO_YAWN: {path}")

cap.release()
cv2.destroyAllWindows()
