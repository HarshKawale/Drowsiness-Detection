import os
import cv2
import mediapipe as mp

DATASET_DIR = "dataset"
OUTPUT_DIR = "dataset_preprocessed"
IMG_SIZE = 64

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

EYE_POINTS_LEFT = [33, 133, 159, 145]
EYE_POINTS_RIGHT = [362, 263, 386, 374]
MOUTH_POINTS = [13, 14, 78, 308, 82, 312]

labels = ["open_eye", "closed_eye", "open_mouth", "closed_mouth"]
for lbl in labels:
    os.makedirs(os.path.join(OUTPUT_DIR, lbl), exist_ok=True)

def crop_region(image, landmarks, points, padding=10):
    h, w, _ = image.shape
    coords = [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points]
    x_min = max(min([x for x, y in coords]) - padding, 0)
    x_max = min(max([x for x, y in coords]) + padding, w)
    y_min = max(min([y for x, y in coords]) - padding, 0)
    y_max = min(max([y for x, y in coords]) + padding, h)
    return image[y_min:y_max, x_min:x_max]

for cls in os.listdir(DATASET_DIR):
    input_path = os.path.join(DATASET_DIR, cls)
    if not os.path.isdir(input_path):
        continue

    print(f"Processing class: {cls}")
    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        landmarks = results.multi_face_landmarks[0].landmark

        if cls == "alert":  # eyes open
            for idx, points in enumerate([EYE_POINTS_LEFT, EYE_POINTS_RIGHT]):
                eye = crop_region(img, landmarks, points, padding=20)
                if eye.size > 0:
                    eye_resized = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
                    eye_gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
                    eye_gray = cv2.GaussianBlur(eye_gray, (3, 3), 0)
                    out_name = f"{img_name.split('.')[0]}_eye{idx}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "open_eye", out_name), eye_gray)

        elif cls == "drowsy":  # eyes closed
            for idx, points in enumerate([EYE_POINTS_LEFT, EYE_POINTS_RIGHT]):
                eye = crop_region(img, landmarks, points, padding=20)
                if eye.size > 0:
                    eye_resized = cv2.resize(eye, (IMG_SIZE, IMG_SIZE))
                    eye_gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
                    eye_gray = cv2.GaussianBlur(eye_gray, (3, 3), 0)
                    out_name = f"{img_name.split('.')[0]}_eye{idx}.png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "closed_eye", out_name), eye_gray)

        elif cls == "yawn":  # mouth open
            mouth = crop_region(img, landmarks, MOUTH_POINTS, padding=40)
            if mouth.size > 0:
                mouth_resized = cv2.resize(mouth, (IMG_SIZE, IMG_SIZE))
                mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
                mouth_gray = cv2.GaussianBlur(mouth_gray, (3, 3), 0)
                out_name = f"{img_name.split('.')[0]}_mouth.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "open_mouth", out_name), mouth_gray)

        elif cls == "no_yawn":  # mouth closed
            mouth = crop_region(img, landmarks, MOUTH_POINTS, padding=40)
            if mouth.size > 0:
                mouth_resized = cv2.resize(mouth, (IMG_SIZE, IMG_SIZE))
                mouth_gray = cv2.cvtColor(mouth_resized, cv2.COLOR_BGR2GRAY)
                mouth_gray = cv2.GaussianBlur(mouth_gray, (3, 3), 0)
                out_name = f"{img_name.split('.')[0]}_mouth.png"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "closed_mouth", out_name), mouth_gray)

print("✅ Done. Cropped images saved in dataset_preprocessed/")
