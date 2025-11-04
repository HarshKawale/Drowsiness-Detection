import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_DIR = "drowsiness_dataset/train"
OUTPUT_DIR = "dataset_augmented"
IMG_SIZE = 64
TARGET_COUNT = 100

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

for cls in os.listdir(INPUT_DIR):
    class_path = os.path.join(INPUT_DIR, cls)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(OUTPUT_DIR, cls)
    os.makedirs(output_class_path, exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.endswith((".png", ".jpg", ".jpeg"))]
    count = len(images)

    if count == 0:
        continue

    idx = 0
    while count < TARGET_COUNT:
        img_name = images[idx % len(images)]
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = np.expand_dims(img, axis=-1)  # shape: (64, 64, 1)
        img = np.expand_dims(img, axis=0)   # shape: (1, 64, 64, 1)

        aug_iter = datagen.flow(img, batch_size=1)
        aug_img = next(aug_iter)[0].astype(np.uint8)

        out_name = f"{os.path.splitext(img_name)[0]}_aug{count}.png"
        cv2.imwrite(os.path.join(output_class_path, out_name), aug_img)

        count += 1
        idx += 1

print("✅ Augmentation complete. Balanced dataset saved in 'dataset_augmented'")
