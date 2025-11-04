import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
BASE_DIR = "/kaggle/input/drowsiness-dataset/train"

# =====================
# Load Eye Images (Closed/Open)
# =====================
def load_eye_images(path):
    X_eye, y_eye = [], []
    for label, folder in enumerate(['Closed', 'Open']):
        folder_path = os.path.join(path, folder)
        for fname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            X_eye.append(img_rgb)
            y_eye.append(label)
    return np.array(X_eye, dtype=np.float32), np.array(y_eye)

# =====================
# Load Yawn Images (yawn/no_yawn)
# =====================
def load_yawn_images(path):
    X_yawn, y_yawn = [], []
    for label, folder in enumerate(['no_yawn', 'yawn']):  # 0: no_yawn, 1: yawn
        folder_path = os.path.join(path, folder)
        for fname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            X_yawn.append(img_rgb)
            y_yawn.append(label)
    return np.array(X_yawn, dtype=np.float32), np.array(y_yawn)

print("Loading eye dataset...")
X_eye, y_eye = load_eye_images(BASE_DIR)
X_eye = X_eye / 255.0

print("Loading yawn dataset...")
X_yawn, y_yawn = load_yawn_images(BASE_DIR)
X_yawn = X_yawn / 255.0

# =====================
# Split & Augment
# =====================
X_eye_train, X_eye_val, y_eye_train, y_eye_val = train_test_split(
    X_eye, y_eye, test_size=0.2, stratify=y_eye, random_state=42
)
X_yawn_train, X_yawn_val, y_yawn_train, y_yawn_val = train_test_split(
    X_yawn, y_yawn, test_size=0.2, stratify=y_yawn, random_state=42
)


aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# =====================
# CNN Architecture (shared for both models)
# =====================
def create_cnn():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =====================
# Train Eye State Model
# =====================
eye_model = create_cnn()
print(eye_model.summary())
eye_model.fit(
    aug.flow(X_eye_train, y_eye_train, batch_size=32),
    epochs=15,
    validation_data=(X_eye_val, y_eye_val),
    verbose=1
)
eye_model.save("/kaggle/working/eye_model.keras")
print("✅ Eye state model saved as eye_model.keras")

# =====================
# Train Yawn State Model
# =====================
yawn_model = create_cnn()
print(yawn_model.summary())
yawn_model.fit(
    aug.flow(X_yawn_train, y_yawn_train, batch_size=32),
    epochs=15,
    validation_data=(X_yawn_val, y_yawn_val),
    verbose=1
)
yawn_model.save("/kaggle/working/yawn_model.keras")
print("✅ Yawn state model saved as yawn_model.keras")





# Yawn model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset directory containing 'train' folder with 'yawn' and 'no_yawn' subfolders
base_dir = '/kaggle/input/yawn-dataset/'

# Image parameters
IMG_SIZE = 64
BATCH_SIZE = 32

# Data augmentation and preprocessing with 20% validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    shear_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

# Training data generator
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Validation data generator
val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the model
EPOCHS = 20

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Save the trained model with a standard keras extension
model.save('yawn_model.h5')
