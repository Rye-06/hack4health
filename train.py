import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# -----------------------------
# Constants
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20

# -----------------------------
# Load metadata (ONE ROW PER CLASS)
# -----------------------------
df = pd.read_csv("./dataset/metadata.csv")

label_map = {
    "non_dem": "NonDemented",
    "very_mild_dem": "VeryMildDemented",
    "mild_dem": "MildDemented",
    "moderat_dem": "ModerateDemented"
}

df["folder_label"] = df["label"].map(label_map)

# Normalize age
df["age"] = df["age"] / 100.0

# Build metadata lookup: label -> metadata
label_to_metadata = {
    row.folder_label: np.array(
        [row.age, row.sex, row.family_history, row.apoe],
        dtype=np.float32
    )
    for _, row in df.iterrows()
}

# -----------------------------
# Load ALL images per folder
# -----------------------------
def load_images_from_label(label):
    directory = Path(f"./dataset/train/{label}")
    images = []

    for file_path in directory.iterdir():
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(file_path))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)

    return images

# -----------------------------
# Build dataset arrays
# -----------------------------
images = []
metadata = []
labels = []

class_map = {
    "NonDemented": 0,
    "VeryMildDemented": 1,
    "MildDemented": 2,
    "ModerateDemented": 3
}

for folder_label, class_idx in class_map.items():
    imgs = load_images_from_label(folder_label)
    meta = label_to_metadata[folder_label]

    for img in imgs:
        images.append(img)
        metadata.append(meta)
        labels.append(class_idx)

images = np.array(images, dtype=np.float32)
metadata = np.array(metadata, dtype=np.float32)
labels = to_categorical(labels, 4)

print("Images:", images.shape, images.dtype)
print("Metadata:", metadata.shape, metadata.dtype)
print("Labels:", labels.shape)

# -----------------------------
# Class imbalance correction
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(4),
    y=np.argmax(labels, axis=1)
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# Model: CNN branch
# -----------------------------
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

cnn_base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_tensor=image_input
)
cnn_base.trainable = False

x = GlobalAveragePooling2D()(cnn_base.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

# -----------------------------
# Model: Metadata branch
# -----------------------------
meta_input = Input(shape=(4,))
y = Dense(32, activation="relu")(meta_input)
y = Dense(16, activation="relu")(y)

# -----------------------------
# Fusion
# -----------------------------
combined = Concatenate()([x, y])
z = Dense(128, activation="relu")(combined)
z = Dropout(0.5)(z)
output = Dense(4, activation="softmax")(z)

model = Model(inputs=[image_input, meta_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------
model.fit(
    [images, metadata],
    labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weights,
    validation_split=0.2,
    shuffle=True
)

model.save("multimodal_alzheimer_model.h5")
print("Model saved: multimodal_alzheimer_model.h5")
