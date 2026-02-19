import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("multimodal_alzheimer_model.h5")

classes = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# Neutral metadata defaults (population average)
DEFAULT_META = np.array([[0.7, 0, 0, 0]])  
# age=70, female, no family history, no APOE ε4

def predict(image_path, age=None, sex=None, family=None, apoe=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if None in (age, sex, family, apoe):
        meta = DEFAULT_META
    else:
        meta = np.array([[age / 100, sex, family, apoe]])

    pred = model.predict([img, meta], verbose=0)
    return classes[np.argmax(pred)]

# Example usage
print(predict("test-normal.jpg"))  # image only
print(predict("test-normal.jpg", age=62, sex=1, family=1, apoe=1)) # extra param


# weight = 1.0
# + 0.5 if APOE ε4
# + 0.3 if family history
# + 0.2 if age < 65 (early onset emphasis)