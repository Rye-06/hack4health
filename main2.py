import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128
CONFIDENCE_WARNING_THRESHOLD = 0.60  # below this = unreliable

model = load_model("multimodal_alzheimer_model.h5")

CLASSES = [
    "NonDemented",
    "VeryMildDemented",
    "MildDemented",
    "ModerateDemented"
]

# Neutral metadata defaults
DEFAULT_META = np.array([[0.7, 0, 0, 0]])
# age=70, female, no family history, no APOE ε4


# ---------------- Preprocessing ----------------

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unreadable")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_meta(age, sex, family, apoe):
    if None in (age, sex, family, apoe):
        return DEFAULT_META
    return np.array([[age / 100, sex, family, apoe]])


# ---------------- Main Prediction ----------------

def predict_alz_analysis(image_path, age=None, sex=None, family=None, apoe=None):
    img = preprocess_image(image_path)
    meta = preprocess_meta(age, sex, family, apoe)

    preds = model.predict([img, meta], verbose=0)[0]
    probs = dict(zip(CLASSES, preds))

    # -------- 1. Stage classification --------
    stage = CLASSES[np.argmax(preds)]
    stage_confidence = probs[stage]

    # -------- 2. Binary Alzheimer-like classification --------
    non_demented_prob = probs["NonDemented"]
    dementia_probs = [probs[c] for c in CLASSES if c != "NonDemented"]
    alz_prob = max(dementia_probs)

    if alz_prob > non_demented_prob:
        binary_result = "Possible Alzheimer-like pattern"
        binary_confidence = alz_prob
        stage_result = stage
        stage_conf = stage_confidence
    else:
        binary_result = "No Alzheimer-like pattern"
        binary_confidence = non_demented_prob
        stage_result = "NonDemented"
        stage_conf = None

    # -------- 3. Risk level (rule-based) --------
    risk_score = alz_prob

    if apoe == 1:
        risk_score += 0.5
    if family == 1:
        risk_score += 0.3
    if age is not None and age < 65:
        risk_score += 0.2

    risk_score = min(risk_score, 1.0)

    if risk_score < 0.33:
        risk_level = "Low"
    elif risk_score < 0.66:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # -------- 4. Confidence warning --------
    confidence_warning = binary_confidence < CONFIDENCE_WARNING_THRESHOLD

    # -------- 5. Progression forecasting (heuristic) --------
    if binary_result == "No Alzheimer-like pattern":
        progression = "No current indication of progression"
    elif stage_result == "VeryMildDemented":
        progression = "Possible early-stage pattern; monitoring suggested"
    elif stage_result == "MildDemented":
        progression = "Likely progression without intervention"
    elif stage_result == "ModerateDemented":
        progression = "Advanced-stage pattern; progression likely ongoing"
    else:
        progression = "Stage could not be determined"

    return {
        "binary_result": binary_result,
        "binary_confidence": binary_confidence,
        "stage_prediction": stage_result,
        "stage_confidence": stage_conf,
        "risk_level": risk_level,
        "confidence_warning": confidence_warning,
        "progression_forecast": progression,
        "raw_probabilities": probs
    }


# ---------------- Example usage ----------------

result = predict_alz_analysis("test-mild.jpg", age=62, sex=1, family=1, apoe=1)

print("\n--- Alzheimer Analysis ---")
print("Binary result:", result["binary_result"])
print("Binary confidence:", round(result["binary_confidence"] * 100, 2), "%")

if result["stage_prediction"]:
    print("Stage:", result["stage_prediction"])
    print("Stage confidence:", round(result["stage_confidence"] * 100, 2), "%")

print("Risk level:", result["risk_level"])
print("Progression forecast:", result["progression_forecast"])

if result["confidence_warning"]:
    print("⚠️ Warning: Low confidence prediction")

print("\nRaw class probabilities:")
for k, v in result["raw_probabilities"].items():
    print(f"{k}: {v:.4f}")


# TODO: Add binary classification alz or not, Add risk levels (low / medium / high), Add a confidence warning threshold, Add progression forecasting (is it to develop, stop or start)