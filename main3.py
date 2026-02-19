import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

NO_ALZ_THRESHOLD = 0.20        # very low dementia signal
CONFIDENCE_WARNING_THRESHOLD = 0.60  # below this = unreliable
ALZ_PRESENT_THRESHOLD = 0.40  # clear Alzheimer-like signal

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

    # -------- 2. Alzheimer status classification (FIXED) --------

    non_demented_prob = probs["NonDemented"]
    dementia_probs = {
        "VeryMildDemented": probs["VeryMildDemented"],
        "MildDemented": probs["MildDemented"],
        "ModerateDemented": probs["ModerateDemented"]
    }

    alz_prob = max(dementia_probs.values())
    alz_stage = max(dementia_probs, key=dementia_probs.get)

    # ---- Case 1: No Alzheimer’s detected ----
    if alz_prob < NO_ALZ_THRESHOLD:
        binary_result = "No Alzheimer’s detected"
        binary_confidence = 1.0 - alz_prob
        stage_result = "NoAlzheimer"
        stage_conf = None

    # ---- Case 2: Alzheimer’s present but NON-DEMENTED ----
    elif alz_prob < ALZ_PRESENT_THRESHOLD:
        binary_result = "Possible Alzheimer’s pathology (non-demented)"
        binary_confidence = alz_prob
        stage_result = "NonDemented"
        stage_conf = non_demented_prob

    # ---- Case 3: Alzheimer’s WITH dementia ----
    else:
        binary_result = "Alzheimer’s with dementia"
        binary_confidence = alz_prob
        stage_result = alz_stage
        stage_conf = dementia_probs[alz_stage]

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
    if stage_result == "NoAlzheimer":
        progression = "No indication of Alzheimer’s pathology"

    elif stage_result == "NonDemented":
        progression = "Possible preclinical Alzheimer’s; no dementia symptoms"

    elif stage_result == "VeryMildDemented":
        progression = "Early dementia stage; progression possible"

    elif stage_result == "MildDemented":
        progression = "Established dementia; progression likely"

    elif stage_result == "ModerateDemented":
        progression = "Advanced dementia; progression ongoing"

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

result = predict_alz_analysis("test-mild.jpg", age=30, sex=1, family=0, apoe=0)

print("\n--- Alzheimer Analysis ---")
print("Binary result:", result["binary_result"])
print("Binary confidence:", round(result["binary_confidence"] * 100, 2), "%")

if result["stage_prediction"]:
    print("Stage:", result["stage_prediction"])
    if result["stage_confidence"]:
        print("Stage confidence:", round(result["stage_confidence"] * 100, 2), "%")

print("Risk level:", result["risk_level"])
print("Progression forecast:", result["progression_forecast"])

if result["confidence_warning"]:
    print("⚠️ Warning: Low confidence prediction")

print("\nRaw class probabilities:")
for k, v in result["raw_probabilities"].items():
    print(f"{k}: {v:.4f}")


# TODO: Add binary classification alz or not, Add risk levels (low / medium / high), Add a confidence warning threshold, Add progression forecasting (is it to develop, stop or start)