import os
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from PIL import Image, ImageChops, ImageEnhance

# --------------------------------------------------
# Paths
# --------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

rf_model_path = os.path.join(base_dir, "ml", "models", "rf_model.pkl")
cnn_model_path = os.path.join(base_dir, "ml", "models", "cnn_model.keras")

# Load models
rf_model = joblib.load(rf_model_path)
cnn_model = load_model(cnn_model_path)

print("✅ Models loaded successfully")

# --------------------------------------------------
# ELA GENERATION (same logic as training)
# --------------------------------------------------
def generate_ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")

    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path)
    ela = ImageChops.difference(original, compressed)

    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela = ImageEnhance.Brightness(ela).enhance(scale)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return np.array(ela)

# --------------------------------------------------
# RF FEATURE EXTRACTION (MUST MATCH TRAINING)
# Total features = 13
# --------------------------------------------------
def extract_rf_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP (same params as training)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 8 + 3),     # 0..10
        range=(0, 8 + 2)
    )

    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)

    # Statistical features
    mean = np.mean(gray)
    std = np.std(gray)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))

    features = np.hstack([hist, mean, std, entropy])
    return features.reshape(1, -1)

# --------------------------------------------------
# ENSEMBLE PREDICTION
# --------------------------------------------------
def predict_image(image_path):
    # 1️⃣ Generate ELA
    ela_img = generate_ela(image_path)

    # 2️⃣ CNN preprocessing
    cnn_img = cv2.resize(ela_img, (224, 224))
    cnn_img = cnn_img.astype("float32") / 255.0
    cnn_img = np.expand_dims(cnn_img, axis=0)

    cnn_prob = cnn_model.predict(cnn_img, verbose=0)[0][0]

    # 3️⃣ RF preprocessing
    rf_feat = extract_rf_features(ela_img)
    rf_prob = rf_model.predict_proba(rf_feat)[0][1]

    # 4️⃣ ENSEMBLE (calibrated)
    final_score = 0.55 * rf_prob + 0.45 * cnn_prob

    # 5️⃣ Threshold (ELA-calibrated)
    label = "FORGED ❌" if final_score > 0.45 else "AUTHENTIC ✅"

    return {
        "rf_prob": float(rf_prob),
        "cnn_prob": float(cnn_prob),
        "final_score": float(final_score),
        "prediction": label
    }

# --------------------------------------------------
# MANUAL TEST
# --------------------------------------------------
if __name__ == "__main__":
    test_image = os.path.join(
        base_dir,
        "data",
        "dataset",
        "Au",   # change to Au for real image
        "Au_ani_00009.jpg"
    )

    result = predict_image(test_image)

    print("\n🔍 ENSEMBLE RESULT")
    print("RF Probability  :", round(result["rf_prob"], 4))
    print("CNN Probability :", round(result["cnn_prob"], 4))
    print("Final Score    :", round(result["final_score"], 4))
    print("Prediction     :", result["prediction"])