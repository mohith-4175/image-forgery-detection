import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from skimage.feature import local_binary_pattern
from PIL import Image, ImageChops, ImageEnhance

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_USE_LEGACY_KERAS'] = '0'

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
CNN_MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "cnn_model_final.keras")
RF_MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "rf_model.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ELA_DIR = os.path.join(OUTPUT_DIR, "ela")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")

os.makedirs(ELA_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
print("Loading models...")
cnn_model = keras.models.load_model(CNN_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
print("✅ CNN + RF models loaded")

# --------------------------------------------------
# STANDARD ELA (MODEL INPUT)
# --------------------------------------------------
def generate_standard_ela(image_path, quality=90):
    original = Image.open(image_path).convert("RGB")
    temp_path = os.path.join(OUTPUT_DIR, "temp_ela.jpg")
    original.save(temp_path, "JPEG", quality=quality)

    compressed = Image.open(temp_path)
    ela = ImageChops.difference(original, compressed)

    extrema = ela.getextrema()
    max_diff = max(e[1] for e in extrema)
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela = ImageEnhance.Brightness(ela).enhance(scale)
    os.remove(temp_path)

    return np.array(ela)

# --------------------------------------------------
# ELA ENHANCEMENT (DISPLAY ONLY)
# --------------------------------------------------
def enhance_ela_for_display(ela):
    ela = Image.fromarray(ela)
    ela = ImageEnhance.Brightness(ela).enhance(1.8)
    ela = ImageEnhance.Contrast(ela).enhance(1.5)
    return np.array(ela)

# --------------------------------------------------
# RF FEATURE EXTRACTION
# --------------------------------------------------
def extract_rf_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        range=(0, 10)
    )

    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)

    mean = np.mean(gray)
    std = np.std(gray)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))

    return np.hstack([hist, mean, std, entropy]).reshape(1, -1)

# --------------------------------------------------
# GRAD-CAM
# --------------------------------------------------
def generate_gradcam(model, img_array, layer_name="conv2d_2"):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# --------------------------------------------------
# APPLY HEATMAP
# --------------------------------------------------
def apply_heatmap(image, heatmap, alpha=0.6):
    base = (image * 255).astype("uint8")

    heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    heatmap = np.power(heatmap, 0.6)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)

# --------------------------------------------------
# FINAL PIPELINE
# --------------------------------------------------
def process_image(image_path):
    filename = os.path.basename(image_path)

    # -------- STANDARD ELA --------
    ela_std = generate_standard_ela(image_path)

    # -------- CNN --------
    cnn_img = cv2.resize(ela_std, (224, 224))
    cnn_img = cnn_img.astype("float32") / 255.0
    cnn_input = np.expand_dims(cnn_img, axis=0)
    cnn_prob = float(cnn_model.predict(cnn_input, verbose=0)[0][0])

    # -------- RF --------
    rf_feat = extract_rf_features(ela_std)
    rf_prob = float(rf_model.predict_proba(rf_feat)[0][1])

    # -------- CONFIDENCE SCORE --------
    final_score = max(rf_prob, cnn_prob)

    # -------- DECISION --------
    if final_score >= 0.21:
        label = "FORGED ❌"
    else:
        label = "AUTHENTIC ✅"

    print(f"\n🔍 {filename}")
    print("RF prob        :", round(rf_prob, 4))
    print("CNN prob       :", round(cnn_prob, 4))
    print("Confidence     :", round(final_score, 4))
    print("LABEL          :", label)

    # -------- SAVE ELA --------
    ela_display = enhance_ela_for_display(ela_std)
    ela_path = os.path.join(ELA_DIR, filename)
    cv2.imwrite(ela_path, cv2.cvtColor(ela_display, cv2.COLOR_RGB2BGR))

    # -------- HEATMAP --------
    heatmap_path = None
    if label == "FORGED ❌":
        heatmap = generate_gradcam(cnn_model, cnn_input)
        overlay = apply_heatmap(cnn_img, heatmap)
        heatmap_path = os.path.join(HEATMAP_DIR, filename)
        cv2.imwrite(heatmap_path, overlay)

    return label, final_score, ela_path, heatmap_path
