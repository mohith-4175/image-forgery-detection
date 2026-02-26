import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageChops, ImageEnhance

# --------------------------------------------------
# Paths
# --------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cnn_model_path = os.path.join(base_dir, "ml", "models", "cnn_model.keras")

model = load_model(cnn_model_path)
print("✅ CNN model loaded")

# --------------------------------------------------
# ELA GENERATION
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

    os.remove(temp_path)
    return np.array(ela)

# --------------------------------------------------
# GRAD-CAM FUNCTION
# --------------------------------------------------
def generate_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# --------------------------------------------------
# APPLY HEATMAP ON IMAGE
# --------------------------------------------------
def apply_heatmap(image, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------
def generate_forgery_heatmap(image_path):
    ela_img = generate_ela(image_path)

    # CNN preprocessing
    img_resized = cv2.resize(ela_img, (224, 224))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediction
    prob = model.predict(img_input, verbose=0)[0][0]
    print("CNN Forgery Probability:", round(float(prob), 4))

    # Last conv layer name (IMPORTANT)
    last_conv_layer = "conv2d_2"   # <-- correct for your model

    heatmap = generate_gradcam(model, img_input, last_conv_layer)
    overlay = apply_heatmap(img_resized, heatmap)

    return img_resized, overlay, prob

# --------------------------------------------------
# TEST
# --------------------------------------------------
if __name__ == "__main__":
    test_image = os.path.join(
        base_dir,
        "data",
        "dataset",
        "Tp",   # forged image
        "Tp_D_NRN_S_B_arc00091_arc00095_11201.jpg"
    )

    original, heatmap_img, prob = generate_forgery_heatmap(test_image)

    cv2.imshow("ELA Image", original)
    cv2.imshow("Forgery Heatmap (Grad-CAM)", heatmap_img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()