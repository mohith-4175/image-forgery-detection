import numpy as np
import cv2
from skimage.feature import local_binary_pattern

def extract_rf_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, 8, 1, "uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 8 + 3),   # EXACT same as training
        range=(0, 8 + 2)
    )

    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)

    mean = np.mean(gray)
    std = np.std(gray)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))

    features = np.hstack([hist, mean, std, entropy])
    return features.reshape(1, -1)