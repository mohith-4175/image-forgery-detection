import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern

# Parameters for LBP
RADIUS = 1
N_POINTS = 8 * RADIUS
METHOD = "uniform"

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_dir = os.path.join(base_dir, "ml", "dataset")
feature_dir = os.path.join(base_dir, "ml", "features")

os.makedirs(feature_dir, exist_ok=True)

# Load datasets
X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))

X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))

X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

print("Datasets loaded successfully!")

# -----------------------------
# Feature Extraction Function
# -----------------------------
def extract_features(images):
    features = []

    for img in images:
        # Convert to grayscale
        gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_BGR2GRAY)

        # LBP feature
        lbp = local_binary_pattern(gray, N_POINTS, RADIUS, METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)

        # Statistical features
        mean = np.mean(gray)
        std = np.std(gray)
        entropy = -np.sum(hist * np.log2(hist + 1e-6))

        feature_vector = np.hstack([hist, mean, std, entropy])
        features.append(feature_vector)

    return np.array(features)

# Extract features
print("Extracting features from training data...")
X_train_feat = extract_features(X_train)

print("Extracting features from validation data...")
X_val_feat = extract_features(X_val)

print("Extracting features from test data...")
X_test_feat = extract_features(X_test)

# Save features
np.save(os.path.join(feature_dir, "X_train_feat.npy"), X_train_feat)
np.save(os.path.join(feature_dir, "y_train.npy"), y_train)

np.save(os.path.join(feature_dir, "X_val_feat.npy"), X_val_feat)
np.save(os.path.join(feature_dir, "y_val.npy"), y_val)

np.save(os.path.join(feature_dir, "X_test_feat.npy"), X_test_feat)
np.save(os.path.join(feature_dir, "y_test.npy"), y_test)

print("🎉 Feature extraction completed successfully!")
print("Train features shape:", X_train_feat.shape)
print("Val features shape  :", X_val_feat.shape)
print("Test features shape :", X_test_feat.shape)