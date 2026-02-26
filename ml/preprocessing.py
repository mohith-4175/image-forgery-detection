import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

# -----------------------------
# Base project directory
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input ELA folders
ela_tp_dir = os.path.join(base_dir, "data", "ela_dataset", "Tp")
ela_au_dir = os.path.join(base_dir, "data", "ela_dataset", "Au")

# Output dataset folder
output_dir = os.path.join(base_dir, "ml", "dataset")
os.makedirs(output_dir, exist_ok=True)

# Image size and limits
IMAGE_SIZE = 224
LIMIT_PER_CLASS = 1500   # 1500 forged + 1500 authentic = 3000 total

X = []
y = []

# -----------------------------
# Load limited images safely
# -----------------------------
def load_images_from_folder(folder, label, limit=1500):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    random.shuffle(files)
    files = files[:limit]   # take only first `limit` images

    count = 0
    for file in files:
        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        # Convert to float32 and normalize (VERY IMPORTANT for memory)
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(label)
        count += 1

    print(f"Loaded {count} images from {folder}")

# -----------------------------
# Load datasets
# -----------------------------
print("Loading forged (Tp) images...")
load_images_from_folder(ela_tp_dir, label=1, limit=LIMIT_PER_CLASS)

print("Loading authentic (Au) images...")
load_images_from_folder(ela_au_dir, label=0, limit=LIMIT_PER_CLASS)

X = np.array(X, dtype="float32")
y = np.array(y)

print("Total images:", len(X))
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# -----------------------------
# Train / Val / Test split
# -----------------------------
# 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# -----------------------------
# Save datasets
# -----------------------------
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)

np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)

np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

print("\n🎉 Preprocessing completed successfully! 🎉")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)