import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------
# Paths
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

feature_dir = os.path.join(base_dir, "ml", "features")
model_dir = os.path.join(base_dir, "ml", "models")

os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Load feature datasets
# -----------------------------
X_train = np.load(os.path.join(feature_dir, "X_train_feat.npy"))
y_train = np.load(os.path.join(feature_dir, "y_train.npy"))

X_val = np.load(os.path.join(feature_dir, "X_val_feat.npy"))
y_val = np.load(os.path.join(feature_dir, "y_val.npy"))

X_test = np.load(os.path.join(feature_dir, "X_test_feat.npy"))
y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

print("Feature datasets loaded successfully!")
print("Train:", X_train.shape, y_train.shape)
print("Val  :", X_val.shape, y_val.shape)
print("Test :", X_test.shape, y_test.shape)

# -----------------------------
# Train Random Forest
# -----------------------------
print("\n🔥 Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("✅ Random Forest training completed!")

# -----------------------------
# Evaluate on validation set
# -----------------------------
y_val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)

print("\n📊 Validation Accuracy:", val_acc)
print("Validation Classification Report:\n")
print(classification_report(y_val, y_val_pred))

# -----------------------------
# Final evaluation on test set
# -----------------------------
y_test_pred = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n🔥 TEST ACCURACY:", test_acc)
print("Test Classification Report:\n")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_test_pred))

# -----------------------------
# Save trained model
# -----------------------------
model_path = os.path.join(model_dir, "rf_model.pkl")
joblib.dump(rf, model_path)

print("\n🎉 Random Forest model saved at:", model_path)