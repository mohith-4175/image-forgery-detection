import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -----------------------------
# Paths
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_dir = os.path.join(base_dir, "ml", "dataset")
model_dir = os.path.join(base_dir, "ml", "models")
os.makedirs(model_dir, exist_ok=True)

# -----------------------------
# Load datasets
# -----------------------------
X_train = np.load(os.path.join(dataset_dir, "X_train.npy"))
y_train = np.load(os.path.join(dataset_dir, "y_train.npy"))
X_val = np.load(os.path.join(dataset_dir, "X_val.npy"))
y_val = np.load(os.path.join(dataset_dir, "y_val.npy"))
X_test = np.load(os.path.join(dataset_dir, "X_test.npy"))
y_test = np.load(os.path.join(dataset_dir, "y_test.npy"))

print("Datasets loaded:", X_train.shape, X_val.shape, X_test.shape)

# -----------------------------
# STABLE CNN ARCHITECTURE
# -----------------------------
inputs = Input(shape=(224, 224, 3))

x = Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, 3, activation="relu", padding="same")(x)
x = BatchNormalization()(x)

x = GlobalAveragePooling2D()(x)   # 🔥 KEY FIX
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)

outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

# -----------------------------
# Compile (LOW LR + SMOOTHING)
# -----------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # 🔥 VERY IMPORTANT
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=os.path.join(model_dir, "cnn_model.keras"),
        monitor="val_accuracy",
        save_best_only=True
    )
]

# -----------------------------
# Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# -----------------------------
# Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("\n🔥 FIXED CNN TEST ACCURACY:", test_acc)

model.save(os.path.join(model_dir, "cnn_model_final.keras"))
print("✅ CNN model saved (KERAS format)")