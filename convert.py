import tensorflow as tf
from tensorflow.keras.models import load_model
import tf2onnx
import onnx

print("Loading Keras model...")
model = load_model("ml/models/cnn_model_final.keras")

print("Converting to ONNX...")
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

onnx.save(model_proto, "ml/models/cnn_model.onnx")
print("✅ Saved! Size: ~20MB")