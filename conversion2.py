import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load('traineddp_model.onnx')

# Convert the ONNX model to a TensorFlow model
tf_rep = prepare(onnx_model)
tf_rep.export_graph('traineddp_model.pb')

# Convert the TensorFlow model to a TFLite model
converter = tf.lite.TFLiteConverter.from_frozen_graph('traineddp_model.pb', input_arrays=['input'], output_arrays=['output'])
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)
