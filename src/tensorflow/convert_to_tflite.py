import tensorflow as tf

model_name = "../../models/feller-9-v1-fine"
converter = tf.lite.TFLiteConverter.from_saved_model(model_name)
tflite_model = converter.convert()

with open("../app/assets/feller-9-v1-fine.tflite", 'wb') as tflite_file:
    tflite_file.write(tflite_model)