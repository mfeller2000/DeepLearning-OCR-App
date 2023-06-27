import tensorflow as tf
from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

def define_flags():
    flags.DEFINE_string("model_file", None,
                      "Path and file name to the TFLite model file.")
    flags.DEFINE_string("model_export", None,
                      "Path to save the TFLite model file.")
    flags.mark_flag_as_required("model_file")
    flags.mark_flag_as_required("model_export")
    
    
def main(_):
    model_name = FLAGS.model_file

    # load model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_name)
    # convert to tflite
    tflite_model = converter.convert()
    
    # save model
    with open(FLAGS.model_export, 'wb') as tflite_file:
        tflite_file.write(tflite_model)
        
if __name__ == "__main__":
    define_flags()
    app.run(main)