import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import datetime, os, re
from dataset import Dataset
import csv
from absl import app
from absl import flags

FLAGS = flags.FLAGS

def define_flags():
    flags.DEFINE_string("model_file", None,
                      "Path and file name to the model file.")
    flags.mark_flag_as_required("model_file")
    
def main(_):
    ds = Dataset(1, -1, 1)
    ds_train, ds_val, ds_test = ds.get_sets()

    # load model from file
    model = tf.keras.models.load_model(FLAGS.model_file)

    # evaluate loss and accuracy for the model
    result = model.evaluate(ds_val.batch(32))

    print(result)

if __name__ == "__main__":
    define_flags()
    app.run(main)