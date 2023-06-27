import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
import tensorflow_datasets as tfds
from dataset import Dataset
import matplotlib.pyplot as plt
import string

FLAGS = flags.FLAGS

def define_flags():
    flags.DEFINE_string("model_file", None,
                      "Path and file name to the TFLite model file.")
    flags.mark_flag_as_required("model_file")

    
def main(_):
    # load dataset use only last half of test set
    ds = Dataset(0, 0, -1)
    ds_train, ds_val, ds_test = ds.get_sets()

    # 62 classes
    class_names = string.digits + string.ascii_uppercase + string.ascii_lowercase

    # load pre trained model
    model = tf.keras.models.load_model(FLAGS.model_file)

    results = {}

    # predict all 62 classes
    for i in range(0, 62):
        # filter by label
        ds_class = ds_test.filter(lambda image, label: label == i)
        temp_results = []

        # predict 100 images per class
        for image, label in ds_class.take(100):
            # Make predictions using the pre-trained model
            prediction = model.predict(image[tf.newaxis, ...])

            # get best prediction
            predicted_label = np.argmax(prediction)

            # check if prediction was right or wrong
            temp_results.append(predicted_label == label.numpy())

        # average result
        results[class_names[i]] = np.mean(temp_results)

    # create and save bar chart
    names = list(results.keys())
    values = list(results.values())
    plt.figure().set_figwidth(15)
    plt.bar(range(len(results)), values, tick_label=names, width = 0.4)
    plt.savefig("dist.png")


if __name__ == "__main__":
    define_flags()
    app.run(main)