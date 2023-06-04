import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from model import create_model
from dataset import get_custom_sets, get_emnist_sets, get_sets
import matplotlib.pyplot as plt
import string

# load dataset use only last half of test set
#ds_train, ds_val, ds_test = get_emnist_sets(1, 1, -1)
ds_train, ds_val, ds_test = get_custom_sets()

# 62 classes
class_names = string.digits + string.ascii_uppercase + string.ascii_lowercase

# load pre trained model
model = tf.keras.models.load_model("models/feller-ES-MAX-E100-250K-N(512, 2, 2, 3, 0.8, 1, 64, 0.0005)-20230523-130946-FINE-TUNED")

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