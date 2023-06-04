import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import datetime, os, re
from dataset import get_custom_sets, get_emnist_sets, get_sets
import csv

models_dir = "models/"
results = []

ds_train, ds_val, ds_test = get_sets(1, -1, 1)
    
# validate multiple models at once in models/ directory
for dir in os.listdir(models_dir):
    # only include certain models with regex
    #if re.search("^feller-ES-MAX-E100-250K-N(512, 2, 2, 3, 0.8, 1, 64, 0.0005)-20230523-130946.*", dir):
    if dir == "feller-ES-MAX-E100-250K-N(512, 2, 2, 3, 0.8, 1, 64, 0.0005)-20230523-130946-FINE-TUNED":
        model_dir = os.path.join(models_dir, dir)
        # load pre trained model
        model = tf.keras.models.load_model(model_dir)
        
        # evaluate loss and accuracy for the model
        loss, acc = model.evaluate(ds_val.batch(32))

        results.append({"model_name": dir, "acc": acc, "loss": loss})
        print("Model " + dir + ", accuracy: {:5.2f}%".format(100 * acc))


# sort by loss
sorted_results = sorted(results, key=lambda x: x['loss'])

# save val results to csv
keys = sorted_results[0].keys()
with open("evaluations/eval-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv", 'w', newline='') as file:
    writer = csv.DictWriter(file, keys)
    writer.writeheader()
    writer.writerows(sorted_results)

# print validation results as table
headers = ['model_name', 'acc', 'loss']
print(f'{headers[0]: <100}{headers[1]: <25}{headers[2]}')

for item in results:
    print(f'{item["model_name"]: <100}{item["acc"]:<25}{item["loss"]}')