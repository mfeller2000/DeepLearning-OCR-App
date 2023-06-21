import numpy as np
import tensorflow as tf
from dataset import get_custom_sets, get_emnist_sets, get_sets

# get sets
ds_train, ds_val, ds_test = get_custom_sets()

# number of epochs for model.fit()
num_epochs = 50
# batch size of dataset for fit
batch_size = 32

# set batch size
ds_train = ds_train.batch(batch_size)
ds_val = ds_val.batch(batch_size)

# load pre trained model
model_name = "../../models/feller-9-v1"
model = tf.keras.models.load_model("models/" + model_name)

# save model with specific name 
run_name = f'{model_name}-FINE-TUNED'

log_dir = "logs/fit/" + run_name
model_dir = "models/" + run_name

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# stop fitting if there is no improvement in validation loss within 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# train model and do validation after each epoch 
history = model.fit(ds_train, validation_data=ds_val, epochs=num_epochs, callbacks=[tensorboard, early_stopping])
model.save(model_dir)

# evaluate model
loss, acc = model.evaluate(ds_val)

summary = f'acc: {acc} loss: {loss}'
epochs_num = len(history.history['loss'])

# store model results and parameters
with open("evaluations/" + run_name + ".txt", 'w') as eval_file:
    eval_file.write(f'{summary} ran_epochs: {epochs_num}\n')
    model.summary(print_fn=lambda x: eval_file.write(x + '\n'))


# append short summary about model performance
with open("summary.txt", 'a') as summary_file:
    summary_file.write(f'{run_name}: {summary} ran_epochs: {epochs_num}\n')