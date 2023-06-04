import numpy as np
import tensorflow as tf
import datetime
from model import create_model
from dataset import get_custom_sets, get_emnist_sets, get_sets
import itertools

train_size = -1
ds_train, ds_val, ds_test = get_sets(train_size, -1, -1)
ds_val_fit = ds_val.take(25000)


# TUNABLE PARAMETERS 
# amount of nodes per fc-layer
nodes_per_layer = [512, 1024]
# amount of fc layers
num_hidden_layers = [2]
# amount of conv. layers per block
conv_layers_per_block = [2]
# amount of conv. block (every block has n-amount of conv. layers with max pooling layer at the end)
num_conv_blocks = [3]
# dropout rate for fc-layers (set 0 for no dropout at all)
dropout_rates = [0.8]
# batch normalization after every fc-layer (1 = add layer; 0 = dont add layer)
batch_normalization = [1]
# filter start, will double the filter amount after every block (1. block 16, 2. block 32 ...)
filter_starts = [32, 64]
# learning rate for adam optimizer
learning_rates = [0.001, 0.0005, 0.00025, 0.0001]

# create combinations for all parameters
combinations = list(itertools.product(nodes_per_layer, num_hidden_layers, conv_layers_per_block, num_conv_blocks, dropout_rates, batch_normalization, filter_starts, learning_rates))
print(len(combinations))

# number of epochs for model.fit()
num_epochs = 100
# batch size of dataset for fit
batch_size = 32

# set batch sizes
ds_train = ds_train.batch(batch_size)
ds_val = ds_val.batch(batch_size)
ds_val_fit = ds_val_fit.batch(batch_size)

# go through all model combinations
for settings in combinations:
    # save model with specific name 
    run_name = "feller-ES-MAX-E" + str(num_epochs) + "-" + str(round(train_size / 1000.0)) + "K-N" + str(settings) + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # create model see model.py
    model = create_model(settings)

    log_dir = "logs/fit/" + run_name
    model_dir = "models/" + run_name

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # stop fitting if there is no improvement in validation loss within 3 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    # train model and do validation after each epoch 
    history = model.fit(ds_train, validation_data=ds_val_fit, epochs=num_epochs, callbacks=[tensorboard, early_stopping])
    model.save(model_dir)
    
    # evaluate model
    loss, acc = model.evaluate(ds_val)
    summary = f'acc: {acc} loss: {loss}'
    
    # get ran epochs
    epochs_num = len(history.history['loss'])
    
    # store model results and parameters
    with open("evaluations/" + run_name + ".txt", 'w') as eval_file:
        eval_file.write(f'{summary} ran_epochs: {epochs_num}\n')
        model.summary(print_fn=lambda x: eval_file.write(x + '\n'))
        
        
    # append short summary about model performance
    with open("summary.txt", 'a') as summary_file:
        summary_file.write(f'{run_name}: {summary} ran_epochs: {epochs_num}\n')