from dataset import Dataset
from absl import app
from absl import flags
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

    
def main(_):
    train_size = 250000
    
    ds = Dataset(train_size, -1, 0)
    ds_train, ds_val, ds_test = ds.get_sets()
    ds_val_fit = ds_val.take(25000)

    # student model copied from train_student.py
    model = keras.Sequential([
        keras.Input(shape=(73, 73, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        
        layers.Dense(62),
    ])

    
    # number of epochs for model.fit()
    num_epochs = 25
    # batch size of dataset for fit
    batch_size = 32

    # set batch sizes
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)
    ds_val_fit = ds_val_fit.batch(batch_size)

    
    run_name = "feller-student-scratch-model-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + run_name
    model_dir = "models/" + run_name

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.summary()
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # stop fitting if there is no improvement in validation loss within 5 epochs
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    
    # train model and do validation after each epoch 
    history = model.fit(ds_train, validation_data=ds_val_fit, epochs=num_epochs, callbacks=[tensorboard])
    
    model.save(model_dir)

    # Evaluate student on test dataset
    loss, acc = model.evaluate(ds_val)
    summary = f'acc: {acc} loss: {loss}'
    
    # get ran epochs
    epochs_num = len(history.history['loss'])
        
    # append short summary about model performance
    with open("summary.txt", 'a') as summary_file:
        summary_file.write(f'{run_name}: {summary} ran_epochs: {epochs_num}\n')
    
if __name__ == "__main__":
    app.run(main)