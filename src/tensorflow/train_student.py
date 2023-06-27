from distiller import Distiller
from dataset import Dataset
from absl import app
from absl import flags
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

FLAGS = flags.FLAGS

def define_flags():
    flags.DEFINE_string("model_file", None,
                      "Path and file name to the teacher model file.")
    flags.mark_flag_as_required("model_file")
    
    
def main(_):
    train_size = 250000
    
    ds = Dataset(train_size, -1, 0)
    ds_train, ds_val, ds_test = ds.get_sets()
    ds_val_fit = ds_val.take(25000)

    # student model for knowledge distillation
    student = keras.Sequential([
        keras.Input(shape=(73, 73, 1)),
        # block 1
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # block 2
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # block 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        # fc layer 1
        layers.Dense(128, activation='relu'),
        # fc layer 2
        layers.Dense(128, activation='relu'),
        
        # output layer 62 classes
        layers.Dense(62),
    ], name="student")

    
    # number of epochs for model.fit()
    num_epochs = 100
    # batch size of dataset for fit
    batch_size = 32

    # set batch sizes
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)
    ds_val_fit = ds_val_fit.batch(batch_size)

    
    run_name = "feller-student-distilled-model-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + run_name
    model_dir = "models/" + run_name
    
    
    teacher = tf.keras.models.load_model(FLAGS.model_file)
    
    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    distiller.compile(
        optimizer=optimizer,
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.5,
        temperature=1,
    )
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    
    # stop fitting if there is no improvement in validation loss within 5 epochs
    #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_student_loss', patience=5)
    
    # train model and do validation after each epoch 
    history = distiller.fit(ds_train, validation_data=ds_val_fit, epochs=num_epochs, callbacks=[tensorboard])

    # evaluate student
    distiller.evaluate(ds_val)
    
    saved_student = distiller.student
    
    saved_student.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
    saved_student.save(model_dir)
    
if __name__ == "__main__":
    define_flags()
    app.run(main)