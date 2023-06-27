import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from dataset import Dataset

FLAGS = flags.FLAGS

def define_flags():
    flags.DEFINE_string("model_file", None,
                      "Path and file name to the model file.")
    flags.DEFINE_string("model_export", None,
                  "Specify the path and file name for the new model")
    flags.mark_flag_as_required("model_file")
    flags.mark_flag_as_required("model_export")
    
def main(_):
    # get sets
    ds = Dataset(0, 0, 0)
    ds_train, ds_val, ds_test = ds.get_custom_sets()

    # number of epochs for model.fit()
    num_epochs = 50
    # batch size of dataset for fit
    batch_size = 32

    # set batch size
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)

    model = tf.keras.models.load_model(FLAGS.model_file)

    log_dir = "logs/fit/" + FLAGS.model_export

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # stop fitting if there is no improvement in validation loss within 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # train model and do validation after each epoch 
    history = model.fit(ds_train, validation_data=ds_val, epochs=num_epochs, callbacks=[tensorboard, early_stopping])
    model.save(FLAGS.model_export)

    # evaluate model
    loss, acc = model.evaluate(ds_val)

    summary = f'acc: {acc} loss: {loss}'
    epochs_num = len(history.history['loss'])

    # store model results and parameters
    with open("evaluations/" + FLAGS.model_export + ".txt", 'w') as eval_file:
        eval_file.write(f'{summary} ran_epochs: {epochs_num}\n')
        model.summary(print_fn=lambda x: eval_file.write(x + '\n'))


    # append short summary about model performance
    with open("summary.txt", 'a') as summary_file:
        summary_file.write(f'{FLAGS.model_export}: {summary} ran_epochs: {epochs_num}\n')
        
if __name__ == "__main__":
    define_flags()
    app.run(main)