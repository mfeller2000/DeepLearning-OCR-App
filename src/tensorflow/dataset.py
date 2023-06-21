import tensorflow as tf
import tensorflow_datasets as tfds
from preprocess_ds import preprocess, preprocess_custom

# get and combine emnist (train_size, val_size, test_size) and custom set
def get_sets(*sizes):

    emnist_sets = get_emnist_sets(*sizes)
    custom_sets = get_custom_sets()
    
    # combine emnist and custom set
    for index in range(len(custom_sets)):
        emnist_sets[index] = emnist_sets[index].concatenate(custom_sets[index])
    
    print(f'Merged Set: Train size: {len(emnist_sets[0])}, Val size: {len(emnist_sets[1])}, Test size: {len(emnist_sets[2])}')
    
    return emnist_sets

# get only custom set
def get_custom_sets():
    ds_train_custom = tf.keras.utils.image_dataset_from_directory("dataset/train", image_size=(73, 73), batch_size=None)
    ds_val_custom = tf.keras.utils.image_dataset_from_directory("dataset/val", image_size=(73, 73), batch_size=None)
    ds_val_test = tf.keras.utils.image_dataset_from_directory("dataset/test", image_size=(73, 73), batch_size=None)
    
    custom_sets = [ds_train_custom, ds_val_custom, ds_val_test]
    
    print(f'Custom Set: Train size: {len(custom_sets[0])}, Val size: {len(custom_sets[1])}, Test size: {len(custom_sets[2])}')
    
    # preprocess dataset
    for index in range(len(custom_sets)):
        custom_sets[index] = custom_sets[index].map(preprocess_custom)
    
    return custom_sets

# get only emnist set with set sizes (train_size, val_size, test_size) (set -1 for whole subset)
def get_emnist_sets(*sizes):
    # load emnist data via TFDS (TensorFlow Datasets), split into train and validation sets
    # Only 50% of test set will used so the other half can be used for validation
    emnist_sets = tfds.load('emnist', split=['train', 'test[0%:50%]', 'test[50%:]'], as_supervised=True, shuffle_files=True)
    
    # iterate through the subsets
    for index in range(len(emnist_sets)):
        # randomize
        emnist_sets[index] = emnist_sets[index].shuffle(5000)
        
        # set subset size
        if sizes[index] != -1:
            emnist_sets[index] = emnist_sets[index].take(sizes[index])
    
        # prerocess subset
        emnist_sets[index] = emnist_sets[index].map(preprocess)
        
    print(f'EMNIST Set: Train size: {len(emnist_sets[0])}, Val size: {len(emnist_sets[1])}, Test size: {len(emnist_sets[2])}')
    
    return emnist_sets