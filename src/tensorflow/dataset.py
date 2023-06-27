import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential, layers

# preprocess for emnist set
def preprocess_emnist(image, label):
    # make images more human readable because emnist dataset is originally flipped and turned by 90 degree
    image = tf.image.flip_left_right(image)
    image = tf.image.rot90(image, k=1, name=None)

    # upscale from 28x28 to 73x73 to fit model input
    image = tf.image.resize(image, [73, 73])

    # normalize
    image = tf.cast(image, tf.float32) / 255.

    # invert colors
    image = 1 - image    

    return image, label


# preprocress for custom dataset         
def preprocess_custom(image, label):            
    # normalize
    image = tf.cast(image, tf.float32) / 255.

    # convert label from int32 to int64 to match emnist set
    label = tf.cast(label, tf.int64)

    # reduce channels 3 (rgb) to 1 (grayscale)
    image = tf.image.rgb_to_grayscale(image)

    return image, label


# add random noise to a 3-dim image
def add_noise_3dims(image, mean, stddev):
    noise = tf.random.normal(shape=tf.shape(image)[:-1], mean=mean, stddev=stddev, dtype=tf.float32)
    noise = tf.expand_dims(noise, axis=-1)  # Add an extra dimension to match image shape
    noise = tf.tile(noise, [1, 1, 3])  # Tile the noise tensor across the channel dimension
    noisy_image = tf.clip_by_value(image + noise, clip_value_min=0.0, clip_value_max=1.0)
    return noisy_image

    
# add random noise to a 1-dim image
def add_noise(image, mean, stddev):
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
    noisy_image = tf.clip_by_value(image + noise, clip_value_min=0.0, clip_value_max=1.0)
    return noisy_image


class Dataset:    
    """This takes 3 arguments to set the sizes for each subset in the emnist set"""
    def __init__(self, emnist_train_size, emnist_val_size, emnist_test_size):
        # data augmentation parameters
        self.__data_augmentation = tf.keras.Sequential([
          layers.RandomRotation(0.04),
          layers.RandomBrightness(0.25),
          layers.RandomContrast(0.2),
        ])
        
        self.emnist_train_size = emnist_train_size
        self.emnist_val_size = emnist_val_size
        self.emnist_test_size = emnist_test_size
        
        self.__ds_train_custom = tf.keras.utils.image_dataset_from_directory("../../data/train", image_size=(73, 73), batch_size=None)    
        self.__ds_val_custom = tf.keras.utils.image_dataset_from_directory("../../data/val", image_size=(73, 73), batch_size=None)
        self.__ds_val_test = tf.keras.utils.image_dataset_from_directory("../../data/test", image_size=(73, 73), batch_size=None)
        self.custom_sets = [self.__ds_train_custom, self.__ds_val_custom, self.__ds_val_test]
        
        # load emnist data via TFDS (TensorFlow Datasets), split into train and validation sets
        # Only 50% of test set will used so the other half can be used for validation
        self.emnist_sets = tfds.load('emnist', split=['train', 'test[0%:50%]', 'test[50%:]'], as_supervised=True, shuffle_files=True)
        
        # apply preprocessing such as augmentation, adding noise and shuffling
        self.__process_emnist_sets()
        self.__process_custom_sets()
        
        # create a combined dataset
        self.combined_sets = self.__process_combined_sets()
    
    
    
    @tf.autograph.experimental.do_not_convert #supress errors related to autograph
    def __process_emnist_sets(self):
        # data augmentation of training set
        self.emnist_sets[0] = self.emnist_sets[0].map(lambda x, y: (self.__data_augmentation(x), y))

        # iterate through the subsets
        for index in range(len(self.emnist_sets)):
            # randomize
            self.emnist_sets[index] = self.emnist_sets[index].shuffle(5000)
            # prerocess subset
            self.emnist_sets[index] = self.emnist_sets[index].map(preprocess_emnist)

        # set subset size
        if self.emnist_train_size != -1:
            self.emnist_sets[0] = self.emnist_sets[0].take(self.emnist_train_size)

        # set subset size
        if self.emnist_val_size != -1:
            self.emnist_sets[1] = self.emnist_sets[1].take(self.emnist_val_size)

        # set subset size
        if self.emnist_test_size != -1:
            self.emnist_sets[2] = self.emnist_sets[2].take(self.emnist_test_size)
            
        # add noise for the training set
        self.emnist_sets[0] = self.emnist_sets[0].map(lambda x, y: (add_noise(x, 0, 0.025), y))
        
        print(f'EMNIST Set Size: Train size: {len(self.emnist_sets[0])}, Val size: {len(self.emnist_sets[1])}, Test size: {len(self.emnist_sets[2])}')
        
        
        
    @tf.autograph.experimental.do_not_convert #supress errors related to autograph
    def __process_custom_sets(self):
        # data augmentation on training set
        self.custom_sets[0] = self.custom_sets[0].map(lambda x, y: (self.__data_augmentation(x), y))

        for index in range(len(self.custom_sets)):
            self.custom_sets[index] = self.custom_sets[index].map(preprocess_custom)
            self.custom_sets[index] = self.custom_sets[index].shuffle(1000)

        # add noise to the training set
        self.custom_sets[0] = self.custom_sets[0].map(lambda x, y: (add_noise(x, 0, 0.025), y))

        print(f'Custom Set Size: Train size: {len(self.custom_sets[0])}, Val size: {len(self.custom_sets[1])}, Test size: {len(self.custom_sets[2])}')
        
        
        
    def __process_combined_sets(self):
        combined_sets = [None] * 3
        
        # combine emnist and custom set
        for index in range(len(self.custom_sets)):
            combined_sets[index] = self.emnist_sets[index].concatenate(self.custom_sets[index])
            combined_sets[index] = combined_sets[index].shuffle(5000)

        print(f'Combined Set Size: Train size: {len(combined_sets[0])}, Val size: {len(combined_sets[1])}, Test size: {len(combined_sets[2])}')
        
        return combined_sets
    
    
    
    # return the combined set
    def get_sets(self):
        return self.combined_sets

    
    
    # return the custom set
    def get_custom_sets(self):
        return self.custom_sets

    
    
    # return the emnist set
    def get_emnist_sets(self):
        return self.emnist_sets