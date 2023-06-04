import tensorflow as tf

# preprocess for emnist set
def preprocess(image, label):
    # make images more human readable because emnist dataset is originally flipped and turned by 90 degree
    image = tf.image.flip_left_right(image)
    image = tf.image.rot90(image, k=1, name=None)

    # upscale from 28x28 to 73x73 to fit model input
    image = tf.image.resize(image, [73, 73])

    # normalize
    image = tf.cast(image, tf.float32) / 255.
                          
    return image, label

# preprocress for custom dataset         
def preprocess_custom(image, label):    
    # normalize
    image = tf.cast(image, tf.float32) / 255.
    
    # convert label from int32 to int64 to match emnist set
    label = tf.cast(label, tf.int64)
    
    # invert colors
    image = 1 - image
    
    # reduce channels 3 (rgb) to 1 (grayscale)
    image = tf.image.rgb_to_grayscale(image)
    
    return image, label