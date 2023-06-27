from tensorflow.keras import Sequential, layers
from tensorflow import keras
import tensorflow as tf

# create dynamic model with specific settings
def create_model(settings):
    nodes_per_layer, num_hidden_layers, conv_layers_per_block, num_conv_blocks, dropout_rate, batch_normalization, filter_start, learning_rate = settings
    
    model = Sequential()

    # filter start, gets doubled after every conv block (example 1. block 16, 2. block 32, 3. block 64 ..)
    filters = filter_start
    
    # this conv. layer is always static and cannot be changed due to the input shape
    model.add(layers.Conv2D(filters, (3, 3), activation='relu', input_shape=(73, 73, 1)))
    
    # add n-amount of conv blocks
    for conv_block in range(num_conv_blocks):
        # add n-amount of conv. layers
        for conv_layer in range(conv_layers_per_block):
            
            # skip first layer in first block because of the static conv. layer above
            if conv_block == 0 and conv_layer == 0: 
                continue
                
            # add conv layer
            model.add(layers.Conv2D(filters, (3, 3), activation='relu'))
            
        # add max pooling layer at the end of the conv block
        model.add(layers.MaxPooling2D((2, 2)))    
        
        # double the filter amount
        filters = filters * 2

    # flatten output
    model.add(layers.Flatten())

    # add n-amount of fc-layers
    for layer in range(num_hidden_layers):
        # add fc-layer with n-amount of nodes
        model.add(layers.Dense(nodes_per_layer, activation='relu'))
        
        # add batch normalization if set to 1
        if batch_normalization == 1:
            model.add(layers.BatchNormalization())
            
        # add dropout if its not 0
        if dropout_rate != 0:
            model.add(layers.Dropout(dropout_rate))
    
    # output layer of all 62 classes
    model.add(layers.Dense(62))

    # set learning rate of adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # enable logits for knowledge distillation
    model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    model.summary()

    return model