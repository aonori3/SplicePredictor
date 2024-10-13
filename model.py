from tensorflow.keras.layers import Input, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

FLANKING_LEN = 200

# Define a block for recurrent layers
def recurrent_block(X, n):
    """
    X: input tensor
    n: number of recurrent units
    """
    X = BatchNormalization()(X)
    X = GRU(units=n, return_sequences=True)(X)
    X = Dropout(0.2)(X)
    return X

# Define the recurrent model
def recurrent_model(input_shape=((FLANKING_LEN * 2 + 1), 4), classes=3):
    """
    Implementation of the neural network model using recurrent layers.
    
    Arguments:
    input_shape -- tuple with the shape of the input sequences.
    classes -- number of output classes predicted by the model (donor, acceptor, other).
    
    Returns:
    model -- a Keras Model instance.
    """
    X_input = Input(input_shape)
    
    # Recurrent blocks
    X = recurrent_block(X_input, n=32)
    X = recurrent_block(X, n=64)
    
    # Flatten the output and add dense layers
    X = Flatten()(X)
    X = Dense(units=16, activation='relu')(X)
    X = Dense(units=classes, activation='softmax')(X)
    
    # Create and return model
    model = Model(inputs=X_input, outputs=X)
    return model

# Learning rate scheduler function
def lr_scheduler(epoch, lr):
    """
    Reduces the learning rate by half after the first five epochs.
    
    Arguments:
    epoch -- current epoch number.
    lr -- current learning rate.
    
    Returns:
    updated learning rate.
    """
    decay_rate = 2
    if epoch > 5:
        return lr / decay_rate
    return lr

# Model compilation and training
def compile_and_train_model(x_train, y_train, x_val, y_val, input_shape, classes, learning_rate=0.001, n_epochs=10, n_batch=64):
    """
    Compiles and trains the recurrent model.
    
    Arguments:
    x_train -- training input data.
    y_train -- training labels.
    x_val -- validation input data.
    y_val -- validation labels.
    input_shape -- shape of the input data.
    classes -- number of output classes.
    learning_rate -- initial learning rate for the optimizer.
    n_epochs -- number of training epochs.
    n_batch -- batch size for training.
    
    Returns:
    history -- training history object.
    model -- the trained Keras model.
    """
    model = recurrent_model(input_shape, classes)
    
    # Compile model
    optimizer = Adam(learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Learning rate scheduler callback
    callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
    
    # Train the model
    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), 
                        epochs=n_epochs, batch_size=n_batch, callbacks=callbacks)
    
    return history, model
