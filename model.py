from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1

FLANKING_LEN = 200

def convolutional_block(x, filters, kernel_size):
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l1(1e-5))(x)
        x = MaxPooling1D()(x)
    return x

def convolutional_model(input_shape=((FLANKING_LEN * 2 + 1), 4), classes=3):
    inputs = Input(input_shape)
    x = inputs

    # Convolutional blocks
    x = convolutional_block(x, filters=16, kernel_size=11)
    x = convolutional_block(x, filters=32, kernel_size=11)
    x = convolutional_block(x, filters=64, kernel_size=21)
    x = convolutional_block(x, filters=64, kernel_size=41)

    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x = Flatten()(x)
    outputs = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model