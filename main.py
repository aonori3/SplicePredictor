import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

from data_processing import get_np_array
from model import convolutional_model
from metrics import performance_graphs, other_metrics, roc_auc_pr
from utils import get_callbacks

# Constants
DATA_DIR = "homo_sapiens"
MODEL_DIR = "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.5
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def load_and_preprocess_data():
    os.chdir(DATA_DIR)
    X, Y = get_np_array()
    x_train, x_val_test, y_train, y_val_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_test, x_val, y_test, y_val = train_test_split(x_val_test, y_val_test, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    
    y_train = tf.keras.utils.to_categorical(y_train)
    y_val = tf.keras.utils.to_categorical(y_val)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def print_dataset_shapes(x_train, x_val, x_test):
    print(f"Training set shape: {x_train.shape}")
    print(f"Validation set shape: {x_val.shape}")
    print(f"Test set shape: {x_test.shape}")

def train_model(model, x_train, y_train, x_val, y_val):
    optimizer = Adam(LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=get_callbacks()
    )
    return history

def evaluate_model(model, x_test, y_test):
    other_metrics(model, x_test, y_test)
    roc_auc_pr(model, x_test, y_test)

def save_model(model):
    os.chdir(os.path.join("..", "..", MODEL_DIR))
    model.save_weights("deep_splicer_weights.h5")
    model.save("deep_splicer.h5")

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()
    print_dataset_shapes(x_train, x_val, x_test)
    
    model = convolutional_model()
    history = train_model(model, x_train, y_train, x_val, y_val)
    
    performance_graphs(EPOCHS, history)
    evaluate_model(model, x_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()