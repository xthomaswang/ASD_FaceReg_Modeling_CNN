# src/models.py

"""
Module: models
Description: Contains functions and classes to build, train, and evaluate a CNN model.
Includes a custom ReLU activation, CNN architecture definition, model training, and evaluation.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Layer, Input
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#######################
# Custom ReLU Activation
#######################
def custom_relu(inputs, slope_positive=1.0):
    """
    Compute a custom ReLU activation.

    Description:
      Returns (slope_positive * x) if x > 0, otherwise returns 0.
    
    Inputs:
      - inputs: A TensorFlow tensor representing input values.
      - slope_positive (float): The scaling factor for positive values (default is 1.0).
    
    Output:
      - A TensorFlow tensor with the custom ReLU applied element-wise.
    """
    return tf.where(inputs > 0, slope_positive * inputs, tf.zeros_like(inputs))

class CustomReLU(Layer):
    """
    A custom Keras layer that applies the custom ReLU activation.

    Inputs:
      - inputs: A TensorFlow tensor.
      - slope_positive (float): The scaling factor for positive values.
    
    Output:
      - A TensorFlow tensor with the custom ReLU activation applied.
    """
    def __init__(self, slope_positive=1.0, **kwargs):
        super().__init__(**kwargs)
        self.slope_positive = slope_positive

    def call(self, inputs):
        return custom_relu(inputs, self.slope_positive)

#######################
# Build CNN Model
#######################
def build_EIB_cnn(
    input_shape=(130, 130, 3),
    slope_positive=1.0,
    filter_size=16,
    num_classes=10
):
    """
    Build and compile a CNN model with custom activation layers.

    Description:
      Constructs a CNN model composed of three convolutional blocks followed by dense layers.
      Each convolutional block increases the number of filters and includes a custom ReLU activation,
      BatchNormalization, MaxPooling, and Dropout. The model ends with a softmax output layer.

    Inputs:
      - input_shape (tuple): Shape of the input images (height, width, channels).
      - slope_positive (float): The positive slope for the CustomReLU activation.
      - filter_size (int): Initial number of convolution filters (doubles with each block).
      - num_classes (int): Number of output classes.
    
    Output:
      - model (tf.keras.Model): A compiled Keras model ready for training.
    """
    model = Sequential()

    # Block 1
    model.add(Conv2D(filters=filter_size, kernel_size=(5, 5),
                     input_shape=input_shape, padding="same"))
    model.add(CustomReLU(slope_positive))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2: Double filter count
    filter_size *= 2
    model.add(Conv2D(filters=filter_size, kernel_size=(5, 5)))
    model.add(CustomReLU(slope_positive))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Block 3: Double filter count again
    filter_size *= 2
    model.add(Conv2D(filters=filter_size, kernel_size=(5, 5)))
    model.add(CustomReLU(slope_positive))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # Naming the second Dense layer consistently with the slope value
    layer_name = f'least2_Dense_{slope_positive}'
    model.add(Dense(64, activation='relu', name=layer_name))
    model.add(Dropout(0.5))

    # Output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model with the Adam optimizer and binary crossentropy loss.
    # Note: Use binary_crossentropy for one-hot encoded multi-output labels.
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

#######################
# Train the Model
#######################
def train_model(
    model,
    data,       # Full dataset as a NumPy array (e.g., images)
    labels,     # Label set matching the order of data; expected as a DataFrame
    n_epochs=10,
    batch_size=64,
    verbose=1
):
    """
    Train the given CNN model using a specific train/test split strategy.

    Description:
      Splits the data so that for every block of 5 images, the first image is used for testing,
      and the remaining 4 images are used for training. Then, the model is trained and the training
      and validation metrics are returned.

    Inputs:
      - model (tf.keras.Model): The compiled Keras model to be trained.
      - data: The complete dataset (e.g., image data) as a NumPy array with shape (N, H, W, C).
      - labels: A DataFrame containing labels corresponding to 'data'. The column 'img_id' is dropped.
      - n_epochs (int): Number of training epochs (default is 10).
      - batch_size (int): Batch size for training (default is 64).
      - verbose (int): Verbosity mode (default is 1).
    
    Output:
      - train_acc: List of training accuracy values for each epoch.
      - train_loss: List of training loss values for each epoch.
      - val_acc: List of validation accuracy values for each epoch.
      - val_loss: List of validation loss values for each epoch.
    """
    X = np.array(data)
    Y = np.array(labels.drop(['img_id'], axis=1))
    
    # Initialize lists for training and testing splits
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    # Use the first image of every 5 as test data; remaining images as training data
    for i in range(len(X)):
        if i % 5 == 0:
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])

    # Convert lists to NumPy arrays with float32 type
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)
    
    # Train the model with validation on the test data
    history = model.fit(
        X_train, Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        verbose=verbose
    )

    # Extract metrics from training history
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history.get('val_accuracy', [])
    val_loss = history.history.get('val_loss', [])

    return train_acc, train_loss, val_acc, val_loss

#######################
# Evaluate the Model
#######################
def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model on test data.

    Inputs:
      - model (tf.keras.Model): The trained Keras model.
      - test_data: Test dataset as a NumPy array with shape (N, H, W, C).
      - test_labels: Corresponding test labels as a NumPy array.
    
    Output:
      - loss (float): The loss value on the test data.
      - accuracy (float): The accuracy on the test data.
    """
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
    return loss, accuracy
