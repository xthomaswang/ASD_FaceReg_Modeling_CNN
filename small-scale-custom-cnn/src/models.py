# src/models.py 

"""
Module: models
Description: Contains optimized functions to build, train, and evaluate a CNN model.
Includes a versatile custom activation layer supporting both ReLU and PReLU-like behaviors,
now with a configurable activation threshold.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Layer, Input, GaussianNoise
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#######################
# Custom Activation Layer (with Threshold)
#######################
# @tf.keras.saving.register_keras_serializable()
class CustomActivation(Layer):
    """
    A custom Keras layer that applies a versatile activation function.
    - NEW: Includes a configurable 'threshold' to shift the activation boundary.
    - If slope_negative is 0, it acts like a shifted & scaled ReLU.
    - If slope_negative > 0, it acts like a shifted LeakyReLU.
    """
    def __init__(self, slope_positive=1.0, slope_negative=0.0, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.slope_positive = slope_positive
        self.slope_negative = slope_negative
        self.threshold = threshold  # The new activation threshold parameter

    def call(self, inputs):
        """Applies the activation logic using the custom threshold."""
        # The condition is now based on the configurable threshold
        return tf.where(inputs > self.threshold, 
                        self.slope_positive * (inputs - self.threshold), 
                        self.slope_negative * (inputs - self.threshold))
    
    def get_config(self):
        """Enables model saving and loading with the new parameter."""
        config = super().get_config()
        config.update({
            'slope_positive': self.slope_positive,
            'slope_negative': self.slope_negative,
            'threshold': self.threshold  # Ensure threshold is saved
        })
        return config

#######################
# Build CNN Model
#######################
def build_cnn(
    input_shape=(130, 130, 3),
    slope_positive=1.0,
    slope_negative=0.0,
    threshold=0.0, 
    noise_level=0,
    filter_size=16,
    num_classes=10,
    learning_rate=0.001,
    categorical=True  # false = 'binary_crossentropy', true = 'categorical_crossentropy'
):
    """
    Builds and compiles a CNN model with the versatile CustomActivation layer.
    """
    model = Sequential(name=f'cnn_model_sp{slope_positive}_sn{slope_negative}_t{threshold}_n{noise_level}')
    
    model.add(Input(shape=input_shape))
    
    # Build convolutional blocks in a loop
    for i in range(3):
        block_name = f'conv_block{i+1}'
        current_filters = filter_size * (2 ** i)
        
        model.add(Conv2D(current_filters, (5, 5), padding="same", name=f'{block_name}_conv'))
        
        # Use the versatile CustomActivation layer with the new threshold
        model.add(CustomActivation(
            slope_positive=slope_positive, 
            slope_negative=slope_negative, 
            threshold=threshold,  # Pass the threshold here
            name=f'{block_name}_relu' 
        ))
        
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.2))
        
        if noise_level > 0:
            model.add(GaussianNoise(stddev=noise_level, name=f'{block_name}_noise'))
    
    # Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    layer_name = f'least2_Dense_sp{slope_positive}_sn{slope_negative}_t{threshold}_n{noise_level}' # Updated name
    model.add(Dense(64, activation='relu', name=layer_name))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    loss_function = 'categorical_crossentropy' if categorical else 'binary_crossentropy'
    optimizer = Adam(learning_rate=learning_rate)
    print(f"➡️ Compiling model with loss function: {loss_function}")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    
    return model

#######################
# Train and Evaluate Functions (Unchanged)
#######################

def train_model(
    model,
    data,
    labels,
    n_epochs=10,
    batch_size=64,
    verbose=1
):
    """
    Trains the given CNN model using an efficient train/test split strategy.
    """
    X = np.array(data, dtype=np.float32)
    if 'img_id' in labels.columns:
        y_df = labels.drop(['img_id'], axis=1)
    else:
        y_df = labels
    Y = y_df.to_numpy(dtype=np.float32)

    test_indices = np.arange(0, len(X), 5)
    train_indices = np.setdiff1d(np.arange(len(X)), test_indices)

    X_train, Y_train = X[train_indices], Y[train_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    
    print(f"Data split efficiently: {len(X_train)} training samples, {len(X_test)} validation samples.")
    
    history = model.fit(
        X_train, Y_train,
        epochs=n_epochs,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        verbose=verbose
    )

    train_acc = history.history.get('accuracy', [])
    train_loss = history.history.get('loss', [])
    val_acc = history.history.get('val_accuracy', [])
    val_loss = history.history.get('val_loss', [])

    return train_acc, train_loss, val_acc, val_loss

def evaluate_model(model, test_data, test_labels):
    """Evaluates the trained model on test data."""
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=1)
    return loss, accuracy