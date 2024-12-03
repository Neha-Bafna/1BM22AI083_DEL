import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)  # Shape: (10000, 28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)    # Shape: (2000, 28, 28, 1)

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    zoom_range=0.1
)
datagen.fit(x_train)

# Convolutional Neural Network (CNN) Model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch > 10:
        return lr * 0.5
    return lr

lr_scheduler = LearningRateScheduler(scheduler)

# Train the CNN model
model = create_cnn_model()

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=20,
    callbacks=[lr_scheduler],
    verbose=1
)

# Plot Training and Validation Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2%}")

# Tangent Distance Classifier
def tangent_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

def tangent_distance_classifier(x_test, x_train, y_train):
    predictions = []
    for test_image in x_test:
        distances = [tangent_distance(test_image, train_image) for train_image in x_train]
        nearest_index = np.argmin(distances)
        predictions.append(y_train[nearest_index])
    return np.array(predictions)

# Apply Tangent Propagation
y_pred = tangent_distance_classifier(
    x_test.reshape(-1, 784),
    x_train.reshape(-1, 784),
    y_train
)

# Evaluate Tangent Distance Classifier
tangent_accuracy = np.mean(y_pred == y_test)
print(f"Tangent Distance Classifier Accuracy: {tangent_accuracy:.2%}")
