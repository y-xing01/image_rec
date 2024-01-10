import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import ImageMode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator


# Apply grayscale filter
def grayscale_filter(img):
    # Convert RGB to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

# Apply Equalization filter
def equalization_filter(img):
    # Apply histogram equalization
    img = cv2.equalizeHist(img)
    
    return img

# Apply Gaussian filter
def gaussian_filter(img):
    # Apply Gaussian filter
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

# Scale down
def scale_down_variance(img):
    return img / 255

# Apply all filters
def preprocess(img):
    # Apply grayscale filter
    img = grayscale_filter(img)
    # Apply equalization filter
    img = equalization_filter(img)
    # Apply Gaussian filter
    img = gaussian_filter(img)
    # Normalize pixel values to the range [0, 1]
    # Helps in stabilizing the learning process and can lead to faster convergence during training
    img = scale_down_variance(img)
    
    return img

# Reshape data
def reshape(images):
    return images.reshape(images.shape[0], 32, 32, 1)
    
# Build the model
def leNet_model():
  model = Sequential()
  # Convolutional layer with 60 filters, kernel size (5, 5), and ReLU activation
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Convolutional layer with 30 filters, kernel size (3, 3), and ReLU activation
  model.add(Conv2D(30, (3, 3), activation='relu'))
  # Max pooling layer with pool size (2, 2)
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Flatten the output for dense layers
  model.add(Flatten())
  # Dense layer with 500 neurons and ReLU activation
  model.add(Dense(500, activation='relu'))
  # Dropout layer with a dropout rate of 0.5
  model.add(Dropout(0.5))
  # Output layer with 'num_classes' neurons and softmax activation
  model.add(Dense(num_of_data, activation='softmax'))
  # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# Build modified model
def modified_model():
  model = Sequential()
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Increasing the depth and adding more layers
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(Conv2D(30, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Global Average Pooling
  GlobalAveragePooling2D(),  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_of_data, activation='softmax'))
  model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def evaluate_model(model, x_test, y_test):
    # Evaluate the model on the test set
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

def analyze_model(history):
    # Plot the training accuracy and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_loss(history):
    # Plot the training loss and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()