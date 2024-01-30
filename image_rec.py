import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2
import requests
from PIL import Image
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import os

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

# Function to read a folder of PNG images
def read_png_folder(folder_path, target_size=(32, 32)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = preprocess(img)  # Apply your preprocessing function
            img = cv2.resize(img, target_size)  # Resize the image to a consistent size
            images.append(img)
    return np.array(images)

# Reshape data
def reshape(images):
    return images.reshape(images.shape[0], 32, 32, 1)

folder_path = "C:/Users/User/Downloads/mush_img/"

print("Folder path:", folder_path)

# Read images and labels
images = []
labels = []

for class_folder in os.listdir(folder_path):
    print("Class folder:", class_folder)

    class_folder_path = os.path.join(folder_path, class_folder)
    print("Class folder path:", class_folder_path)

    if os.path.isdir(class_folder_path):
        class_images = read_png_folder(class_folder_path)
        print("Class images shape:", class_images.shape)

        class_labels = [int(class_folder)] * len(class_images)
        print("Class labels:", class_labels)

        images.extend(class_images)
        labels.extend(class_labels)


# Get the first image and its corresponding label
print("Total images:", len(images))
print("Total labels:", len(labels))

# Check if any images were read
if len(images) == 0:
    print("No images were read. Please check the folder path and image format.")
else:
    # Print the first image and its corresponding label
    image = images[0]
    label = labels[0]

    print("Image shape:", image.shape)
    print("Label:", label)

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title('Class label: {}'.format(label))
    plt.show()

# def leNet_model():
#   model = Sequential()
#   # Convolutional layer with 60 filters, kernel size (5, 5), and ReLU activation
#   model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
#   # Max pooling layer with pool size (2, 2)
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   # Convolutional layer with 30 filters, kernel size (3, 3), and ReLU activation
#   model.add(Conv2D(30, (3, 3), activation='relu'))
#   # Max pooling layer with pool size (2, 2)
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   # Flatten the output for dense layers
#   model.add(Flatten())
#   # Dense layer with 500 neurons and ReLU activation
#   model.add(Dense(500, activation='relu'))
#   # Dropout layer with a dropout rate of 0.5
#   model.add(Dropout(0.5))
#   # Output layer with 'num_classes' neurons and softmax activation
#   model.add(Dense(num_of_data, activation='softmax'))
#   # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
#   model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#   return model

# # Build modified model
# def modified_model():
#   model = Sequential()
#   model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
#   model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
#   model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   # Increasing the depth and adding more layers
#   model.add(Conv2D(30, (3, 3), activation='relu'))
#   model.add(Conv2D(30, (3, 3), activation='relu'))
#   model.add(Conv2D(30, (3, 3), activation='relu'))
#   model.add(MaxPooling2D(pool_size=(2, 2)))
#   # Global Average Pooling
#   GlobalAveragePooling2D(),  
#   model.add(Flatten())
#   model.add(Dense(500, activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(num_of_data, activation='softmax'))
#   model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#   return model


