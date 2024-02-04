
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np
import cv2
import os

# Load image using keras.preprocessing
img_keras = image.load_img("C:/Users/User/Downloads/basedata/train/good/good1.png")

# Display image using matplotlib
plt.imshow(img_keras)
plt.show()

# Load image using OpenCV
img_cv2 = cv2.imread("C:/Users/User/Downloads/basedata/train/good/good1.png")

# Print shape
print("Image shape (using OpenCV):", img_cv2.shape)

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('C:/Users/User/Downloads/basedata/train/',
                                          target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')

validation_dataset = train.flow_from_directory('C:/Users/User/Downloads/basedata/validation/',
                                          target_size=(200,200),
                                         batch_size=3,
                                         class_mode='binary')

# Print class indices
print(train_dataset.class_indices)

# Print dataset classes
print(train_dataset.classes)

def model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
    
model = model()

model_fit = model.fit(train_dataset, 
                      steps_per_epoch=5, 
                      epochs=30, 
                      validation_data=validation_dataset)

