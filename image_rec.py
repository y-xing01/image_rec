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

# Define ImageDataGenerator with preprocessing for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Define ImageDataGenerator without augmentation for validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 3 using train_datagen generator
train_dataset = train_datagen.flow_from_directory(
    'C:/Users/User/Downloads/basedata/train/',
    target_size=(200, 200),
    batch_size=3,
    class_mode='binary'
)

# Flow validation images in batches of 3 using validation_datagen generator
validation_dataset = validation_datagen.flow_from_directory(
    'C:/Users/User/Downloads/basedata/validation/',
    target_size=(200, 200),
    batch_size=3,
    class_mode='binary'
)

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

model = model()

# Fit the model and get history for analysis
history = model.fit(
    train_dataset,
    steps_per_epoch=5,
    epochs=50,
    validation_data=validation_dataset
)

# Analyze and visualize the model performance
analyze_model(history)
plot_loss(history)

# Evaluation on test set
x_test, y_test = validation_dataset.next()
evaluate_model(model, x_test, y_test)

dir_path = 'C:/Users/User/Downloads/basedata/test/'

for i in os.listdir(dir_path):
    img_path = os.path.join(dir_path, i)

    if os.path.isfile(img_path):  # Check if it's a file, not a directory
        img = image.load_img(img_path, target_size=(200, 200))
        plt.imshow(img)
        plt.axis("off")

        # Preprocess the image
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        X = X / 255.0  # Normalization

        # Predict the class
        val = model.predict(X)

        if val <= 0.5:
            label = "Good Mushroom"
        else:
            label = "Defect Mushroom"

        plt.title(label)
        plt.show()
