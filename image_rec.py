import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

def build_model():
    # Using a pre-trained VGG16 model with transfer learning
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))  # Adding dropout for regularization
    model.add(Dense(1, activation='sigmoid'))

    # Freeze the convolutional base
    base_model.trainable = False

    model.compile(Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
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

# Load image using OpenCV
img_cv2 = cv2.imread("C:/Users/User/Downloads/FYProject/image_rec/basedata/train/good/good1.png")

# Print shape
print("Image shape (using OpenCV):", img_cv2.shape)

# Define ImageDataGenerator with preprocessing for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # Augment with rotation
    width_shift_range=0.2,  # Augment with width shift
    height_shift_range=0.2,  # Augment with height shift
    brightness_range=[0.8, 1.2],  # Augment with brightness adjustment
    horizontal_flip=True
)

# Define ImageDataGenerator without augmentation for validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_dataset = train_datagen.flow_from_directory(
    'C:/Users/User/Downloads/FYProject/image_rec/basedata/train/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

# Flow validation images in batches using validation_datagen generator
validation_dataset = validation_datagen.flow_from_directory(
    'C:/Users/User/Downloads/FYProject/image_rec/basedata/validation/',
    target_size=(200, 200),
    batch_size=32,
    class_mode='binary'
)

# Build and compile the model
model = build_model()

# Fit the model and get history for analysis
history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_dataset),
    epochs=30,
    validation_data=validation_dataset,
    validation_steps=len(validation_dataset)
)

# Analyze and visualize the model performance
analyze_model(history)
plot_loss(history)

# Evaluation on a test set
x_test, y_test = validation_dataset.next()
evaluate_model(model, x_test, y_test)

# Load images from the test directory and make predictions
dir_path = 'C:/Users/User/Downloads/FYProject/image_rec/basedata/test/'

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
        prediction = model.predict(X)

        if prediction >= 0.5:
            label = "Good Mushroom"
        else:
            label = "Defect Mushroom"

        plt.title(label)
        plt.show()
