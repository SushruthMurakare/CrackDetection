# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the path to the dataset
data_dir = 'DataSet\Concrete\Concrete'

# Set up image data augmentation and preprocessing
# Includes rescaling pixel values and applying various transformations to increase dataset diversity
datagen = ImageDataGenerator(
    rescale=1./255,                 # Normalize image pixel values
    validation_split=0.2,           # Reserve 20% of the data for validation
    rotation_range=10,              # Randomly rotate images up to 30 degrees
    width_shift_range=0.1,          # Randomly shift images horizontally
    height_shift_range=0.1,         # Randomly shift images vertically
    shear_range=0.1,                # Shear transformations
    zoom_range=0.1,                 # Zoom in/out
    horizontal_flip=True            # Randomly flip images horizontally
)

# Create the training data generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Create the validation data generator
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Save list of training image filenames to a text file
with open('train_files.txt', 'w') as f:
    for file in train_generator.filepaths:
        f.write(file + '\n')

# Save list of validation image filenames to a text file
with open('validation_files.txt', 'w') as f:
    for file in validation_generator.filepaths:
        f.write(file + '\n')

# Build the model with dropout layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),  # 1st convolution layer
    MaxPooling2D((2, 2)),                                              # 1st pooling layer
    Conv2D(64, (3, 3), activation='relu'),                             # 2nd convolution layer
    MaxPooling2D((2, 2)),                                              # 2nd pooling layer
    Conv2D(128, (3, 3), activation='relu'),                            # 3rd convolution layer
    MaxPooling2D((2, 2)),                                              # 3rd pooling layer
    Flatten(),                                                         # Flatten 3D features to 1D
    Dense(128, activation='relu'),                                     # Fully connected dense layer
    Dropout(0.5),                                                      # Dropout layer for regularization
    Dense(1, activation='sigmoid')                                     # Output layer for binary classification
])

# Compile the model with binary crossentropy loss and Adam optimizer
optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model with class weights
# Class 0: No Crack, Class 1: Crack
class_weight = {0: 2.0, 1: 1.0}

# Train the model with early stopping callback
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    class_weight=class_weight,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate the trained model on validation data
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Generate predictions on validation data
predictions = model.predict(validation_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)  # Convert probabilities to class labels
true_classes = validation_generator.classes             # True class labels

# Compute and display confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Generate and print classification report (precision, recall, f1-score)
class_report = classification_report(true_classes, predicted_classes, target_names=['No Crack', 'Crack'])
print("Classification Report:")
print(class_report)

# Save the trained model to a file
model.save('crack_detection_model.h5')

# Optional: Create a download link for the model file in a notebook environment
from IPython.display import FileLink
FileLink(r'crack_detection_model.h5')