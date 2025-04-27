import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = 'DataSet\Concrete\Concrete'

# Data Augmentation with more aggressive techniques
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Load data
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=False
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Save training filenames
with open('train_files.txt', 'w') as f:
    for file in train_generator.filepaths:
        f.write(file + '\n')

# Save validation filenames
with open('validation_files.txt', 'w') as f:
    for file in validation_generator.filepaths:
        f.write(file + '\n')

# # Build the model with dropout layers
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model with class weights
# class_weight = {0: 1.0, 1: 0.5}
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=validation_generator,
#     class_weight=class_weight,
#     callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
# )



# Build the model with dropout layers
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()



# Train the model with class weights
class_weight = {0: 1.0, 1: 0.5}
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    class_weight=class_weight,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Predictions
predictions = model.predict(validation_generator)
predicted_classes = np.where(predictions > 0.5, 1, 0)
true_classes = validation_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(true_classes, predicted_classes, target_names=['No Crack', 'Crack'])
print("Classification Report:")
print(class_report)


# Save the model
model.save('crack_detection_model.h5')


from IPython.display import FileLink

# Create a link to download the h5 model file
FileLink(r'crack_detection_model.h5')