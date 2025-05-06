# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
import shutil
from PIL import Image

# --- SETTINGS ---
model_path = 'crack_detection_model.h5'           # Path to the saved model file
validation_list_path = 'validation_files.txt'     # File containing list of validation image paths
output_dir = 'annotated_predictions'              # Directory where annotated prediction images will be saved
img_size = (150, 150)                             # Expected input size for the model

# --- Load the pre-trained model ---
model = load_model(model_path)

# --- Clean up and recreate the output directory ---
# Remove any existing directory and its contents
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
# Create a fresh directory
os.makedirs(output_dir, exist_ok=True)

# --- Define preprocessing function for images ---
def preprocess_image(image_path):
    # Load and resize image
    img = load_img(image_path, target_size=img_size)
    # Convert to numpy array and normalize pixel values
    img_array = img_to_array(img) / 255.0
    # Expand dimensions to match model input shape (1, 150, 150, 3)
    return np.expand_dims(img_array, axis=0)

# --- Define class mappings ---
class_map = {'No Crack': 0, 'Crack': 1}               # Class names to numeric labels
inv_class_map = {v: k for k, v in class_map.items()} # Reverse mapping for display

# --- Load list of validation image paths ---
with open(validation_list_path, 'r') as f:
    image_paths = [line.strip() for line in f if line.strip()]  # Remove empty lines and strip whitespace

# --- Initialize ground truth and prediction lists ---
y_true = []  # True labels
y_pred = []  # Predicted labels

# --- Process and predict each image ---
for img_path in image_paths:
    # Determine ground truth class based on filename keyword
    if 'Negative' in img_path:
        true_class = class_map['No Crack']
    elif 'Positive' in img_path:
        true_class = class_map['Crack']
    else:
        raise ValueError(f"Cannot determine class from path: {img_path}")

    # Preprocess image and get prediction
    x = preprocess_image(img_path)
    prob = model.predict(x, verbose=0)[0][0]         # Probability of being 'Crack'
    pred_class = int(prob > 0.5)                     # Convert probability to binary label

    # Save labels for evaluation
    y_true.append(true_class)
    y_pred.append(pred_class)

    # Map predicted and true classes to human-readable labels
    label_pred = inv_class_map[pred_class]
    label_true = inv_class_map[true_class]

    # Print result to console
    img_name = os.path.basename(img_path)
    print(f"{img_name} | GT: {label_true} | Pred: {label_pred} | Prob: {prob:.4f}")

    # Save a copy of the image with the prediction in the filename
    new_name = f"GT_{label_true}_PRED_{label_pred}_{img_name}"
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, new_name))

# --- Print overall evaluation metrics ---
print("\n--- Evaluation Stats ---")
print("Accuracy:", accuracy_score(y_true, y_pred))  # Overall accuracy
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))  # True vs. predicted
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['No Crack', 'Crack']))  # Precision, recall, f1