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
model_path = 'crack_detection_model.h5'              # Path to the trained model file
positive_dir = r'DataSet\Concrete4Test\Positive'     # Directory containing images with cracks
negative_dir = r'DataSet\Concrete4Test\Negative'     # Directory containing images without cracks
output_dir = 'annotated_predictions_2'               # Directory where annotated prediction images will be saved
img_size = (150, 150)                                # Expected input size for the model

# --- Load the trained model ---
model = load_model(model_path)

# --- Clean up and recreate output directory ---
# Delete previous results if they exist
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
# Create a fresh directory for the new predictions
os.makedirs(output_dir, exist_ok=True)

# --- Define a function to preprocess images before prediction ---
def preprocess_image(image_path):
    # Load image and resize to expected dimensions
    img = load_img(image_path, target_size=img_size)
    # Convert image to a numpy array and normalize pixel values
    img_array = img_to_array(img) / 255.0
    # Add batch dimension (1, 150, 150, 3)
    return np.expand_dims(img_array, axis=0)

# --- Define class labels ---
class_map = {'No Crack': 0, 'Crack': 1}              # Human-readable to numerical label
inv_class_map = {v: k for k, v in class_map.items()} # Reverse mapping for display

# --- Collect image paths and true labels ---
image_paths = []
y_true = []

# Add all 'No Crack' (Negative) images and labels
for filename in os.listdir(negative_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(os.path.join(negative_dir, filename))
        y_true.append(class_map['No Crack'])

# Add all 'Crack' (Positive) images and labels
for filename in os.listdir(positive_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(os.path.join(positive_dir, filename))
        y_true.append(class_map['Crack'])

# --- Predict and save annotated images ---
y_pred = []

# Loop over each image path and true label
for img_path, true_class in zip(image_paths, y_true):
    # Preprocess the image
    x = preprocess_image(img_path)
    # Predict probability of 'Crack'
    prob = model.predict(x, verbose=0)[0][0]
    # Convert probability to binary class (threshold at 0.5)
    pred_class = int(prob > 0.5)
    y_pred.append(pred_class)

    # Convert numeric labels to human-readable labels
    label_pred = inv_class_map[pred_class]
    label_true = inv_class_map[true_class]

    # Print image result to console
    img_name = os.path.basename(img_path)
    print(f"{img_name} | GT: {label_true} | Pred: {label_pred} | Prob: {prob:.4f}")

    # Save image with new filename indicating prediction and ground truth
    new_name = f"GT_{label_true}_PRED_{label_pred}_{img_name}"
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, new_name))

# --- Print evaluation metrics ---
print("\n--- Evaluation Stats ---")     
print("Accuracy:", accuracy_score(y_true, y_pred))                                                               # Overall accuracy
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))                                                   # TP, TN, FP, FN matrix
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['No Crack', 'Crack']))     # Detailed precision, recall, f1-score