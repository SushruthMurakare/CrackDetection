import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
import shutil
from PIL import Image

# --- SETTINGS ---
model_path = 'crack_detection_model.h5'          # Path to your model
validation_list_path = 'validation_files.txt'    # Path to your list of image paths
output_dir = 'annotated_predictions'             # Directory to save annotated images
img_size = (150, 150)                            # Image size expected by the model

# --- Load Model ---
model = load_model(model_path)

# --- Cleanup Output Directory ---
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# --- Preprocessing Function ---
def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Class Mapping ---
class_map = {'No Crack': 0, 'Crack': 1}
inv_class_map = {v: k for k, v in class_map.items()}

# --- Load Validation Image Paths ---
with open(validation_list_path, 'r') as f:
    image_paths = [line.strip() for line in f if line.strip()]

# --- Initialize Lists ---
y_true = []
y_pred = []

# --- Process Each Image ---
for img_path in image_paths:
    # Determine true class based on path (simple keyword search)
    if 'Negative' in img_path:
        true_class = class_map['No Crack']
    elif 'Positive' in img_path:
        true_class = class_map['Crack']
    else:
        raise ValueError(f"Cannot determine class from path: {img_path}")


    # Preprocess and predict
    x = preprocess_image(img_path)
    prob = model.predict(x, verbose=0)[0][0]
    pred_class = int(prob > 0.5)

    y_true.append(true_class)
    y_pred.append(pred_class)

    label_pred = inv_class_map[pred_class]
    label_true = inv_class_map[true_class]

    img_name = os.path.basename(img_path)
    print(f"{img_name} | GT: {label_true} | Pred: {label_pred} | Prob: {prob:.4f}")

    # Save annotated copy
    new_name = f"GT_{label_true}_PRED_{label_pred}_{img_name}"
    img = Image.open(img_path)
    img.save(os.path.join(output_dir, new_name))

# --- Print Evaluation Stats ---
print("\n--- Evaluation Stats ---")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['No Crack', 'Crack']))