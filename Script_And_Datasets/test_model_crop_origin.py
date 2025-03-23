from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
import matplotlib.pyplot as plt
from tabulate import tabulate

# Image dimensions
img_width, img_height = 224, 224

# Class names (must match training labels)
class_names = ['bb', 'bk', 'bn', 'bp', 'bq', 'br', 'empty', 'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

# Load trained model
model = load_model("saved_model.h5")

# Output file for saving predictions
output_file = "predictions_crop_origin.txt"

def prepare_img(img_file):
    """Load and preprocess an image."""
    try:
        img = load_img(img_file, target_size=(img_height, img_width))
        img_res = img_to_array(img)
        img_res = np.expand_dims(img_res, axis=0)
        img_res = img_res / 255.0  # Normalize
        return img_res
    except Exception as e:
        print(f"Error loading {img_file}: {e}")
        return None

def predict_class(img_class, save_dir="predictions_crop_model_origin_images"):
    """Predict the class for all images in a given folder and save results."""
    TEST_DIR = Path(f'original_chess_dataset/test/{img_class}')
    prediction = {class_name: 0 for class_name in class_names}

    if not TEST_DIR.exists():  # Handle missing directories
        print(f"Warning: {TEST_DIR} does not exist. Skipping...\n")
        return

    for img_path in TEST_DIR.iterdir():
        if img_path.is_file():
            img_for_model = prepare_img(img_path)
            if img_for_model is None:
                continue  # Skip invalid images
            
            res_arr = model.predict(img_for_model, batch_size=32, verbose=0)
            answer = np.argmax(res_arr, axis=1)
            text = class_names[answer[0]]
            prediction[text] += 1

    # Save text output
    with open(output_file, "a") as f:
        f.write(f"\n### Prediction Results for: {img_class} ###\n")
        f.write(tabulate(prediction.items(), headers=["Class", "Count"], tablefmt="grid"))
        f.write("\n\n")

    # Save bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(prediction.keys(), prediction.values(), color='skyblue')
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.title(f"Predictions for: {img_class}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create output folder for images
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{save_dir}/{img_class}_predictions.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved predictions for {img_class} to {output_file} and {save_path}")

# Clear the previous output file
open(output_file, "w").close()

# Run predictions for all classes
for img_class in class_names:
    predict_class(img_class)
