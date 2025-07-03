import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define class names at module level
class_names = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    image = cv2.resize(image, (128, 128))

    image = np.array(image, dtype=np.float32) / 255.0

    image = np.expand_dims(image, axis=0)
    return image


def predict_disease(model_path, image_path, class_names):
    model = load_model(model_path)

    processed_image = load_and_preprocess_image(image_path)

    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]

    return class_names[predicted_class_index], confidence


if __name__ == "__main__":
    # Use absolute path for model
    MODEL_PATH = os.path.join(SCRIPT_DIR, "saved_models", "best_model.h5")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please ensure the model file exists in the saved_models directory.")
        exit(1)

    test_image_path = input("Enter the path to the plant image: ")

    # Convert relative path to absolute path if necessary
    if not os.path.isabs(test_image_path):
        test_image_path = os.path.abspath(test_image_path)

    if not os.path.exists(test_image_path):
        print(f"Error: Image file not found at {test_image_path}")
        print("Please check the file path and try again.")
        exit(1)

    try:
        predicted_class, confidence = predict_disease(
            MODEL_PATH, test_image_path, class_names
        )
        print(f"\nPredicted Disease: {predicted_class.replace('_', ' ')}")
        print(f"Confidence: {confidence * 100:.2f}%")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        print(
            "Please ensure both the model and image files are accessible and in the correct format."
        )
