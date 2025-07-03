import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def is_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


def process_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, target_size)
    return np.array(image, dtype=np.float32) / 255.0


def load_dataset(dataset_path, batch_size=1000):
    skip_items = {
        "__pycache__",
        "plant-disease-detection",
        "frontend",
        "models",
        "saved_models",
        "training",
        ".py",
    }

    total_images = 0
    unique_labels = set()

    print("Counting images and collecting labels...")
    for folder in os.listdir(dataset_path):
        if folder in skip_items or folder.endswith(".py"):
            continue

        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
        total_images += len(image_files)
        unique_labels.add(folder)

    print(f"Total images to process: {total_images}")
    print(f"Number of classes: {len(unique_labels)}")

    X = np.zeros((total_images, 128, 128, 3), dtype=np.float32)
    y = np.zeros(total_images, dtype=object)

    current_idx = 0
    skipped = 0

    for folder in os.listdir(dataset_path):
        if folder in skip_items or folder.endswith(".py"):
            continue

        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {folder}")

        image_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
        for image_file in image_files:
            try:
                image_path = os.path.join(folder_path, image_file)
                processed_image = process_image(image_path)

                if processed_image is not None:
                    X[current_idx] = processed_image
                    y[current_idx] = folder
                    current_idx += 1

                    if current_idx % batch_size == 0:
                        print(f"Processed {current_idx}/{total_images} images...")
                else:
                    skipped += 1

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                skipped += 1
                continue

    print(f"\nTotal images skipped: {skipped}")
    print(f"Total images loaded: {current_idx}")

    X = X[:current_idx]
    y = y[:current_idx]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, le.classes_


if __name__ == "__main__":
    dataset_path = os.path.dirname(os.path.abspath(__file__))
    try:
        X_train, X_test, y_train, y_test, classes = load_dataset(dataset_path)
        print("\nDataset loaded successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Number of classes: {len(classes)}")
    except Exception as e:
        print(f"Error: {str(e)}")
