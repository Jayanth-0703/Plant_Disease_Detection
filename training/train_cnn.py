import os
import sys

# Add parent directory to path before any other imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from data_preparation import load_dataset
from models.cnn_model import create_cnn_model


physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


DATASET_PATH = r"c:\Users\Jayanth Kumar\Downloads\PlantVillage"
MODEL_SAVE_PATH = r"c:\Users\Jayanth Kumar\Downloads\PlantVillage\saved_models"

print("Loading dataset...")
X_train, X_test, y_train, y_test, classes = load_dataset(DATASET_PATH)


num_classes = len(classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


print("Creating model...")
input_shape = (128, 128, 3)
model = create_cnn_model(input_shape, num_classes)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, "best_model.h5"),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1
    ),
]


print("Starting training...")
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2,
    callbacks=callbacks,
    shuffle=True,
)


print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")


os.makedirs("saved_models", exist_ok=True)


model_save_path = os.path.join("saved_models", "best_model.h5")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
