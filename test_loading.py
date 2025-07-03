import os
import sys

import psutil

from data_preparation import load_dataset


def print_memory_usage():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")


def test_dataset_loading(dataset_path, batch_size=1000, target_size=(128, 128)):
    print(f"Found dataset directory at: {dataset_path}")
    print("Initial memory usage:")
    print_memory_usage()

    try:
        print("\nStarting data loading...")
        X_train, X_test, y_train, y_test, classes = load_dataset(
            dataset_path, batch_size=batch_size
        )

        print("\nDataset loading successful!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Image shape: {X_train[0].shape}")
        print(f"Number of classes: {len(classes)}")
        print("\nClasses found:")
        for i, class_name in enumerate(classes):
            print(f"{i + 1}. {class_name}")

        print("\nMemory usage after loading:")
        print_memory_usage()

        return True

    except MemoryError:
        print("\nError: Not enough memory to load the dataset!")
        print("Solutions:")
        print(
            "1. Reduce image size (currently", target_size[0], "x", target_size[1], ")"
        )
        print("2. Decrease batch size (currently", batch_size, ")")
        print("3. Free up system memory")
        return False

    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        return False


if __name__ == "__main__":
    DATASET_PATH = r"c:\Users\Jayanth Kumar\Downloads\PlantVillage"

    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset path does not exist: {DATASET_PATH}")
        sys.exit(1)

    success = test_dataset_loading(DATASET_PATH)
    if not success:
        sys.exit(1)
