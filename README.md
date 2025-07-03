# PlantVillage Disease Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🌱 Project Overview

PlantVillage Disease Detection is a deep learning project for classifying plant diseases from leaf images using Convolutional Neural Networks (CNNs). The project leverages the PlantVillage dataset and provides scripts for data preparation, model training, evaluation, and prediction.

---

## ✨ Features
- Image preprocessing and dataset loading
- Custom CNN and improved ResNet-like model architectures
- Training with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- Prediction script for new images
- Memory usage monitoring for large datasets
- Modular code structure

---

## 📁 Project Structure

```
PlantVillage/
├── data_preparation.py         # Data loading and preprocessing
├── models/
│   └── cnn_model.py           # Model architectures
├── predict.py                 # Prediction script
├── test_loading.py            # Dataset loading & memory test
├── training/
│   └── train_cnn.py           # Model training script
├── saved_models/              # Trained models
├── requirements.txt           # Python dependencies
├── architecture.md            # Architecture diagram (Mermaid)
├── TODO.md                    # Project TODOs
└── README.md                  # Project documentation
```

---

## 🏗️ Architecture

See [architecture.md](architecture.md) for a detailed Mermaid diagram.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PlantVillage.git
cd PlantVillage
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
- Download the PlantVillage dataset and place it in the project directory or update the path in scripts.

### 4. Test Data Loading
```bash
python test_loading.py
```

### 5. Train the Model
```bash
python training/train_cnn.py
```

### 6. Make Predictions
```bash
python predict.py
```

---

## 🧩 Dynamic Effects
- [x] Badges for Python, TensorFlow, License, and Status
- [x] Mermaid diagram for architecture
- [x] Modular and extensible codebase

---

## 📝 TODO
See [TODO.md](TODO.md) for planned features and improvements.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
This project is licensed under the MIT License.
