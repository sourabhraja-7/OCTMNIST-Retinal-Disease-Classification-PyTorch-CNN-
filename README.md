# OCTMNIST-Retinal-Disease-Classification-PyTorch-CNN-

 A Convolutional Neural Network built from scratch using PyTorch to classify retinal OCT images into four disease categories.

---

## 📋 Overview

This project implements a **Convolutional Neural Network (CNN)** using PyTorch to classify retinal OCT images into four disease categories using the **OCTMNIST** dataset.

The dataset contains **109,309 retinal OCT images** (28×28 grayscale) across 4 classes, making it a real-world medical image classification task.

---

## ✨ Features

- 🧠 Custom CNN built from scratch using PyTorch
- 🎯 Achieved **>80% test accuracy**
- ⚖️ Class imbalance handling using **weighted loss**
- 🗺️ Feature map, activation, and kernel visualization
- 🚀 Model deployment using **Streamlit** web app

---

## 🏗️ Model Architecture

```
Conv → BatchNorm → ReLU → MaxPool
  └─→ Conv → BatchNorm → ReLU → MaxPool
        └─→ Conv → ReLU
              └─→ Fully Connected → Dropout → Output (4 classes)
```

| Component   | Value             |
|-------------|-------------------|
| Loss        | `CrossEntropyLoss` |
| Optimizer   | `Adam`             |

---

## 📊 Results

| Model        | Test Accuracy |
|--------------|---------------|
| Base CNN     | 79%           |
| Improved CNN | **>80%**      |

### Evaluation Includes:
- 📈 Accuracy & Loss plots
- 🟦 Confusion Matrix
- 📉 ROC Curve
- 🔍 Model Interpretability

---

## 🚀 Deployment

An interactive web app is built using **Streamlit**.

### Run Locally

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Launch the app:**
```bash
streamlit run app.py
```

**3. Upload an OCT image → Get a prediction instantly.**

---

## 🏋️ How to Run Training

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Open and run the training notebook:**
```bash
jupyter notebook octmnsit_classification.ipynb
```

---
