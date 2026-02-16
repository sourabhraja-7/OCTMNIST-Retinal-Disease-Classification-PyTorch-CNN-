# OCTMNIST-Retinal-Disease-Classification-PyTorch-CNN-

Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify retinal OCT images into four disease categories using the OCTMNIST dataset.

The dataset contains 109,309 retinal OCT images (28×28 grayscale) across 4 classes, making it a real-world medical image classification task. 


Features

• Custom CNN built from scratch using PyTorch
• Achieved >80% test accuracy
• Class imbalance handling using weighted loss
• Feature map, activation, and kernel visualization
• Model deployment using Streamlit web app

Model Architecture

CNN Architecture:

Conv → BatchNorm → ReLU → MaxPool
Conv → BatchNorm → ReLU → MaxPool
Conv → ReLU
Fully Connected → Dropout → Output (4 classes)

Loss: CrossEntropyLoss
Optimizer: Adam

Results
Model	Test Accuracy
Base CNN	79%
Improved CNN	>80%

Includes:

• Accuracy & Loss plots
• Confusion Matrix
• ROC Curve
• Model Interpretability

Deployment

Interactive web app built using Streamlit.

Run locally:

pip install -r requirements.txt
streamlit run app.py


Upload an OCT image → Get prediction instantly.

How to Run Training
pip install -r requirements.txt
octmnsit_classification.ipynb
