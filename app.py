import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


st.set_page_config(page_title="OCT Retinal Classifier")
st.title("OCT Retinal Image Classifier")
st.write("**Name:** Sourabh Raja")
st.write("Upload a retinal OCT image and the application will  classify it.")


class RetinalCNN(nn.Module):
    def __init__(self, num_classes=4, dropout_p=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))






@st.cache_resource
def load_model():
    model = RetinalCNN()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()
st.success(" Model loaded successfully... LETS START!!!")


uploaded = st.file_uploader("Upload an OCT image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  


    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100


    st.markdown("---")
    st.subheader(f"Prediction: Class {pred_idx}")
    st.progress(int(confidence))
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.markdown("#### All class probabilities:")
    for idx, prob in enumerate(probs):
        st.write(f"`Class {idx}`: {prob*100:.2f}%")
        st.progress(int(prob * 100))