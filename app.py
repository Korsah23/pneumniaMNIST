import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Pneumonia X-ray Classifier", page_icon="🫁")


# ---- Rebuild the EXACT same architecture used in training ----
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)  # no need to redownload pretrained weights
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.maxpool = nn.Identity()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    state_dict = torch.load("pneumonia_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()

# ---- Same preprocessing as training ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class_names = ["Normal", "Pneumonia"]

st.title("🫁 Pneumonia Chest X-ray Classifier")
st.write("Upload a chest X-ray image to classify it as Normal or Pneumonia.")
st.caption("Research/demo tool only — not for clinical diagnosis.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    st.subheader(f"Prediction: **{class_names[pred_idx]}**")
    st.write(f"Confidence: {probs[pred_idx].item():.2%}")

    st.bar_chart({class_names[i]: probs[i].item() for i in range(len(class_names))})
