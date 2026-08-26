import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(page_title="Pneumonia X-ray Classifier", page_icon="🫁")


# ---- Must match the PneumoniaModel class from your notebook EXACTLY ----
class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.ReLu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ReLu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ReLu(x)
        x = self.pool3(x)

        x = self.drop(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ---- Rebuild the EXACT same architecture used in training ----
@st.cache_resource
def load_model():
    model = PneumoniaModel()
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
