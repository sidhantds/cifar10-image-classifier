import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Same architecture as in ImageClassifierCNN.ipynb
class NeuralNet(nn.Module):

  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 12, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(12, 24, 5)
    self.fc1 = nn.Linear(24*5*5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


@st.cache_resource
def load_model():
  net = NeuralNet()
  net.load_state_dict(torch.load('trained_net.pth', map_location='cpu'))
  net.eval()
  return net


st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image and the CNN will predict which of these classes it belongs to: "
         + ", ".join(class_names) + ".")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp", "bmp"])

if uploaded is not None:
  image = Image.open(uploaded).convert("RGB")
  st.image(image, caption="Uploaded image", width="stretch")

  net = load_model()
  with torch.no_grad():
    output = net(transform(image).unsqueeze(0))
    probs = F.softmax(output, dim=1)[0]

  top = int(probs.argmax())
  st.success(f"Prediction: **{class_names[top]}** ({probs[top].item():.1%} confidence)")

  st.subheader("Class probabilities")
  st.bar_chart({"probability": {name: probs[i].item() for i, name in enumerate(class_names)}})
