# Simple Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((self.label_emb(labels), noise), -1)
        return self.model(x).view(-1, 1, 28, 28)

# Instantiate
latent_dim = 100
num_classes = 10
generator = Generator(latent_dim, num_classes)

#app.py
import torch
import streamlit as st
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from model import Generator  # the same class from training

# Load model
latent_dim = 100
num_classes = 10
model = Generator(latent_dim, num_classes)
model.load_state_dict(torch.load("digit_generator.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit (0-9):", list(range(10)))

if st.button("Generate Images"):
    noise = torch.randn(5, latent_dim)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = model(noise, labels)

    grid = make_grid(images, nrow=5, normalize=True, pad_value=1)
    plt.figure(figsize=(10, 2))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    st.pyplot(plt)
